
from pathlib import Path
from hashlib import sha256

import chromadb

from ai_knowledge_mcp_server.document_processor import read_document
from ai_knowledge_mcp_server.text_chunker import chunk_text
from ai_knowledge_mcp_server.embedding_service import (
    MODEL_NAME,
    EMBEDDING_DIMENSION,
    get_embedding_service,
)


# ---------------------------------
# EVALUATION CONFIGURATION
# ---------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "documents" / "retrieval_tests"
DATABASE_DIR = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "knowledge_bge_m3_eval_v1"

TEST_DOCUMENTS = [
    "mcp.txt",
    "rag.txt",
    "computer_vision_ar.txt",
    "ros2.txt",
]

TEST_QUERIES = [
    {
        "query": "How do AI agents access external tools?",
        "relevant": ["mcp.txt"],
    },
    {
        "query": "كيف نسترجع معلومات لتزويد نموذج اللغة بسياق؟",
        "relevant": ["rag.txt"],
    },
    {
        "query": "What model can detect objects in images?",
        "relevant": ["computer_vision_ar.txt"],
    },
    {
        "query": "كيف تتواصل مكونات الروبوت مع بعضها؟",
        "relevant": ["ros2.txt"],
    },
]


# ---------------------------------
# PREPARE DOCUMENTS
# ---------------------------------

def prepare_documents():
    """Read and chunk the evaluation documents."""

    records = []

    for filename in TEST_DOCUMENTS:
        path = TEST_DIR / filename
        result = read_document(str(path))

        if not result["success"]:
            raise RuntimeError(
                f"Failed to read {filename}: {result['error']}"
            )

        chunks = chunk_text(
            result["content"],
            chunk_size=500,
            chunk_overlap=100,
        )

        if not chunks:
            raise RuntimeError(
                f"No chunks generated for {filename}"
            )

        document_hash = sha256(
            filename.encode("utf-8")
        ).hexdigest()[:16]

        for index, chunk in enumerate(chunks):
            records.append({
                "id": f"{document_hash}_chunk_{index}",
                "content": chunk,
                "metadata": {
                    "filename": filename,
                    "chunk_index": index,
                    "embedding_model": MODEL_NAME,
                },
            })

        print(
            f"Prepared: {filename} "
            f"({len(chunks)} chunks)"
        )

    return records


# ---------------------------------
# CREATE ISOLATED COLLECTION
# ---------------------------------

def create_evaluation_collection(client):
    """Create a dedicated evaluation collection."""

    # Never delete or modify the production collection.
    # This evaluation namespace is separate.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "embedding_model": MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "hnsw:space": "cosine",
        },
    )


# ---------------------------------
# INDEX DOCUMENTS
# ---------------------------------

def index_documents(collection, records, service):
    """Generate embeddings and index test records."""

    texts = [
        record["content"]
        for record in records
    ]

    embeddings = service.encode(texts)

    collection.upsert(
        ids=[
            record["id"]
            for record in records
        ],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            record["metadata"]
            for record in records
        ],
    )

    print(
        "\nEvaluation collection:",
        collection.name,
    )

    print(
        "Stored chunks:",
        collection.count(),
    )


# ---------------------------------
# RETRIEVAL EVALUATION
# ---------------------------------

def evaluate(collection, service):
    """Evaluate multilingual retrieval quality."""

    total_queries = len(TEST_QUERIES)

    top1_hits = 0
    recall3_sum = 0.0
    reciprocal_rank_sum = 0.0

    query_texts = [
        test["query"]
        for test in TEST_QUERIES
    ]

    query_embeddings = service.encode(query_texts)

    print("\n" + "=" * 55)
    print("MULTILINGUAL RETRIEVAL EVALUATION")
    print("=" * 55)

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        relevant = set(test["relevant"])

        results = collection.query(
            query_embeddings=[query_embeddings[i]],
            n_results=min(3, collection.count()),
            include=["metadatas", "distances"],
        )

        metadata_list = results["metadatas"][0]
        distance_list = results["distances"][0]

        ranked_documents = [
            metadata["filename"]
            for metadata in metadata_list
        ]

        relevant_ranks = [
            rank
            for rank, filename in enumerate(
                ranked_documents,
                start=1,
            )
            if filename in relevant
        ]

        first_rank = (
            min(relevant_ranks)
            if relevant_ranks
            else None
        )

        if first_rank == 1:
            top1_hits += 1

        retrieved_relevant = relevant.intersection(
            ranked_documents
        )

        recall3_sum += (
            len(retrieved_relevant) / len(relevant)
        )

        if first_rank is not None:
            reciprocal_rank_sum += 1.0 / first_rank

        print(f"\nTest {i + 1}")
        print("Query:", query)
        print("Relevant documents:", sorted(relevant))

        print("Ranked results:")

        for rank, (metadata, distance) in enumerate(
            zip(metadata_list, distance_list),
            start=1,
        ):
            filename = metadata["filename"]

            marker = (
                "MATCH"
                if filename in relevant
                else "-"
            )

            print(
                f"  {rank}. {filename}"
                f" | distance={distance:.4f}"
                f" | {marker}"
            )

        print(
            "First relevant rank:",
            first_rank if first_rank else "Not found",
        )

    top1_accuracy = top1_hits / total_queries
    recall_at_3 = recall3_sum / total_queries
    mrr_at_3 = reciprocal_rank_sum / total_queries

    print("\n" + "=" * 55)
    print("FINAL EVALUATION RESULTS")
    print("=" * 55)

    print(f"Queries evaluated: {total_queries}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2%}")
    print(f"Recall@3: {recall_at_3:.2%}")
    print(f"MRR@3: {mrr_at_3:.4f}")

    return {
        "top1_accuracy": top1_accuracy,
        "recall_at_3": recall_at_3,
        "mrr_at_3": mrr_at_3,
    }


# ---------------------------------
# MAIN
# ---------------------------------

def main():
    print("=" * 55)
    print("BGE-M3 ISOLATED RETRIEVAL EVALUATION")
    print("=" * 55)

    records = prepare_documents()

    service = get_embedding_service()

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(DATABASE_DIR)
    )

    collection = create_evaluation_collection(client)

    # Avoid contaminating repeated evaluations:
    # require a fresh evaluation namespace.
    if collection.count() != 0:
        raise RuntimeError(
            "Evaluation collection is not empty. "
            "Use a new evaluation collection name "
            "for a clean evaluation."
        )

    index_documents(collection, records, service)

    evaluate(collection, service)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()

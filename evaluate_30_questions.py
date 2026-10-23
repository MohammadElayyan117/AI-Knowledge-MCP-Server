
import json
import time
from pathlib import Path

import chromadb

from ai_knowledge_mcp_server.document_processor import read_document
from ai_knowledge_mcp_server.text_chunker import chunk_text
from ai_knowledge_mcp_server.embedding_service import (
    MODEL_NAME,
    get_embedding_service,
)


ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents" / "retrieval_tests"
REPORT_DIR = ROOT / "reports"

DOCUMENTS = [
    "mcp.txt",
    "rag.txt",
    "computer_vision_ar.txt",
    "ros2.txt",
]

# Development evaluation: 24 answerable questions.
# Each question has a known relevant document.
TESTS = [
    # MCP
    ("What is Model Context Protocol?", "mcp.txt"),
    ("How can AI applications use external tools?", "mcp.txt"),
    ("What do MCP servers expose?", "mcp.txt"),
    ("How are AI clients connected to data sources?", "mcp.txt"),
    ("كيف يربط MCP تطبيقات الذكاء الاصطناعي بالأدوات؟", "mcp.txt"),
    ("ما وظيفة الأدوات والموارد في MCP؟", "mcp.txt"),

    # RAG
    ("What is retrieval augmented generation?", "rag.txt"),
    ("Why are documents split into chunks?", "rag.txt"),
    ("How do embeddings support retrieval?", "rag.txt"),
    ("How does retrieved context help an LLM?", "rag.txt"),
    ("كيف نستخدم قاعدة بيانات متجهية مع نموذج لغة؟", "rag.txt"),
    ("كيف يعمل استرجاع المعلومات قبل توليد الإجابة؟", "rag.txt"),

    # Computer Vision
    ("What is computer vision?", "computer_vision_ar.txt"),
    ("Which AI model detects objects in images?", "computer_vision_ar.txt"),
    ("How can computers analyze video?", "computer_vision_ar.txt"),
    ("Where is image-based object detection used?", "computer_vision_ar.txt"),
    ("ما استخدام YOLO في تحليل الصور؟", "computer_vision_ar.txt"),
    ("كيف تستخدم الروبوتات الرؤية الحاسوبية؟", "computer_vision_ar.txt"),

    # ROS2
    ("What is Robot Operating System 2?", "ros2.txt"),
    ("How do robotics nodes communicate?", "ros2.txt"),
    ("What are ROS2 topics and services?", "ros2.txt"),
    ("What supports modular robotic software?", "ros2.txt"),
    ("كيف تتبادل عقد ROS2 الرسائل؟", "ros2.txt"),
    ("ما دور Topics وActions في ROS2؟", "ros2.txt"),

    # Unanswerable from the supplied documents
    ("What is the capital of Brazil?", None),
    ("Who won the 2018 football World Cup?", None),
    ("What is the boiling point of ethanol?", None),
    ("كم يبلغ عدد سكان اليابان؟", None),
    ("من كتب رواية البؤساء؟", None),
    ("What is the current price of gold?", None),
]


def load_documents():
    texts = []
    filenames = []

    for filename in DOCUMENTS:
        result = read_document(str(DOCUMENTS_DIR / filename))

        if not result["success"]:
            raise RuntimeError(
                f"Document error: {filename}: {result['error']}"
            )

        chunks = chunk_text(
            result["content"],
            chunk_size=500,
            chunk_overlap=100,
        )

        if not chunks:
            raise RuntimeError(f"Empty document: {filename}")

        for chunk in chunks:
            texts.append(chunk)
            filenames.append(filename)

    return texts, filenames


def main():
    print("BGE-M3 DEVELOPMENT EVALUATION — 30 QUESTIONS")
    print("=" * 55)

    start = time.perf_counter()

    # In-memory collection: independent of stored ChromaDB.
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="development_eval_30",
        configuration={"hnsw": {"space": "cosine"}},
    )

    service = get_embedding_service()

    texts, filenames = load_documents()
    embeddings = service.encode(texts)

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(texts))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"filename": name}
            for name in filenames
        ],
    )

    print("Model:", MODEL_NAME)
    print("Indexed chunks:", collection.count())
    print("Questions:", len(TESTS))

    queries = [question for question, _ in TESTS]
    query_embeddings = service.encode(queries)

    records = []
    hits_at_1 = 0
    hits_at_3 = 0
    reciprocal_rank_sum = 0.0
    answerable_count = 0
    negative_distances = []

    for index, (question, expected) in enumerate(TESTS):
        response = collection.query(
            query_embeddings=[query_embeddings[index]],
            n_results=min(3, collection.count()),
            include=["metadatas", "distances"],
        )

        ranked = [
            metadata["filename"]
            for metadata in response["metadatas"][0]
        ]

        distances = [
            float(d)
            for d in response["distances"][0]
        ]

        rank = (
            ranked.index(expected) + 1
            if expected is not None and expected in ranked
            else None
        )

        if expected is not None:
            answerable_count += 1

            if rank == 1:
                hits_at_1 += 1

            if rank is not None:
                hits_at_3 += 1
                reciprocal_rank_sum += 1.0 / rank
        elif distances:
            negative_distances.append(distances[0])

        record = {
            "question": question,
            "expected": expected,
            "ranked_documents": ranked,
            "distances": distances,
            "relevant_rank": rank,
            "answerable": expected is not None,
        }
        records.append(record)

        print(f"\nQuestion {index + 1}: {question}")
        print("Expected:", expected or "UNANSWERABLE")
        print("Retrieved:", ranked)
        print("Distances:", [round(d, 4) for d in distances])

        if expected is not None:
            print("Relevant rank:", rank or "NOT FOUND")
        else:
            print("Negative-case diagnostic only")

    metrics = {
        "top1_accuracy": hits_at_1 / answerable_count,
        "recall_at_3": hits_at_3 / answerable_count,
        "mrr_at_3": reciprocal_rank_sum / answerable_count,
    }

    duration = time.perf_counter() - start

    report = {
        "model": MODEL_NAME,
        "dataset_type": "small_synthetic_development",
        "document_count": len(DOCUMENTS),
        "indexed_chunks": collection.count(),
        "total_questions": len(TESTS),
        "answerable_questions": answerable_count,
        "negative_questions": len(TESTS) - answerable_count,
        "retrieval_metrics": metrics,
        "negative_top1_distances": negative_distances,
        "total_runtime_seconds": round(duration, 2),
        "results": records,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "development_eval_30.json"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)

    print("Top-1 Accuracy:", f"{metrics['top1_accuracy']:.2%}")
    print("Recall@3:", f"{metrics['recall_at_3']:.2%}")
    print("MRR@3:", f"{metrics['mrr_at_3']:.4f}")
    print("Total runtime:", round(duration, 2), "seconds")
    print("Report saved:", report_path)

    print(
        "\nNote: Negative queries are diagnostic only. "
        "No abstention classifier has been evaluated."
    )


if __name__ == "__main__":
    main()

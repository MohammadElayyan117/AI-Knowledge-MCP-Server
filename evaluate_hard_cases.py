
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
DOCS_DIR = ROOT / "documents" / "retrieval_tests"
REPORT_DIR = ROOT / "reports"

DOCUMENTS = [
    "mcp.txt",
    "rag.txt",
    "computer_vision_ar.txt",
    "ros2.txt",
]

# Positive cases may have multiple relevant documents.
POSITIVE_CASES = [
    {
        "question": "How do assistants access external systems?",
        "relevant": ["mcp.txt"],
    },
    {
        "question": "كيف يحصل نموذج اللغة على سياق من المستندات؟",
        "relevant": ["rag.txt"],
    },
    {
        "question": "How does a robot identify objects in camera images?",
        "relevant": ["computer_vision_ar.txt"],
    },
    {
        "question": "كيف تتواصل عقد الروبوت عبر المواضيع والخدمات؟",
        "relevant": ["ros2.txt"],
    },
    {
        "question": "What AI-related technologies are used in robotics?",
        "relevant": [
            "computer_vision_ar.txt",
            "ros2.txt",
        ],
    },
    {
        "question": "How can AI applications work with external data?",
        "relevant": [
            "mcp.txt",
            "rag.txt",
        ],
    },
    {
        "question": "Which system provides tools and resources to AI clients?",
        "relevant": ["mcp.txt"],
    },
    {
        "question": "كيف تُستخدم embeddings لاسترجاع المعلومات؟",
        "relevant": ["rag.txt"],
    },
]

# Questions related to the domain but whose specific
# answers are NOT present in the evaluation documents.
NEGATIVE_CASES = [
    "Which OAuth version does our MCP server implement?",
    "What is the exact authentication token lifetime?",
    "Which YOLO version achieves the highest FPS?",
    "ما إصدار DDS المستخدم في إعدادات ROS2 عندنا؟",
    "What chunking algorithm gives the best Recall@5?",
    "ما حجم الذاكرة المطلوبة لتشغيل ROS2 على Jetson؟",
    "Which encryption algorithm protects our MCP database?",
    "ما دقة نموذج YOLO المستخدم في مشروع الروبوت؟",
]


def create_collection(service):
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="hard_evaluation",
        configuration={
            "hnsw": {"space": "cosine"}
        },
    )

    texts = []
    metadata = []

    for filename in DOCUMENTS:
        result = read_document(
            str(DOCS_DIR / filename)
        )

        if not result["success"]:
            raise RuntimeError(
                f"{filename}: {result['error']}"
            )

        chunks = chunk_text(
            result["content"],
            chunk_size=500,
            chunk_overlap=100,
        )

        if not chunks:
            raise RuntimeError(
                f"No chunks: {filename}"
            )

        for index, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append({
                "filename": filename,
                "chunk_index": index,
            })

    vectors = service.encode(texts)

    collection.add(
        ids=[f"eval_{i}" for i in range(len(texts))],
        embeddings=vectors,
        documents=texts,
        metadatas=metadata,
    )

    return collection


def run_search(collection, vector, top_k=3):
    start = time.perf_counter()

    results = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, collection.count()),
        include=["metadatas", "distances"],
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    return [
        {
            "filename": metadata["filename"],
            "distance": float(distance),
        }
        for metadata, distance in zip(
            results["metadatas"][0],
            results["distances"][0],
        )
    ], latency_ms


def evaluate():
    print("=" * 60)
    print("BGE-M3 HARD RETRIEVAL EVALUATION")
    print("=" * 60)

    service = get_embedding_service()
    collection = create_collection(service)

    cases = POSITIVE_CASES + [
        {
            "question": question,
            "relevant": [],
        }
        for question in NEGATIVE_CASES
    ]

    vectors = service.encode([
        case["question"] for case in cases
    ])

    records = []

    precision_at_1_hits = 0
    recall_at_3_sum = 0.0
    mrr_at_3_sum = 0.0

    for index, case in enumerate(cases):
        results, latency_ms = run_search(
            collection,
            vectors[index],
        )

        relevant = set(case["relevant"])
        ranked = [r["filename"] for r in results]

        matching_ranks = [
            rank
            for rank, filename in enumerate(
                ranked, start=1
            )
            if filename in relevant
        ]

        if relevant:
            precision_at_1_hits += int(
                bool(ranked) and ranked[0] in relevant
            )

            recall_at_3_sum += (
                len(relevant.intersection(ranked))
                / len(relevant)
            )

            if matching_ranks:
                mrr_at_3_sum += (
                    1 / min(matching_ranks)
                )

        record = {
            "question": case["question"],
            "relevant_documents": sorted(relevant),
            "type": (
                "answerable"
                if relevant else "unanswerable"
            ),
            "results": results,
            "latency_chroma_query_ms": round(
                latency_ms, 3
            ),
        }

        records.append(record)

        print(f"\nCase {index + 1}")
        print("Question:", case["question"])
        print("Relevant:", sorted(relevant) or "NONE")

        for rank, match in enumerate(
            results, start=1
        ):
            print(
                f"  {rank}. {match['filename']} "
                f"| distance={match['distance']:.4f}"
            )

    n = len(POSITIVE_CASES)

    metrics = {
        "top1_hit_rate": precision_at_1_hits / n,
        "recall_at_3": recall_at_3_sum / n,
        "mrr_at_3": mrr_at_3_sum / n,
    }

    report = {
        "model": MODEL_NAME,
        "dataset": "hard_development_v1",
        "positive_cases": n,
        "negative_cases": len(NEGATIVE_CASES),
        "stored_chunks": collection.count(),
        "metrics_positive_only": metrics,
        "negative_cases_diagnostic_only": True,
        "results": records,
    }

    REPORT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    output = (
        REPORT_DIR / "hard_evaluation_v1.json"
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Top-1 Hit Rate: "
        f"{metrics['top1_hit_rate']:.2%}"
    )
    print(
        f"Recall@3: "
        f"{metrics['recall_at_3']:.2%}"
    )
    print(
        f"MRR@3: "
        f"{metrics['mrr_at_3']:.4f}"
    )
    print(
        "Negative cases:",
        len(NEGATIVE_CASES),
        "(diagnostic only)"
    )
    print("Report:", output)
    print(
        "No abstention mechanism is evaluated yet."
    )


if __name__ == "__main__":
    evaluate()

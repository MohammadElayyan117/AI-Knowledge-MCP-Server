
import uuid

from ai_knowledge_mcp_server.vector_store import VectorStore


def main():
    store = VectorStore()

    # Unique test filename prevents overwriting real documents.
    filename = f"reindex_test_{uuid.uuid4().hex}.txt"

    print("Testing document:", filename)

    original_chunks = [
        "Model Context Protocol connects AI applications to tools.",
        "Retrieval augmented generation uses external knowledge.",
        "Computer vision processes images and video.",
    ]

    first = store.add_document(filename, original_chunks)

    print("Initial indexing:", first)

    if not first["success"]:
        raise RuntimeError("Initial indexing failed.")

    document_id = store._document_id(filename)

    before = store.collection.get(
        where={"document_id": document_id}
    )

    assert len(before["ids"]) == 3, (
        "Expected three chunks before reindexing."
    )

    updated_chunks = [
        "Updated document about Model Context Protocol."
    ]

    second = store.add_document(filename, updated_chunks)

    print("Reindexing:", second)

    if not second["success"]:
        raise RuntimeError("Reindexing failed.")

    after = store.collection.get(
        where={"document_id": document_id}
    )

    assert len(after["ids"]) == 1, (
        "Old chunks were not removed correctly."
    )

    assert after["documents"] == updated_chunks, (
        "Stored document content was not updated."
    )

    print("PASS: Old chunks removed successfully.")
    print("PASS: Updated content stored correctly.")
    print("REINDEXING TEST PASSED")

    # Intentionally leave the test record for inspection.
    # No existing user documents are modified.


if __name__ == "__main__":
    main()

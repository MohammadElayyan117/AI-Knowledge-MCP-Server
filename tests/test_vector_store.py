
import hashlib

import pytest

import ai_knowledge_mcp_server.vector_store as vector_module


class FakeEmbeddingService:
    """Deterministic embeddings for isolated unit tests."""

    def encode(self, texts):
        vectors = []

        for text in texts:
            digest = hashlib.sha256(
                text.encode("utf-8")
            ).digest()

            vector = [
                float(digest[i % len(digest)]) / 255.0
                for i in range(1024)
            ]

            vectors.append(vector)

        return vectors

    def encode_query(self, query):
        return self.encode([query])[0]


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Create an isolated ChromaDB database for each test."""

    test_database_dir = tmp_path / "chroma_db"

    monkeypatch.setattr(
        vector_module,
        "DATABASE_DIR",
        test_database_dir,
    )

    monkeypatch.setattr(
        vector_module,
        "get_embedding_service",
        lambda: FakeEmbeddingService(),
    )

    return vector_module.VectorStore()


def test_empty_store(store):
    assert store.count() == 0


def test_add_document(store):
    result = store.add_document(
        "example.txt",
        ["First chunk", "Second chunk"],
    )

    assert result["success"] is True
    assert result["chunks_stored"] == 2
    assert store.count() == 2


def test_empty_chunks_rejected(store):
    result = store.add_document("empty.txt", [])

    assert result["success"] is False
    assert store.count() == 0


def test_reindex_removes_stale_chunks(store):
    filename = "updated.txt"

    first = store.add_document(
        filename,
        ["First", "Second", "Third"],
    )

    assert first["success"] is True
    assert store.count() == 3

    second = store.add_document(
        filename,
        ["Updated content"],
    )

    assert second["success"] is True
    assert store.count() == 1

    document_id = store._document_id(filename)

    records = store.collection.get(
        where={"document_id": document_id}
    )

    assert len(records["ids"]) == 1
    assert records["documents"] == ["Updated content"]


def test_empty_search_returns_no_results(store):
    result = store.search("AI")

    assert result["success"] is True
    assert result["results"] == []


def test_search_retrieves_document(store):
    text = "MCP connects AI applications to tools."

    added = store.add_document(
        "example.txt",
        [text],
    )

    assert added["success"] is True

    result = store.search(
        text,
        n_results=1,
    )

    assert result["success"] is True
    assert len(result["results"]) == 1

    assert (
        result["results"][0]["metadata"]["filename"]
        == "example.txt"
    )

    assert result["results"][0]["content"] == text


def test_invalid_search_query(store):
    result = store.search("")

    assert result["success"] is False


def test_invalid_result_count(store):
    result = store.search("AI", n_results=0)

    assert result["success"] is False

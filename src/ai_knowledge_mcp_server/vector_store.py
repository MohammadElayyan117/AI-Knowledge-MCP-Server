
from hashlib import sha256
from pathlib import Path

import chromadb

from ai_knowledge_mcp_server.embedding_service import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    get_embedding_service,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_DIR = PROJECT_ROOT / "data" / "chroma_db"

# New collection: preserves the old MiniLM data.
COLLECTION_NAME = "knowledge_bge_m3_v1"


class VectorStore:
    """Persistent multilingual vector store using BGE-M3."""

    def __init__(self):
        DATABASE_DIR.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(DATABASE_DIR)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "embedding_model": MODEL_NAME,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "hnsw:space": "cosine",
            },
        )

        metadata = self.collection.metadata or {}

        if (
            metadata.get("embedding_model") != MODEL_NAME
            or metadata.get("embedding_dimension")
            != EMBEDDING_DIMENSION
        ):
            raise RuntimeError(
                "Collection embedding configuration mismatch."
            )

    @staticmethod
    def _document_id(filename: str) -> str:
        """Create a stable identifier for each filename."""
        return sha256(
            filename.encode("utf-8")
        ).hexdigest()[:24]

    def add_document(
        self,
        filename: str,
        chunks: list[str],
    ) -> dict:
        """Index a document using BGE-M3 embeddings."""

        if not filename.strip():
            return {
                "success": False,
                "error": "Filename cannot be empty.",
            }

        if not chunks or any(
            not isinstance(chunk, str) or not chunk.strip()
            for chunk in chunks
        ):
            return {
                "success": False,
                "error": "Chunks must be non-empty strings.",
            }

        embeddings = get_embedding_service().encode(chunks)

        document_id = self._document_id(filename)

        ids = [
            f"{document_id}_chunk_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "filename": filename,
                "document_id": document_id,
                "chunk_index": i,
                "embedding_model": MODEL_NAME,
            }
            for i in range(len(chunks))
        ]

        # Preserve previous records until new embeddings
        # have been generated successfully.
        previous_records = self.collection.get(
            where={"document_id": document_id}
        )
        previous_ids = set(previous_records["ids"])

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        # Remove stale chunks if the document became shorter.
        stale_ids = list(previous_ids - set(ids))

        if stale_ids:
            self.collection.delete(ids=stale_ids)

        return {
            "success": True,
            "filename": filename,
            "chunks_stored": len(chunks),
            "collection": COLLECTION_NAME,
            "embedding_model": MODEL_NAME,
        }

    def search(
        self,
        query: str,
        n_results: int = 3,
    ) -> dict:
        """Search documents using multilingual embeddings."""

        if not isinstance(query, str) or not query.strip():
            return {
                "success": False,
                "error": "Search query cannot be empty.",
            }

        if not isinstance(n_results, int) or n_results < 1:
            return {
                "success": False,
                "error": "n_results must be positive.",
            }

        total_chunks = self.collection.count()

        if total_chunks == 0:
            return {
                "success": True,
                "results": [],
            }

        query_embedding = (
            get_embedding_service().encode_query(query)
        )

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, total_chunks),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        matches = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents, metadatas, distances
        ):
            matches.append({
                "content": document,
                "metadata": metadata,
                "distance": float(distance),
            })

        return {
            "success": True,
            "results": matches,
        }

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return self.collection.count()

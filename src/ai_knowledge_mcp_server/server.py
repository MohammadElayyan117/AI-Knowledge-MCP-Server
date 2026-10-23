
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from ai_knowledge_mcp_server.document_processor import read_document
from ai_knowledge_mcp_server.text_chunker import chunk_text
from ai_knowledge_mcp_server.vector_store import VectorStore


# -----------------------------
# SERVER CONFIGURATION
# -----------------------------

mcp = MCPServer("AI Knowledge MCP Server")

DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "documents"


# -----------------------------
# INTERNAL HELPERS
# -----------------------------

def get_document_path(filename: str) -> Path:
    """Validate that a file is inside the documents directory."""

    documents_dir = DOCUMENTS_DIR.resolve()

    if (
        not filename
        or filename in {".", ".."}
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or ":" in filename
    ):
        raise ValueError("Provide a valid filename only, not a path.")

    file_path = (documents_dir / filename).resolve()

    if not file_path.is_relative_to(documents_dir):
        raise ValueError(
            "Access outside documents directory is not allowed."
        )

    return file_path


# -----------------------------
# MCP TOOLS
# -----------------------------

@mcp.tool()
def server_info() -> dict:
    """Return basic information about the MCP server."""
    return {
        "name": "AI Knowledge MCP Server",
        "version": "0.1.0",
        "status": "running",
        "mode": "local",
    }


@mcp.tool()
def count_words(text: str) -> dict:
    """Count words and characters in a text."""
    return {
        "words": len(text.split()),
        "characters": len(text),
    }


@mcp.tool()
def read_knowledge_document(filename: str) -> dict:
    """Read a PDF, TXT, or Markdown document."""

    try:
        file_path = get_document_path(filename)
    except ValueError as error:
        return {"success": False, "error": str(error)}

    return read_document(str(file_path))


@mcp.tool()
def chunk_knowledge_document(
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> dict:
    """Split a document into overlapping text chunks."""

    result = read_knowledge_document(filename)

    if not result["success"]:
        return result

    try:
        chunks = chunk_text(
            result["content"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ValueError as error:
        return {"success": False, "error": str(error)}

    return {
        "success": True,
        "filename": filename,
        "total_chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks": [
            {
                "chunk_id": index,
                "content": chunk,
            }
            for index, chunk in enumerate(chunks)
        ],
    }


@mcp.tool()
def index_knowledge_document(filename: str) -> dict:
    """Index a local document into the ChromaDB vector database."""

    result = read_knowledge_document(filename)

    if not result["success"]:
        return result

    try:
        chunks = chunk_text(
            result["content"],
            chunk_size=500,
            chunk_overlap=100,
        )

        store = VectorStore()

        return store.add_document(
            filename=filename,
            chunks=chunks,
        )

    except (ValueError, RuntimeError, OSError) as error:
        return {
            "success": False,
            "error": str(error),
        }


@mcp.tool()
def search_knowledge(
    query: str,
    n_results: int = 3,
) -> dict:
    """Search indexed documents using semantic similarity."""

    try:
        store = VectorStore()

        return store.search(
            query=query,
            n_results=n_results,
        )

    except (ValueError, RuntimeError, OSError) as error:
        return {
            "success": False,
            "error": str(error),
        }


# -----------------------------
# MCP RESOURCES
# -----------------------------

@mcp.resource("knowledge://status")
def knowledge_status() -> str:
    """Return the current knowledge base status."""

    store = VectorStore()

    return (
        f"AI Knowledge Base Status\n"
        f"Stored chunks: {store.count()}\n"
        f"Storage mode: Local ChromaDB"
    )


# -----------------------------
# MCP PROMPTS
# -----------------------------

@mcp.prompt()
def summarize_document(text: str) -> str:
    """Generate a prompt to summarize a document."""

    return (
        "Summarize the following document and highlight "
        "its key ideas and important details:\n\n"
        f"{text}"
    )


# -----------------------------
# SERVER ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    mcp.run()

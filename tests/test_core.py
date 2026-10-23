
import pytest

from ai_knowledge_mcp_server.document_processor import read_document
from ai_knowledge_mcp_server.text_chunker import chunk_text


# ---------------------------------
# DOCUMENT PROCESSING TESTS
# ---------------------------------

def test_read_valid_txt(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("Hello MCP Server", encoding="utf-8")

    result = read_document(str(path))

    assert result["success"] is True
    assert result["words"] == 3
    assert result["filename"] == "sample.txt"


def test_read_valid_markdown(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text("# AI Knowledge", encoding="utf-8")

    result = read_document(str(path))

    assert result["success"] is True
    assert "AI Knowledge" in result["content"]


def test_missing_document(tmp_path):
    result = read_document(
        str(tmp_path / "missing.txt")
    )

    assert result["success"] is False


def test_unsupported_extension(tmp_path):
    path = tmp_path / "program.exe"
    path.write_text("test", encoding="utf-8")

    result = read_document(str(path))

    assert result["success"] is False


def test_empty_document(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    result = read_document(str(path))

    assert result["success"] is False


# ---------------------------------
# TEXT CHUNKING TESTS
# ---------------------------------

def test_chunking_with_overlap():
    chunks = chunk_text(
        "ABCDEFGHIJKLMNO",
        chunk_size=6,
        chunk_overlap=2,
    )

    assert chunks == [
        "ABCDEF",
        "EFGHIJ",
        "IJKLMN",
        "MNO",
    ]


def test_empty_text():
    assert chunk_text("") == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            "Hello",
            chunk_size=0,
        )


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            "Hello",
            chunk_size=5,
            chunk_overlap=5,
        )


def test_negative_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            "Hello",
            chunk_size=5,
            chunk_overlap=-1,
        )


def test_short_text():
    chunks = chunk_text(
        "Hello",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert chunks == ["Hello"]


def test_chunk_size_limit():
    text = "A" * 1000

    chunks = chunk_text(
        text,
        chunk_size=200,
        chunk_overlap=50,
    )

    assert all(len(chunk) <= 200 for chunk in chunks)
    assert chunks[0] == text[:200]
    assert chunks[-1].endswith(text[-50:])

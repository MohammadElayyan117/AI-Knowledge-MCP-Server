
from unittest.mock import patch

import pymupdf

from ai_knowledge_mcp_server.document_processor import (
    read_document,
    MAX_PDF_PAGES,
    MAX_FILE_SIZE_BYTES,
)


def test_corrupted_pdf(tmp_path):
    path = tmp_path / "corrupted.pdf"
    path.write_bytes(b"This is not a valid PDF")

    result = read_document(str(path))

    assert result["success"] is False
    assert "error" in result


def test_password_protected_pdf(tmp_path):
    path = tmp_path / "protected.pdf"

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Confidential content")

    document.save(
        str(path),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-test",
        user_pw="user-test",
    )
    document.close()

    result = read_document(str(path))

    assert result["success"] is False
    assert "Password-protected" in result["error"]


def test_pdf_page_limit(tmp_path):
    path = tmp_path / "many_pages.pdf"

    document = pymupdf.open()

    for _ in range(MAX_PDF_PAGES + 1):
        document.new_page()

    document.save(str(path))
    document.close()

    result = read_document(str(path))

    assert result["success"] is False
    assert "page limit" in result["error"]


def test_file_size_limit(tmp_path):
    path = tmp_path / "large.txt"
    path.write_text("Small file", encoding="utf-8")

    original_stat = type(path).stat

    def fake_stat(self, *args, **kwargs):
        result = original_stat(self, *args, **kwargs)

        if self == path:
            values = list(result)
            values[6] = MAX_FILE_SIZE_BYTES + 1
            return type(result)(values)

        return result

    with patch.object(type(path), "stat", fake_stat):
        result = read_document(str(path))

    assert result["success"] is False
    assert "size limit" in result["error"]


def test_valid_pdf_still_works(tmp_path):
    path = tmp_path / "valid.pdf"

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "MCP document testing")
    document.save(str(path))
    document.close()

    result = read_document(str(path))

    assert result["success"] is True
    assert result["pages"] == 1
    assert "MCP document testing" in result["content"]


from pathlib import Path

import pymupdf


# -----------------------------
# DOCUMENT CONFIGURATION
# -----------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

MAX_PDF_PAGES = 200
MAX_TEXT_CHARACTERS = 5_000_000


# -----------------------------
# HELPERS
# -----------------------------

def error_result(message: str) -> dict:
    """Return a consistent document processing error."""
    return {
        "success": False,
        "error": message,
    }


def validate_document(path: Path) -> dict | None:
    """Validate file existence, format and size."""

    if not path.is_file():
        return error_result("File does not exist.")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return error_result("Unsupported file format.")

    file_size = path.stat().st_size

    if file_size > MAX_FILE_SIZE_BYTES:
        return error_result(
            f"File exceeds the {MAX_FILE_SIZE_MB} MB size limit."
        )

    if file_size == 0:
        return error_result(
            "Document contains no extractable text."
        )

    return None


# -----------------------------
# DOCUMENT READER
# -----------------------------

def read_document(file_path: str) -> dict:
    """Read a supported document with resource limits."""

    try:
        path = Path(file_path).expanduser().resolve()

        validation_error = validate_document(path)

        if validation_error is not None:
            return validation_error

        extension = path.suffix.lower()

        if extension == ".pdf":
            with pymupdf.open(path) as document:

                if document.needs_pass:
                    return error_result(
                        "Password-protected PDFs are not supported."
                    )

                page_count = len(document)

                if page_count > MAX_PDF_PAGES:
                    return error_result(
                        f"PDF exceeds the {MAX_PDF_PAGES}-page limit."
                    )

                text_parts = []
                total_characters = 0

                for page in document:
                    page_text = page.get_text(sort=True)

                    total_characters += len(page_text)

                    if text_parts:
                        total_characters += 1

                    if total_characters > MAX_TEXT_CHARACTERS:
                        return error_result(
                            "Document exceeds the text length limit."
                        )

                    text_parts.append(page_text)

                text = "\n".join(text_parts)

        else:
            text = path.read_text(encoding="utf-8-sig")
            page_count = None

            if len(text) > MAX_TEXT_CHARACTERS:
                return error_result(
                    "Document exceeds the text length limit."
                )

        if not text.strip():
            return error_result(
                "Document contains no extractable text."
            )

        return {
            "success": True,
            "filename": path.name,
            "file_type": extension,
            "pages": page_count,
            "characters": len(text),
            "words": len(text.split()),
            "content": text,
        }

    except (OSError, UnicodeError, ValueError, TypeError):
        return error_result(
            "Unable to read or process the document."
        )

    except pymupdf.FileDataError:
        return error_result(
            "Invalid or corrupted PDF document."
        )

    except pymupdf.EmptyFileError:
        return error_result(
            "PDF document is empty or invalid."
        )

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Dict

from pypdf import PdfReader

from src.api.errors import ApiError
from src.application.config import ApiSettings


MIME_TYPES: Dict[str, set[str]] = {
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    },
}


class UploadValidator:
    def __init__(self, settings: ApiSettings):
        self.settings = settings

    def safe_filename(self, supplied: str) -> str:
        if not supplied or supplied in (".", "..") or "\x00" in supplied:
            raise ApiError(400, "unsafe_filename", "A safe filename is required")
        if "/" in supplied or "\\" in supplied or Path(supplied).name != supplied:
            raise ApiError(400, "unsafe_filename", "Filename paths are not allowed")
        normalized = unicodedata.normalize("NFKC", supplied).strip()
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
        if not stem or stem.startswith(".") or len(stem) > 160:
            raise ApiError(400, "unsafe_filename", "Filename is not allowed")
        extension = Path(stem).suffix.lower()
        if extension not in self.settings.allowed_extensions:
            raise ApiError(415, "unsupported_extension", "Document extension is not supported")
        return stem

    def validate(self, filename: str, content_type: str, data: bytes) -> str:
        safe_name = self.safe_filename(filename)
        if not data:
            raise ApiError(400, "empty_upload", "Uploaded document is empty")
        if len(data) > self.settings.max_upload_bytes:
            raise ApiError(413, "upload_too_large", "Uploaded document exceeds the configured size limit")
        extension = Path(safe_name).suffix.lower()
        normalized_mime = (content_type or "application/octet-stream").split(";", 1)[0].lower()
        if normalized_mime not in MIME_TYPES[extension]:
            raise ApiError(415, "mime_mismatch", "Content type does not match the document extension")
        if extension == ".pdf":
            self._validate_pdf(data)
        elif extension == ".docx":
            self._validate_docx(data)
        else:
            self._validate_text(data)
        return safe_name

    def _validate_text(self, data: bytes) -> None:
        if b"\x00" in data[:4096]:
            raise ApiError(400, "malformed_document", "Text document contains binary data")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ApiError(400, "malformed_document", "Text document must use UTF-8 encoding") from error
        if len(text) > self.settings.max_extracted_characters:
            raise ApiError(413, "document_too_large", "Extracted document text exceeds the configured limit")

    def _validate_pdf(self, data: bytes) -> None:
        if not data.startswith(b"%PDF-"):
            raise ApiError(415, "mime_mismatch", "PDF signature does not match the extension")
        try:
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted:
                raise ApiError(400, "encrypted_document", "Encrypted PDF documents are not supported")
            if len(reader.pages) > self.settings.max_pdf_pages:
                raise ApiError(413, "too_many_pages", "PDF exceeds the configured page-count limit")
            extracted = 0
            for page in reader.pages:
                extracted += len(page.extract_text() or "")
                if extracted > self.settings.max_extracted_characters:
                    raise ApiError(413, "document_too_large", "Extracted document text exceeds the configured limit")
        except ApiError:
            raise
        except Exception as error:
            raise ApiError(400, "malformed_document", "PDF document could not be validated") from error

    def _validate_docx(self, data: bytes) -> None:
        if not data.startswith(b"PK"):
            raise ApiError(415, "mime_mismatch", "DOCX signature does not match the extension")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > self.settings.max_archive_entries:
                    raise ApiError(413, "archive_too_large", "Document archive contains too many entries")
                total = sum(entry.file_size for entry in entries)
                if total > self.settings.max_archive_uncompressed_bytes:
                    raise ApiError(413, "archive_too_large", "Expanded document exceeds the configured limit")
                names = {entry.filename for entry in entries}
                if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                    raise ApiError(400, "malformed_document", "DOCX package is missing required components")
        except ApiError:
            raise
        except (zipfile.BadZipFile, OSError) as error:
            raise ApiError(400, "malformed_document", "DOCX package could not be validated") from error


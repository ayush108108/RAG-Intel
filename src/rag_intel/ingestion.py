"""Multi-format extraction boundary with explicit failure semantics."""

from __future__ import annotations

import csv
import json
import uuid
from pathlib import Path

SOURCE_TYPE_MAP = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".json": "json",
    ".csv": "csv",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".gif": "image",
    ".txt": "text",
    ".md": "text",
}


class ExtractionError(RuntimeError):
    """Raised when a source cannot be extracted safely."""


def get_source_type(filename: str) -> str:
    return SOURCE_TYPE_MAP.get(Path(filename).suffix.lower(), "text")


def route_file(
    file_path: str | Path,
    filename: str | None = None,
    doc_id: str | None = None,
) -> dict:
    path = Path(file_path)
    source_type = get_source_type(path.name)
    extractors = {
        "pdf": extract_pdf,
        "html": extract_html,
        "json": extract_json,
        "csv": extract_csv,
        "image": extract_image,
        "text": extract_text,
    }
    content = extractors[source_type](path)
    display_name = filename or path.name
    return {
        "doc_id": doc_id or str(uuid.uuid4()),
        "content": content,
        "metadata": {
            "source_type": source_type,
            "filename": display_name,
            "title": path.stem.replace("_", " ").replace("-", " ").strip() or display_name,
        },
    }


def extract_text(file_path: str | Path) -> str:
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ExtractionError(f"text extraction failed: {file_path}") from exc


def extract_json(file_path: str | Path) -> str:
    try:
        data = json.loads(Path(file_path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"JSON extraction failed: {file_path}") from exc
    return _flatten_json(data)


def _flatten_json(value: object, prefix: str = "") -> str:
    if isinstance(value, dict):
        return " | ".join(
            _flatten_json(item, f"{prefix}.{key}" if prefix else str(key))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return " | ".join(
            _flatten_json(item, f"{prefix}[{index}]") for index, item in enumerate(value)
        )
    return f"{prefix}: {value}" if prefix else str(value)


def extract_csv(file_path: str | Path) -> str:
    try:
        with Path(file_path).open(
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            rows = [
                ". ".join(f"{key} is {value}" for key, value in row.items() if value)
                for row in reader
            ]
    except OSError as exc:
        raise ExtractionError(f"CSV extraction failed: {file_path}") from exc
    return f"Columns: {', '.join(headers)}. " + " | ".join(rows)


def extract_html(file_path: str | Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ExtractionError("HTML support requires the 'parsers' extra") from exc
    raw = extract_text(file_path)
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def extract_pdf(file_path: str | Path) -> str:
    try:
        import fitz
    except ImportError as exc:
        raise ExtractionError("PDF support requires the 'parsers' extra") from exc
    try:
        with fitz.open(file_path) as document:
            return "\n".join(page.get_text() for page in document)
    except Exception as exc:
        raise ExtractionError(f"PDF extraction failed: {file_path}") from exc


def extract_image(file_path: str | Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ExtractionError("OCR support requires the 'parsers' extra") from exc
    try:
        with Image.open(file_path) as image:
            return pytesseract.image_to_string(image)
    except Exception as exc:
        raise ExtractionError(f"image OCR failed: {file_path}") from exc

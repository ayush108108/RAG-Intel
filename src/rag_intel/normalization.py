"""Deterministic text normalization."""

import re
import unicodedata

_HEADER_FOOTER_PATTERNS = [
    r"page\s+\d+\s+of\s+\d+",
    r"^\s*\d+\s*$",
    r"confidential\s*[-–]\s*do not distribute",
    r"copyright\s+©?\s*\d{4}",
    r"all rights reserved",
    r"www\.[^\s]+\.[a-z]{2,}",
]
_COMPILED_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in _HEADER_FOOTER_PATTERNS
]


def normalize_text(text: str) -> str:
    """Normalize Unicode/whitespace and remove generic header/footer noise."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    lines = [
        line
        for line in text.split("\n")
        if not any(pattern.search(line) for pattern in _COMPILED_PATTERNS)
    ]
    text = "\n".join(lines)
    # Join single line breaks inside paragraphs while preserving paragraph breaks.
    text = (
        text.replace("\r\n", "\n")
        .replace("\n\n", "\x00")
        .replace("\n", " ")
        .replace("\x00", "\n\n")
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()

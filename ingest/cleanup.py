import re
import unicodedata
from collections import Counter

from ingest.models import PageRecord


def clean_pages(pages: list[PageRecord]) -> tuple[list[PageRecord], dict]:
    if not pages:
        return pages, {}

    header = _detect_repeated(pages, is_header=True)
    footer = _detect_repeated(pages, is_header=False)

    cleaned = []
    for page in pages:
        text = page.text

        if header and text.startswith(header):
            text = text[len(header):]
        if footer and text.endswith(footer):
            text = text[:-len(footer)]

        text = re.sub(r"-\s*\n\s*([a-z])", r"\1", text)  # dehyphenate
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)  # page numbers
        text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)  # latex commands
        text = re.sub(r"\\[a-zA-Z]+", "", text)  # latex remnants
        text = _normalize_whitespace(text)

        cleaned.append(PageRecord(doc_id=page.doc_id, page=page.page, text=text))

    return cleaned, {}


def _detect_repeated(pages: list[PageRecord], is_header: bool, max_lines: int = 3):
    if len(pages) < 3:
        return None

    texts = []
    for page in pages:
        lines = page.text.split("\n")
        selected = lines[:max_lines] if is_header else lines[-max_lines:]
        text = "\n".join(selected).strip()
        if text:
            texts.append(text)

    if len(texts) < 3:
        return None

    most_common, count = Counter(texts).most_common(1)[0]
    return most_common if count > len(pages) / 2 else None


def _normalize_whitespace(text: str) -> str:
    # Remove invalid unicode
    cleaned = []
    for char in text:
        try:
            cat = unicodedata.category(char)
            if cat != "Cs" and (cat[0] != "C" or char in ("\n", "\t", "\r")):
                cleaned.append(char)
        except Exception:
            pass

    text = "".join(cleaned)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

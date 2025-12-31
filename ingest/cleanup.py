"""Deterministic text cleanup module - aggressive cleaning to remove noise."""

import re
from collections import Counter
from typing import Optional

from ingest.models import PageRecord


def clean_pages(pages: list[PageRecord]) -> tuple[list[PageRecord], dict]:
    """
    Apply aggressive deterministic cleanup to page text.

    Philosophy: Be aggressive - minor data loss acceptable, poisoned data is bad.
    - Prefer removing questionable text over keeping noise
    - Use regex-based rules, no ML
    - Track what was removed for debugging

    Args:
        pages: List of PageRecord objects to clean

    Returns:
        Tuple of (cleaned pages, cleanup stats dict)
    """
    stats = {
        "total_pages": len(pages),
        "headers_removed": 0,
        "footers_removed": 0,
        "dehyphenated": 0,
        "whitespace_collapsed": 0,
        "page_numbers_removed": 0,
    }

    if not pages:
        return pages, stats

    # Detect repeated headers/footers across pages
    repeated_header = _detect_repeated_header(pages)
    repeated_footer = _detect_repeated_footer(pages)

    cleaned_pages = []
    for page in pages:
        text = page.text

        # Remove detected headers
        if repeated_header:
            if text.startswith(repeated_header):
                text = text[len(repeated_header) :]
                stats["headers_removed"] += 1

        # Remove detected footers
        if repeated_footer:
            if text.endswith(repeated_footer):
                text = text[: -len(repeated_footer)]
                stats["footers_removed"] += 1

        # Dehyphenate line breaks (word- \nbreak → wordbreak)
        original_text = text
        text = _dehyphenate(text)
        if text != original_text:
            stats["dehyphenated"] += 1

        # Strip page numbers (common patterns)
        original_text = text
        text = _remove_page_numbers(text)
        if text != original_text:
            stats["page_numbers_removed"] += 1

        # Remove common LaTeX artifacts
        text = _remove_latex_artifacts(text)

        # Normalize whitespace aggressively
        original_text = text
        text = _normalize_whitespace(text)
        if text != original_text:
            stats["whitespace_collapsed"] += 1

        # Create cleaned page record
        cleaned_pages.append(
            PageRecord(doc_id=page.doc_id, page=page.page, text=text)
        )

    return cleaned_pages, stats


def _detect_repeated_header(pages: list[PageRecord], max_lines: int = 3) -> Optional[str]:
    """
    Detect repeated header by finding common text at start of pages.

    Args:
        pages: List of pages
        max_lines: Maximum number of lines to consider for header

    Returns:
        Repeated header text if detected, None otherwise
    """
    if len(pages) < 3:  # Need at least 3 pages to detect pattern
        return None

    # Get first N lines from each page
    first_lines_list = []
    for page in pages:
        lines = page.text.split("\n")[:max_lines]
        first_text = "\n".join(lines).strip()
        if first_text:
            first_lines_list.append(first_text)

    # Find most common first N lines
    if len(first_lines_list) < 3:
        return None

    counter = Counter(first_lines_list)
    most_common, count = counter.most_common(1)[0]

    # If it appears in >50% of pages, consider it a header
    if count > len(pages) / 2:
        return most_common

    return None


def _detect_repeated_footer(pages: list[PageRecord], max_lines: int = 2) -> Optional[str]:
    """
    Detect repeated footer by finding common text at end of pages.

    Args:
        pages: List of pages
        max_lines: Maximum number of lines to consider for footer

    Returns:
        Repeated footer text if detected, None otherwise
    """
    if len(pages) < 3:
        return None

    # Get last N lines from each page
    last_lines_list = []
    for page in pages:
        lines = page.text.split("\n")[-max_lines:]
        last_text = "\n".join(lines).strip()
        if last_text:
            last_lines_list.append(last_text)

    # Find most common last N lines
    if len(last_lines_list) < 3:
        return None

    counter = Counter(last_lines_list)
    most_common, count = counter.most_common(1)[0]

    # If it appears in >50% of pages, consider it a footer
    if count > len(pages) / 2:
        return most_common

    return None


def _dehyphenate(text: str) -> str:
    """
    Remove line-break hyphens (word- \\nbreak → wordbreak).

    Aggressive: Remove hyphens at end of lines followed by newline.
    May occasionally merge words that should stay hyphenated, but that's acceptable.
    """
    # Pattern: hyphen followed by newline and lowercase letter
    # This is aggressive - removes most end-of-line hyphens
    text = re.sub(r"-\s*\n\s*([a-z])", r"\1", text)
    return text


def _remove_page_numbers(text: str) -> str:
    """
    Remove common page number patterns.

    Looks for isolated numbers at start/end of text or on their own lines.
    """
    # Remove lines that are just numbers (likely page numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove patterns like "Page 5" or "5 of 10"
    text = re.sub(r"^\s*Page\s+\d+\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s+of\s+\d+\s*$", "", text, flags=re.MULTILINE)

    return text


def _remove_latex_artifacts(text: str) -> str:
    """
    Remove common LaTeX/formatting artifacts.

    Aggressive removal of known noise patterns.
    """
    # Remove LaTeX commands that slipped through (e.g., \\textbf{}, \\cite{})
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)

    # Remove isolated LaTeX command remnants
    text = re.sub(r"\\[a-zA-Z]+", "", text)

    # Remove excessive newlines (more than 2 consecutive)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def _normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace aggressively.

    - Collapse multiple spaces to single space
    - Collapse multiple newlines to max 2
    - Strip leading/trailing whitespace
    - Remove invalid unicode characters (surrogates, etc.)
    """
    # Remove invalid unicode characters (surrogates, control chars except newline/tab)
    # This fixes PDF extraction errors with malformed unicode
    import unicodedata

    cleaned_chars = []
    for char in text:
        try:
            cat = unicodedata.category(char)
            # Keep normal chars, newlines, tabs
            # Remove surrogates (Cs), some control chars, etc.
            if cat != "Cs" and (
                cat[0] != "C" or char in ("\n", "\t", "\r")
            ):  # Allow \n, \t, \r
                cleaned_chars.append(char)
        except Exception:
            # If we can't categorize it, skip it
            pass

    text = "".join(cleaned_chars)

    # Collapse multiple spaces (but preserve single newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse multiple newlines to maximum 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

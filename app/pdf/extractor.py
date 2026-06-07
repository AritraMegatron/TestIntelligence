from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


# Keep this aligned with MVP-supported RF tests:
# 1. GAIN
# 2. EVM
# 3. ACPR
# 4. CURRENT
RF_KEYWORDS = {
    "gain": [
        "gain",
        "small signal gain",
        "power gain",
        "s21",
    ],
    "evm": [
        "evm",
        "error vector magnitude",
        "802.11",
        "wifi",
        "wlan",
        "he20",
        "he40",
        "he80",
        "he160",
        "vht",
        "ofdm",
        "modulation",
    ],
    "acpr": [
        "acpr",
        "aclr",
        "adjacent channel",
        "adjacent channel power",
        "spectrum mask",
        "sem",
        "dBc",
    ],
    "current": [
        "current",
        "supply current",
        "current consumption",
        "quiescent current",
        "icc",
        "idd",
        "iq",
        "ma",
        "amp",
    ],
    "shared_rf": [
        "frequency",
        "freq",
        "mhz",
        "ghz",
        "pout",
        "output power",
        "input power",
        "pin",
        "psat",
        "vcc",
        "vdd",
        "supply",
        "voltage",
        "temperature",
        "duty cycle",
        "mode",
        "tx",
        "rx",
        "typ",
        "min",
        "max",
        "condition",
        "test condition",
        "specification",
        "electrical characteristics",
    ],
}


@dataclass
class ExtractedPage:
    page_number: int
    text: str
    score: int
    matched_terms: list[str]


def _clean_text(text: str) -> str:
    """
    Normalize PDF text so the LLM sees cleaner content.
    """
    if not text:
        return ""

    # Normalize non-breaking spaces and weird whitespace.
    text = text.replace("\u00a0", " ")

    # Fix common PDF extraction spacing issues.
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove repeated spaces around newlines.
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def _extract_page_text(page) -> str:
    """
    Extract text from one page using PyMuPDF.

    For MVP:
    - no OCR
    - use normal text extraction
    - fallback to block extraction if needed
    """
    text = page.get_text("text") or ""
    text = _clean_text(text)

    if text:
        return text

    # Fallback: block extraction sometimes works better for odd PDFs.
    blocks = page.get_text("blocks") or []
    block_texts = []

    for block in blocks:
        # PyMuPDF block tuple usually has text at index 4.
        if len(block) >= 5 and isinstance(block[4], str):
            cleaned = _clean_text(block[4])
            if cleaned:
                block_texts.append(cleaned)

    return _clean_text("\n".join(block_texts))


def _score_page(text: str) -> tuple[int, list[str]]:
    """
    Score pages by RF usefulness.

    Higher score means the page is more likely to contain useful test-generation data.
    """
    lowered = text.lower()
    score = 0
    matched_terms: list[str] = []

    for group_name, terms in RF_KEYWORDS.items():
        for term in terms:
            term_lower = term.lower()
            if term_lower in lowered:
                matched_terms.append(term)

                if group_name in {"gain", "evm", "acpr", "current"}:
                    score += 5
                else:
                    score += 2

    # Tables/spec pages often contain many units and min/typ/max language.
    numeric_unit_patterns = [
        r"\b\d+(\.\d+)?\s*mhz\b",
        r"\b\d+(\.\d+)?\s*ghz\b",
        r"\b-?\d+(\.\d+)?\s*dbm\b",
        r"\b-?\d+(\.\d+)?\s*db\b",
        r"\b\d+(\.\d+)?\s*v\b",
        r"\b\d+(\.\d+)?\s*ma\b",
        r"\b\d+(\.\d+)?\s*%\b",
    ]

    for pattern in numeric_unit_patterns:
        matches = re.findall(pattern, lowered)
        score += min(len(matches), 10)

    # Electrical characteristics/spec tables are especially useful.
    table_markers = [
        "min",
        "typ",
        "max",
        "unit",
        "condition",
        "parameter",
        "symbol",
    ]

    table_marker_hits = sum(1 for marker in table_markers if marker in lowered)
    if table_marker_hits >= 3:
        score += 10

    # De-duplicate matched terms while preserving order.
    seen = set()
    unique_terms = []
    for term in matched_terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            unique_terms.append(term)

    return score, unique_terms


def _format_page(page: ExtractedPage) -> str:
    terms = ", ".join(page.matched_terms[:12]) if page.matched_terms else "none"

    return (
        f"\n--- Page {page.page_number} | RF relevance score: {page.score} | matched: {terms} ---\n"
        f"{page.text}"
    )


def _build_extraction_summary(
    path: Path,
    total_pages: int,
    extracted_pages: list[ExtractedPage],
    selected_pages: list[ExtractedPage],
    scanned_or_empty_pages: list[int],
) -> str:
    selected_numbers = [p.page_number for p in selected_pages]
    high_score_pages = [
        p.page_number for p in extracted_pages
        if p.score > 0
    ]

    summary_lines = [
        "=== RF SPEC EXTRACTION SUMMARY ===",
        f"file_name: {path.name}",
        f"total_pages: {total_pages}",
        f"pages_with_rf_matches: {high_score_pages}",
        f"selected_pages_in_prompt: {selected_numbers}",
        f"empty_or_scanned_pages_without_text: {scanned_or_empty_pages}",
        "",
        "MVP supported test types for extraction:",
        "- GAIN",
        "- EVM",
        "- ACPR",
        "- CURRENT",
        "",
        "Extraction notes:",
        "- Text was extracted with PyMuPDF.",
        "- No OCR was performed.",
        "- Pages were ranked by RF relevance so important RF spec pages are less likely to be lost by truncation.",
        "- Source evidence should still quote only phrases found in the extracted text.",
        "=== END RF SPEC EXTRACTION SUMMARY ===",
    ]

    return "\n".join(summary_lines)


def extract_text_from_pdf(file_path: str | Path, max_chars: int = 50000) -> str:
    """
    Extract RF-relevant text from a PDF using PyMuPDF.

    MVP behavior:
    - Works well for text-based PDFs.
    - Does not OCR scanned images.
    - Scores pages for RF relevance.
    - Keeps page numbers for traceability.
    - Prioritizes useful RF/spec pages before truncating.
    - Still returns a single string, so existing SPEC upload flow does not need UI changes.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    extracted_pages: list[ExtractedPage] = []
    scanned_or_empty_pages: list[int] = []

    with fitz.open(path) as doc:
        total_pages = len(doc)

        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            page_text = _extract_page_text(page)

            if not page_text:
                scanned_or_empty_pages.append(page_number)
                continue

            score, matched_terms = _score_page(page_text)

            extracted_pages.append(
                ExtractedPage(
                    page_number=page_number,
                    text=page_text,
                    score=score,
                    matched_terms=matched_terms,
                )
            )

    if not extracted_pages:
        return (
            "=== RF SPEC EXTRACTION SUMMARY ===\n"
            f"file_name: {path.name}\n"
            "No selectable text was extracted from this PDF.\n"
            "This may be a scanned/image-only datasheet. OCR is not enabled in the MVP.\n"
            "=== END RF SPEC EXTRACTION SUMMARY ==="
        )

    # Always include the first page because it often contains product identity and headline specs.
    first_page = extracted_pages[0]

    # Rank remaining pages by RF usefulness.
    ranked_pages = sorted(
        extracted_pages,
        key=lambda p: (p.score, -p.page_number),
        reverse=True,
    )

    selected_pages: list[ExtractedPage] = []

    def add_page(page: ExtractedPage):
        if all(existing.page_number != page.page_number for existing in selected_pages):
            selected_pages.append(page)

    add_page(first_page)

    # Add highest scoring RF pages.
    for page in ranked_pages:
        add_page(page)

    summary = _build_extraction_summary(
        path=path,
        total_pages=len(extracted_pages) + len(scanned_or_empty_pages),
        extracted_pages=extracted_pages,
        selected_pages=selected_pages,
        scanned_or_empty_pages=scanned_or_empty_pages,
    )

    output_parts = [summary]

    current_chars = len(summary)

    for page in selected_pages:
        formatted = _format_page(page)
        next_len = current_chars + len(formatted)

        if next_len > max_chars:
            remaining = max_chars - current_chars

            if remaining > 1000:
                output_parts.append(formatted[:remaining])
            break

        output_parts.append(formatted)
        current_chars = next_len

    return "\n\n".join(output_parts).strip()
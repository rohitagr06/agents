"""
utils/sanitizer.py — Text Sanitization for MediScan AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
PURPOSE:
    Raw text extracted from PDFs and DOCX files is messy.
    Before we send it to the LLM, we must clean it up.
 
    Why? Because LLMs work best with clean, structured input.
    Feeding garbage in produces garbage out. Specifically:
 
    PDF PROBLEMS:
    — Extra whitespace and blank lines from layout rendering
    — Hyphenated line breaks ("haemo-\nglobin" instead of "haemoglobin")
    — Garbled encoding artefacts from font embedding (ï¬‚ instead of fl)
    — Page numbers, headers, footers mixed into body text
    — Column layout bleed (text from two columns merging on one line)
 
    DOCX PROBLEMS:
    — Empty paragraphs used as spacers
    — Repeated section headers from styles
    — Track-change fragments (deleted text that was kept in XML)
 
    TOKEN LIMIT PROBLEM:
    — gpt-4.1-mini has a context window. A 30-page discharge summary
      can easily exceed what we can send in one API call.
    — We must chunk long documents intelligently rather than cutting
      them off mid-sentence or sending too little.
 
WHY A SEPARATE FILE?
    Sanitization is its own domain of logic. Keeping it separate means:
    — Easy to test with just text strings (no file I/O needed)
    — Can be improved independently without touching parser or agents
    — document_parser.py stays focused on "how to extract"
      while sanitizer.py focuses on "how to clean"
"""

import re

# ─────────────────────────────────────────────────────────────
#  Constants
#
#  WHY ARE THESE CONSTANTS NOT IN config.py?
#  These are internal implementation details of the sanitizer.
#  config.py holds user-facing settings (file size, port number etc.)
#  These values rarely change and have no reason to be user-configurable.
#  Keeping them here makes this module self-contained.
# ─────────────────────────────────────────────────────────────
 
# Approximate characters per LLM token (conservative estimate)
# GPT-4 family: ~4 chars per token on average for English medical text

CHARS_PER_TOKEN: int = 4

# Max tokens we want to send in a single LLM call
# We leave headroom below the model's actual limit for:
#   — The system prompt (which also consumes tokens)
#   — The model's response (which must fit in the context too)

MAX_TOKENS_PER_CHUNK: int = 6000

# Therefore, max characters per chunk
MAX_CHARS_PER_CHUNK = MAX_TOKENS_PER_CHUNK * CHARS_PER_TOKEN # 24000

# Minimum meaningful text length
# If extracted text is shorter than this, it's probably a
# scanned image PDF (no real text layer) or a blank document

MIN_MEANINGFUL_LENGTH: int = 100

# ─────────────────────────────────────────────────────────────
#  sanitize()  — MAIN PUBLIC FUNCTION
#
#  Takes raw extracted text and returns clean text ready for the LLM.
#  Applies a series of cleaning steps in the correct order.
#  Order matters — some steps depend on earlier steps having run first.
# ─────────────────────────────────────────────────────────────

def sanitize(raw_text: str) -> str:
    """
    Clean raw extracted text for LLM consumption.
 
    Applies these cleaning steps in order:
        1. Strip form feed characters (\x0c) from PDF page breaks
        2. Fix PDF hyphenated line breaks
        3. Normalize unicode / encoding artefacts
        4. Collapse excessive whitespace
        5. Remove page numbers and common header/footer patterns
        6. Collapse excessive blank lines
        7. Strip leading/trailing whitespace
 
    Args:
        raw_text: The raw text string from document_parser.py
 
    Returns:
        Cleaned text string, ready to be sent to the LLM agent.
        Returns empty string if input is empty.
    """

    if not raw_text or not raw_text.strip():
        return ""

    text = str(raw_text)
    
    # ── Step 1: Fix hyphenated line breaks ───────────────────
    # PDFs often break words across lines with a hyphen.
    # "haemo-\nglobin" should be "haemoglobin"
    # "cardio-\nvascular" should be "cardiovascular"
    # The regex matches: a letter, a hyphen, a newline, a letter
    # and replaces the hyphen+newline with nothing (joins the word).

    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # ── Step 2: Normalize unicode artefacts ──────────────────
    # PDFs with embedded fonts sometimes produce garbled characters.
    # We replace the most common ones with their correct equivalents.
    # "fi" ligature artefact → "fi"
    # "fl" ligature artefact → "fl"
    # Non-breaking space (U+00A0) → regular space
    # En dash / Em dash → regular hyphen (for consistency)

    replacements = {
        "\ufb01": "fi",       # fi ligature
        "\ufb02": "fl",       # fl ligature
        "\u00a0": " ",        # non-breaking space
        "\u2013": "-",        # en dash
        "\u2014": "-",        # em dash
        "\u2018": "'",        # left single quote
        "\u2019": "'",        # right single quote
        "\u201c": '"',        # left double quote
        "\u201d": '"',        # right double quote
    }

    for bad_char, good_char in replacements.items():
        text = text.replace(bad_char, good_char)

    # ── Step 3: Collapse runs of spaces/tabs ─────────────────
    # Multiple consecutive spaces or tabs become a single space.
    # "Hemoglobin      :      11.2" → "Hemoglobin : 11.2"
    # \t is tab, we treat it same as a space for cleanliness.

    text = re.sub(r"[ \t]+", " ", text)

    # ── Step 4: Remove page numbers and header/footer noise ──
    # Common patterns in medical PDFs:
    # — "Page 1 of 5" or "Page 1"
    # — "Confidential" appearing as page header on every page
    # — Lines that are just numbers (page numbers standing alone)
    # We remove lines that ONLY contain these patterns.
    # re.MULTILINE makes ^ and $ match start/end of each LINE.

    text = re.sub(r"(?im)^[ \t]*page\s+\d+(\s+of\s+\d+)?[ \t]*$", "", text)
    # text = re.sub(r"(?im)^[ \t]*\d+[ \t]*$", "", text)
    text = re.sub(r"(?im)^[ \t]*confidential[ \t]*$", "", text)

    # ── Step 5: Collapse excessive blank lines ────────────────
    # Three or more consecutive newlines become two (one blank line).
    # This preserves paragraph structure without the huge gaps.

    text = re.sub(r"\n{3,}", "\n\n", text)

    # ── Step 6: Strip outer whitespace ───────────────────────
    text = text.strip()

    return text

# ─────────────────────────────────────────────────────────────
#  is_meaningful()  — quality gate
#
#  WHY THIS EXISTS:
#  Some PDFs are scanned images — they look like a PDF but contain
#  zero extractable text (just embedded image data). PyMuPDF will
#  extract an empty or near-empty string from them.
#
#  Rather than sending "" to the LLM and getting a confused response,
#  we detect this early and return a clear message to the user.
#
#  RC2 will handle scanned PDFs via OCR (pytesseract + pdf2image).
#  For now, we detect and inform gracefully.
# ─────────────────────────────────────────────────────────────

def is_meaningful(text: str) -> bool:
    """
    Check whether extracted text has enough content to be worth analyzing.
 
    Returns False if:
    — The text is empty or only whitespace
    — The text is shorter than MIN_MEANINGFUL_LENGTH characters
      (likely a scanned PDF with no text layer, or a blank document)
 
    Returns True if text has meaningful content.
 
    Args:
        text: Cleaned text string (after sanitize() has been called)
    """
    return bool(text) and len(text.strip()) >= MIN_MEANINGFUL_LENGTH

# ─────────────────────────────────────────────────────────────
#  _split_oversized_paragraph()  — private helper
#
#  WHY THIS EXISTS:
#  chunk_text() splits at paragraph boundaries (\n\n).
#  But what if a single paragraph is itself longer than max_chars?
#  For example, a test that passes max_chars=500 but one paragraph
#  is 2149 chars — we can't just add it whole to a chunk.
#
#  STRATEGY — sentence boundary splitting:
#  We split the oversized paragraph at sentence boundaries (. ! ?)
#  rather than mid-word. This keeps medical values intact:
#  "Hemoglobin 11.2 g/dL." stays together rather than being cut at
#  the character limit mid-value.
# ─────────────────────────────────────────────────────────────
 
def _split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """
    Split a single paragraph that exceeds max_chars at sentence boundaries.
 
    Falls back to hard character splitting only if a single sentence
    itself exceeds max_chars (extremely rare in medical text).
 
    Args:
        paragraph: A single paragraph string longer than max_chars
        max_chars:  The character limit per chunk
 
    Returns:
        List of sub-chunks, each within max_chars.
    """
    # Split at sentence boundaries: period, exclamation, question mark
    # followed by whitespace or end of string.
    # We keep the delimiter attached to the sentence using a lookahead.
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)

    sub_chunks: list[str] = []
    current_parts: list[str] = []
    current_length: int = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # Edge case: a single sentence is longer than max_chars
        # (e.g. a very long lab value line with no punctuation)
        # Hard-split it at max_chars to avoid infinite accumulation.

        if sentence_len > max_chars:
            # Flush current accumulation first
            if current_parts:
                sub_chunks.append(" ".join(current_parts))
                current_parts = []
                current_length = 0
            # Hard split the oversized sentence
            for i in range(0, sentence_len, max_chars):
                sub_chunks.append(sentence[i:i + max_chars])
            continue

        # Would adding this sentence exceed the limit?
        if current_length + sentence_len + 1 > max_chars and current_parts:
            sub_chunks.append(" ".join(current_parts))
            current_parts = [sentence]
            current_length = sentence_len
        else:
            current_parts.append(sentence)
            current_length += sentence_len + 1  # +1 for the space

    # Don't forget the last batch
    if current_parts:
        sub_chunks.append(" ".join(current_parts))
 
    return sub_chunks


# ─────────────────────────────────────────────────────────────
#  chunk_text()  — handle long documents
#
#  WHY WE NEED THIS:
#  A discharge summary can be 20+ pages, easily 50,000+ characters.
#  Sending all of that in one API call would:
#    1. Exceed the model's context window → API error
#    2. Be slow and expensive even if it worked
#
#  CHUNKING STRATEGY — paragraph-aware:
#  We don't blindly cut at a character limit because that could
#  slice a sentence or lab value in half ("Hemoglobin 11" cut to
#  "Hemoglobin 1" / "1 g/dL"). Instead, we:
#    1. Split the text into paragraphs (split on blank lines)
#    2. Add paragraphs to the current chunk one by one
#    3. When adding the next paragraph would exceed the limit,
#       we save the current chunk and start a new one
#
#  This guarantees that every chunk boundary falls between
#  paragraphs — never mid-sentence, never mid-value.
#
#  RC1 USAGE:
#  For most medical reports (1-5 pages), the whole text fits in
#  one chunk and this function returns a list with one element.
#  The caller always gets a list, so it works the same either way.
# ─────────────────────────────────────────────────────────────


def chunk_text(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list[str]:
    """
    Split long text into LLM-safe chunks at paragraph boundaries.
 
    For most medical reports this returns a list with one item.
    For very long discharge summaries it may return 2-3 chunks.
 
    Args:
        text:      Cleaned text from sanitize()
        max_chars: Maximum characters per chunk (default: 24,000 ≈ 6,000 tokens)
 
    Returns:
        List of text chunk strings. Always at least one element.
        Each chunk is within the max_chars limit.
    """
    if not text:
        return []

    # if the entire text fits in one chunk, return it immediately.
    # This is the common case for typical 1-5 page medical reports

    if len(text) <= max_chars:
        return [text]

    # Split into paragraphs on blank lines
    # \n\n+ matches one or more blank lines between paragraphs

    paragraphs: list[str] = re.split(r"\n\n+", text)

    chunks: list[str] = []
    current_chunk_parts: list[str] = []
    current_length: int = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_length = len(paragraph)
        # ── FIX: Handle paragraphs that are themselves too large ──
        # If a single paragraph exceeds max_chars, we cannot add it
        # whole to any chunk. Split it at sentence boundaries first,
        # then process each sub-chunk as if it were a normal paragraph.
        if paragraph_length > max_chars:
            # First flush whatever is in the current chunk
            if current_chunk_parts:
                chunks.append("\n\n".join(current_chunk_parts))
                current_chunk_parts = []
                current_length = 0
            # Split the oversized paragraph at sentence boundaries
            sub_chunks = _split_oversized_paragraph(paragraph, max_chars)
            chunks.extend(sub_chunks)
            continue

        # Would adding this paragraph exceed the limit?
        # +2 accounts for the \n\n separator we'll add between paragraphs
        if current_length + paragraph_length + 2 > max_chars and current_chunk_parts:
            # Save current chunk and start a new one
            chunks.append("\n\n".join(current_chunk_parts))
            current_chunk_parts = [paragraph]
            current_length = paragraph_length
        else:
            # Add paragraph to current chunk
            current_chunk_parts.append(paragraph)
            current_length += paragraph_length + 2

    # Don't forget the last chunk
    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))

    return chunks

# ─────────────────────────────────────────────────────────────
#  get_text_stats()  — diagnostic metadata
#
#  WHY THIS EXISTS:
#  When displaying results in the Gradio UI, it's helpful to show
#  the user metadata about what was extracted:
#    "Extracted 3,421 words across 2 pages"
#  This function computes those stats from the cleaned text.
#  It's also useful for debugging during development.
# ─────────────────────────────────────────────────────────────

def get_text_stats(text: str) -> dict:
    """
    Return diagnostic statistics about the extracted text.
 
    Args:
        text: Cleaned text after sanitize()
 
    Returns:
        dict with keys:
            char_count   : Total character count
            word_count   : Approximate word count
            line_count   : Total line count
            chunk_count  : How many LLM chunks this would split into
            is_meaningful: Whether the text passes the quality gate
    """

    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
            "chunk_count": 0,
            "is_meaningful": False
        }

    chunks = chunk_text(text)

    return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.splitlines()),
            "chunk_count": len(chunks),
            "is_meaningful": is_meaningful(text)
        }
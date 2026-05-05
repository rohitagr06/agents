"""
utils/validator.py - File Validation for Mediscan AI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Purpose:
    This is the first thing that runs whether a user uploads a file.
    Before we spend any time or money parsing a document or calling the
    LLM API, we want to make sure the file is actually valid.
    Think of this as the "security guard at the door" -- it checks:

    1. Was a file actually uploaded? (Not None)
    2. Is the file extension allowed? (.pdf, .docx and .doc only)
    3. Is the file size within limits? (default max: 10MB)
    4. Is the file physically readable? (not corrupted/empty)

WHY A SEPARATE FILE?
    We could put this logic directly in app.py or document_parser.py,
    but separating it means:
    — It's easy to find, test, and modify independently
    — We can import it anywhere without circular dependencies
    — Week 3 agents can also call it before processing
 
DESIGN PATTERN:
    Every function returns a consistent ValidationResult dataclass so
    callers always get the same shape of data — a success flag, a
    human-readable message, and optional metadata about the file.
    This is much safer than returning raw booleans or raising exceptions
    for expected failure cases (wrong file type, too big etc.).
"""

import os
from dataclasses import dataclass, field
import config

# ─────────────────────────────────────────────────────────────
#  ValidationResult — structured return type
#
#  WHY A DATACLASS?
#  We use a dataclass instead of returning a plain tuple like
#  (True, "ok", {...}) because:
#    — Named fields are self-documenting: result.is_valid is
#      clearer than result[0]
#    — It's impossible to accidentally mix up the order of values
#    — We can add new fields later without breaking existing callers
#    — Pydantic BaseModel would also work, but dataclass is lighter
#      since we don't need JSON serialization here
# ─────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Returned by every validate_*() function in this module.
 
    Fields:
        is_valid  : True if validation passed, False if it failed
        message   : Human-readable message shown in the Gradio status bar
        file_name : The original filename (empty string if not available)
        file_size : File size in bytes (0 if unknown)
        extension : Lowercase file extension without dot (e.g. "pdf")
        metadata  : Any extra info the caller might find useful
    """
    is_valid: bool
    message: str
    file_name: str = ""
    file_size: int = 0
    extension: str = ""
    metadata: dict = field(default_factory=dict)

# ─────────────────────────────────────────────────────────────
#  _get_extension()  — private helper
#
#  WHY PRIVATE (underscore prefix)?
#  This is only used internally by validate_file(). Prefixing with
#  underscore signals to other developers "don't import or call this
#  directly — it's an implementation detail". It won't show up in
#  autocomplete suggestions when importing from this module.
#
#  WHY rsplit instead of split?
#  rsplit(".", 1) splits from the RIGHT, taking only the last part.
#  This handles edge cases like "report.final.v2.pdf" correctly —
#  it gives us "pdf", not "report".
#  [-1] gets the last element. lower() normalises "PDF" → "pdf".
# ───────────────────────────────────────────────────────────── 

def _get_extension(filename: str) -> str:
    """Extract and reurn the lowercase file extension without the dot."""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()

# ─────────────────────────────────────────────────────────────
#  _format_size()  — private helper
#
#  WHY THIS EXISTS:
#  When we show an error like "File too large (10.5 MB)", it's much
#  friendlier than "File too large (11010048 bytes)". This helper
#  converts raw bytes into the most readable unit automatically.
# ─────────────────────────────────────────────────────────────

def _format_size(size_bytes: int) -> str:
    """ Convert bytes to a human-readable string like '4.2 MB' or '850 KB'."""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} bytes"

# ─────────────────────────────────────────────────────────────
#  validate_file()  — MAIN PUBLIC FUNCTION
#
#  This is the single entry point that app.py and the orchestrator
#  will call. It runs all four checks in sequence:
#
#    Check 1 → Presence   (did anything get uploaded?)
#    Check 2 → Extension  (is it PDF/DOCX/DOC?)
#    Check 3 → Size       (is it under the configured limit?)
#    Check 4 → Readability(can we actually open and read the file?)
#
#  We return IMMEDIATELY on the first failure (early return pattern).
#  There's no point checking the file size if there's no file at all.
#  This keeps the logic clean and errors specific.
#
#  WHY "file" PARAMETER IS TYPED AS "object":
#  Gradio passes the uploaded file as a NamedString / tempfile object.
#  The exact type varies across Gradio versions, so we type it loosely
#  as "object" and use hasattr() to safely access .name. This makes
#  the validator resilient to Gradio version changes.
# ─────────────────────────────────────────────────────────────

def validate_file(file: object) -> ValidationResult:
    """
    Run all validation checks on an uploaded file.
 
    This is the ONLY function you need to call from outside this module.
    It runs checks in order and returns immediately on the first failure.
 
    Args:
        file: The file object passed by Gradio's gr.File component.
              Has a .name attribute containing the temp file path.
 
    Returns:
        ValidationResult with is_valid=True if all checks pass,
        or is_valid=False with a descriptive message if any check fails.
 
    Usage:
        result = validate_file(file)
        if not result.is_valid:
            return result.message   # show this in the Gradio status bar
    """

    # ── Check 1: Presence ────────────────────────────────────
    # Gradio passes None if the user clicks "Analyze" without
    # uploading anything. We check this first before touching
    # any file attributes, to avoid AttributeError crashes.

    if file is None:
        return ValidationResult(
            is_valid=False,
            message="⚠️ No file uploaded. Please upload a PDF or DOCX report."
        )

    # ── Extract file path safely ──────────────────────────────
    # Gradio wraps the uploaded file. The actual path on disk
    # is in the .name attribute. We use hasattr() to be safe
    # across different Gradio versions.

    file_path: str = file.name if hasattr(file, "name") else str(file)
    file_name: str = os.path.basename(file_path)
    extension: str = _get_extension(file_name)

    # ── Check 2: Extension ───────────────────────────────────
    # We only accept PDF, DOCX, and DOC. We compare the extension
    # against config. ALLOWED_EXTENSIONS which is [".pdf", ".docx", ".doc"].
    # We add the dot prefix to match the config format: "pdf" → ".pdf"

    if f".{extension}" not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(config.ALLOWED_EXTENSIONS)
        return ValidationResult(
            is_valid=False,
            message=(
                f"❌ Unsupported file type: '.{extension}'. "
                f"Please upload one of: {allowed}"
            ),
            file_name=file_name,
            extension=extension
        )

    # ── Check 3: Size ────────────────────────────────────────
    # os.path.getsize() returns the file size in bytes from disk.
    # We compare against config.MAX_FILE_SIZE_BYTES (default: 10MB).
    # We do this BEFORE reading the file content, so we never load
    # a 500MB file into memory just to reject it.

    try:
        file_size: int = os.path.getsize(file_path)
    except OSError:
        # File path is inaccessible — treat as unreadable
        file_size = 0

    if file_size == 0:
        return ValidationResult(
            is_valid=False,
            message="❌ The uploaded file appears to be empty. Please upload a valid report.",
            file_name=file_name,
            extension=extension
        )

    if file_size > config.MAX_FILE_SIZE_BYTES:
        return ValidationResult(
            is_valid=False,
            message=(
                f"❌ File too large: {_format_size(file_size)}. "
                f"Maximum allowed size is {config.MAX_FILE_SIZE_MB} MB."
            ),
            file_name=file_name,
            extension=extension,
            file_size=file_size
        )

    # ── Check 4: Readability ─────────────────────────────────
    # We try to actually open the file to make sure it's accessible
    # and not corrupted at the filesystem level. We only read 8 bytes
    # (the file "magic bytes" / header) — enough to confirm the file
    # opens correctly without loading the whole thing into memory.
    # Deep format validation (is it really a valid PDF?) happens in
    # document_parser.py which uses proper PDF/DOCX libraries.

    try:
        with open(file_path, "rb") as f:
            f.read(8)
    except (IOError, OSError) as e:
        return ValidationResult(
            is_valid=False,
            message=f"❌ Cannot read the uploaded file. It may be corrupted. ({e})",
            file_name=file_name,
            extension=extension,
            file_size=file_size
        )

    # ── All checks passed ─────────────────────────────────────
    # Return success with full file metadata so the caller
    # (document_parser.py) doesn't need to re-derive these values.

    return ValidationResult(
        is_valid=True,
        message=f"✅ File validated: {file_name} ({_format_size(file_size)})",
        file_name=file_name,
        file_size=file_size,
        extension=extension,
        metadata={
            "file_path": file_path,
            "size_formatted": _format_size(file_size),
        }
    )


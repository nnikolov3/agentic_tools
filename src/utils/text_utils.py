"""
This module provides utility functions for text processing and cleaning.
"""

import re
import logging
from typing import Final

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Explicitly set logger level to DEBUG
RE_NON_PRINTABLE: Final[re.Pattern[str]] = re.compile(r"[^\x20-\x7E\n\r\t]")
RE_MULTI_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")
LIGATURES: Final[dict[str, str]] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


def clean_text(text: str) -> str:
    """
    Cleans a string by normalizing whitespace and removing non-printable characters.

    Why: This ensures that text fed into embedding models is consistent and
    free of artifacts that can degrade embedding quality.
    """

    """Cleans text by removing ligatures and non-printable characters."""
    if not text:
        return ""

    for ligature, replacement in LIGATURES.items():
        text = text.replace(ligature, replacement)

    text = RE_NON_PRINTABLE.sub("", text)
    text = RE_MULTI_WHITESPACE.sub(" ", text).strip()
    return text

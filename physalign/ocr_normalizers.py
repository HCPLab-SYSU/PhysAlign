"""Frozen OCR-only normalization for PhysAlign 2.1.

No numerical evaluation, unit conversion, case folding or fuzzy matching.
A None result denotes a malformed/unsupported model field; invalid gold fields
are an export error. Original strings are never mutated in stored artifacts.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

NORMALIZER_IDS = ("ocr_literal_v2_1", "ocr_label_v2_1")
# The unit list is lexical, case-sensitive and deliberately small. It grants no
# physics-value conversion and does not relabel m as M, kg as Kg, or N as n.
UNITS = ("kg", "g", "mg", "m", "cm", "mm", "km", "s", "ms", "A", "mA", "V", "mV", "N", "J", "W", "C", "T", "Hz", "K", "Pa", "Ω")
NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
NUMERIC_UNIT = re.compile(r"(" + NUMBER + r")[ \t\u00a0\u202f]*(" + "|".join(map(re.escape, sorted(UNITS, key=len, reverse=True))) + r")")
WRAPPER = re.compile(r"\\(?:mathrm|text)\{([A-Za-z]+)\}")
OUTER_PAIRS = (("$$", "$$"), ("\\(", "\\)"), ("\\[", "\\]"), ("$", "$"))


def meaningful_text(value: Any) -> bool:
    """Reject blank and zero-width/format/control-only observations, unchanged."""
    return isinstance(value, str) and any(
        not c.isspace() and unicodedata.category(c)[0] not in ("C", "Z")
        for c in value
    )


def normalize(value: Any, normalizer_id: str) -> str | None:
    if normalizer_id not in NORMALIZER_IDS:
        raise ValueError("UNKNOWN_NORMALIZER: " + str(normalizer_id))
    if not meaningful_text(value):
        return None
    s = unicodedata.normalize("NFC", value).strip()
    if normalizer_id == "ocr_literal_v2_1":
        return s
    # Label profile is a restricted typography grammar, not a LaTeX parser.
    for opening, closing in OUTER_PAIRS:
        if s.startswith(opening):
            if not s.endswith(closing) or len(s) <= len(opening) + len(closing):
                return None
            s = s[len(opening):-len(closing)].strip()
            break  # Only ONE complete external math-delimiter pair is allowed.
    if "$" in s:
        return None
    s = WRAPPER.sub(lambda m: m.group(1), s)
    for spacing in (r"\,", r"\;", r"\:", r"\!", r"\ "):
        s = s.replace(spacing, " ")
    # Unsupported commands/wrappers fail, rather than silently losing semantics
    # (e.g. \vec, \mathbf, \frac). Braces in m_{1} remain literal, not erased.
    if "\\" in s:
        return None
    s = s.replace("\u2212", "-").strip()
    if not meaningful_text(s):
        return None
    m = NUMERIC_UNIT.fullmatch(s)
    return m.group(1) + " " + m.group(2) if m else s


def equivalent(prediction: Any, gold: str, normalizer_id: str) -> bool:
    g = normalize(gold, normalizer_id)
    if g is None:
        raise ValueError("UNSUPPORTED_GOLD_READING: " + normalizer_id)
    p = normalize(prediction, normalizer_id)
    return p is not None and p == g


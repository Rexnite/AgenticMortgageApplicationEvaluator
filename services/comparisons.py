"""Pure exact/normalised comparisons used before semantic LLM reasoning."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable


def normalize_text(value: str | None) -> str | None:
    if not value or not value.strip():
        return None
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", normalized.casefold()) or None


def normalized_match(first: str | None, second: str | None) -> bool | None:
    left, right = normalize_text(first), normalize_text(second)
    return None if left is None or right is None else left == right


def all_normalized_match(values: Iterable[str | None]) -> bool | None:
    normalized = [normalize_text(value) for value in values]
    if len(normalized) < 2 or any(value is None for value in normalized):
        return None
    return len(set(normalized)) == 1

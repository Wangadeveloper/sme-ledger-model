"""Robust JSON extraction and canonicalization for model output parsing."""

from __future__ import annotations

import json
from typing import Any, Optional


def extract_json_value(text: str) -> Optional[Any]:
    """Extracts a valid JSON object or list from text, even if surrounded by markdown or commentary."""
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    # Fast-path: entire string is valid JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Search for matching bracket pairs
    starts = [(i, ch) for i, ch in enumerate(text) if ch in "{["]

    for start, opening in starts:
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break

    return None


def canonical_value(value: Any) -> str:
    """Produces a deterministic, canonical string representation of a value or data structure."""
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return str(value).strip()

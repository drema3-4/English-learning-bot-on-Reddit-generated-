from __future__ import annotations

import json
import re
from typing import Any


class JSONParseError(ValueError):
    pass


JsonParseError = JSONParseError


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise JSONParseError("Empty JSON payload")

    direct = _try_parse_object(stripped)
    if direct is not None:
        return direct

    for block in re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE):
        parsed = _try_parse_object(block.strip())
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise JSONParseError("Could not parse a JSON object")


def parse_json_array(raw: str) -> list[dict[str, Any]]:
    stripped = raw.strip()
    if not stripped:
        raise JSONParseError("Empty JSON payload")

    direct = _try_parse_array(stripped)
    if direct is not None:
        return direct

    for block in re.findall(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE):
        parsed = _try_parse_array(block.strip())
        if parsed is not None:
            return parsed

    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end > start:
        parsed = _try_parse_array(stripped[start : end + 1])
        if parsed is not None:
            return parsed

    raise JSONParseError("Could not parse a JSON array")


def _try_parse_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _try_parse_array(value: str) -> list[dict[str, Any]] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed
    return None

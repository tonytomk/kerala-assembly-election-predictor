from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_OVERRIDE_PATH = Path(__file__).with_name("constituency_overrides.json")


@lru_cache(maxsize=1)
def load_constituency_overrides() -> dict[str, list[dict[str, Any]]]:
    if not DEFAULT_OVERRIDE_PATH.exists():
        return {}

    payload = json.loads(DEFAULT_OVERRIDE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}

    cleaned: dict[str, list[dict[str, Any]]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            cleaned[key] = [item for item in value if isinstance(item, dict)]
    return cleaned


def get_override_rows(section: str) -> list[dict[str, Any]]:
    return list(load_constituency_overrides().get(section, []))

"""Small persisted UI/settings helpers."""

from __future__ import annotations

from loha import cache

INDUSTRY_FILTER_KEY = "industry_filter"


def excluded_industries() -> list[str]:
    data = cache.get_json("settings", INDUSTRY_FILTER_KEY) or {}
    values = data.get("excluded_industries", [])
    if not isinstance(values, list):
        return []
    return sorted({str(v) for v in values if str(v).strip()})


def set_excluded_industries(values: list[str]) -> list[str]:
    cleaned = sorted({str(v).strip() for v in values if str(v).strip()})
    cache.put_json("settings", {"excluded_industries": cleaned}, INDUSTRY_FILTER_KEY)
    return cleaned

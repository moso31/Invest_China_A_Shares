"""File-based pickle cache for AKShare responses.

Each cache key maps to one .pkl file. Reads return None when stale or
missing; the caller is expected to fetch fresh data and `put()` it.
Pickle keeps the cache layer dependency-free (no pyarrow / fastparquet).
"""

from __future__ import annotations

import json
import pickle
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from loha import request_stats
from loha.config import CACHE_DIR, CACHE_TTL_HOURS

_ALLOW_STALE_READ_DEPTH = 0


@contextmanager
def allow_stale_reads():
    """Temporarily read existing cache files even when TTL has expired."""
    global _ALLOW_STALE_READ_DEPTH
    _ALLOW_STALE_READ_DEPTH += 1
    try:
        yield
    finally:
        _ALLOW_STALE_READ_DEPTH = max(0, _ALLOW_STALE_READ_DEPTH - 1)


def _safe_key(parts: list[str]) -> str:
    return "_".join(p.replace("/", "_").replace("\\", "_") for p in parts)


def _path(kind: str, key: str) -> Path:
    sub = CACHE_DIR / kind
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{key}.pkl"


def _meta_path(kind: str, key: str) -> Path:
    return _path(kind, key).with_suffix(".meta.json")


def _is_fresh(meta_file: Path, ttl_hours: float) -> bool:
    if not meta_file.exists():
        return False
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    try:
        ttl_hours = float(meta.get("ttl_hours", ttl_hours))
    except (TypeError, ValueError):
        pass
    age_h = (time.time() - float(meta.get("ts", 0))) / 3600
    return age_h <= ttl_hours


def get(kind: str, *key_parts: str) -> pd.DataFrame | None:
    """Return cached DataFrame or None if missing/stale."""
    key = _safe_key(list(key_parts))
    path = _path(kind, key)
    meta = _meta_path(kind, key)
    ttl = CACHE_TTL_HOURS.get(kind, 24)
    stale_allowed = _ALLOW_STALE_READ_DEPTH > 0
    if not path.exists():
        request_stats.record_cache_read(kind, hit=False)
        return None
    fresh = _is_fresh(meta, ttl)
    if not stale_allowed and not fresh:
        request_stats.record_cache_read(kind, hit=False)
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        request_stats.record_cache_read(kind, hit=True, stale=not fresh)
        return data
    except Exception:
        request_stats.record_cache_read(kind, hit=False)
        return None


def get_stale(kind: str, *key_parts: str) -> pd.DataFrame | None:
    """Return an existing cache file even if its TTL has expired."""
    with allow_stale_reads():
        return get(kind, *key_parts)


def put(
    kind: str,
    df: pd.DataFrame,
    *key_parts: str,
    allow_empty: bool = False,
    ttl_hours: float | None = None,
) -> None:
    if df is None or (df.empty and not allow_empty):
        return
    key = _safe_key(list(key_parts))
    path = _path(kind, key)
    meta = _meta_path(kind, key)
    try:
        with open(path, "wb") as fh:
            pickle.dump(df, fh, protocol=pickle.HIGHEST_PROTOCOL)
        meta_data: dict[str, Any] = {"ts": time.time()}
        if df.empty:
            meta_data["empty"] = True
        if ttl_hours is not None:
            meta_data["ttl_hours"] = ttl_hours
        meta.write_text(json.dumps(meta_data), encoding="utf-8")
        request_stats.record_cache_write(kind, empty=bool(df.empty))
    except Exception:
        # Cache is best-effort; never let it break a screen run.
        pass


def put_empty(kind: str, *key_parts: str, ttl_hours: float = 1.0) -> None:
    """Cache a short-lived empty result so failing endpoints are not retried on every view."""
    put(kind, pd.DataFrame(), *key_parts, allow_empty=True, ttl_hours=ttl_hours)


def get_json(kind: str, *key_parts: str) -> Any | None:
    key = _safe_key(list(key_parts))
    sub = CACHE_DIR / kind
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{key}.json"
    meta = path.with_suffix(".meta.json")
    ttl = CACHE_TTL_HOURS.get(kind, 24)
    stale_allowed = _ALLOW_STALE_READ_DEPTH > 0
    if not path.exists():
        request_stats.record_cache_read(kind, hit=False)
        return None
    fresh = _is_fresh(meta, ttl)
    if not stale_allowed and not fresh:
        request_stats.record_cache_read(kind, hit=False)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        request_stats.record_cache_read(kind, hit=True, stale=not fresh)
        return data
    except Exception:
        request_stats.record_cache_read(kind, hit=False)
        return None


def get_stale_json(kind: str, *key_parts: str) -> Any | None:
    """Return an existing JSON cache file even if its TTL has expired."""
    with allow_stale_reads():
        return get_json(kind, *key_parts)


def put_json(kind: str, data: Any, *key_parts: str, ttl_hours: float | None = None) -> None:
    key = _safe_key(list(key_parts))
    sub = CACHE_DIR / kind
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{key}.json"
    meta = path.with_suffix(".meta.json")
    try:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        meta_data: dict[str, Any] = {"ts": time.time()}
        if ttl_hours is not None:
            meta_data["ttl_hours"] = ttl_hours
        meta.write_text(json.dumps(meta_data), encoding="utf-8")
        request_stats.record_cache_write(kind, empty=data in (None, {}, []))
    except Exception:
        pass


def clear_kind(kind: str) -> None:
    """Delete one cache namespace under CACHE_DIR."""
    root = CACHE_DIR.resolve()
    target = (CACHE_DIR / kind).resolve()
    if target != root and root not in target.parents:
        return
    if target.exists():
        shutil.rmtree(target)


def clear_data_sources() -> None:
    """Clear raw AKShare caches for the complete full-market refresh.

    Fast full-market refresh intentionally does not call this; it preserves
    per-symbol raw data so the full Q/D/V/T/R scoring chain can reuse it.
    """
    for kind in ("universe", "price_hist", "indicator", "financial", "dividend", "info", "bulk"):
        clear_kind(kind)


def delete(kind: str, *key_parts: str) -> None:
    """Delete a specific cache entry and its metadata."""
    key = _safe_key(list(key_parts))
    for path in (_path(kind, key), _meta_path(kind, key)):
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def delete_json(kind: str, *key_parts: str) -> None:
    """Delete a specific JSON cache entry and its metadata."""
    key = _safe_key(list(key_parts))
    sub = CACHE_DIR / kind
    path = sub / f"{key}.json"
    meta = path.with_suffix(".meta.json")
    for item in (path, meta):
        try:
            if item.exists():
                item.unlink()
        except Exception:
            pass

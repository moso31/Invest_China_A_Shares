"""Lightweight per-run HTTP request accounting."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from urllib.parse import urlparse

_LOCK = threading.Lock()
_ACTIVE = False
_START_TS = 0.0
_STATS: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0, "failed": 0})
_API_STATS: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"total": 0, "success": 0, "failed": 0})
_CACHE_STATS: dict[str, dict[str, int]] = defaultdict(
    lambda: {"reads": 0, "hits": 0, "misses": 0, "stale_hits": 0, "writes": 0, "empty_writes": 0}
)


def reset(active: bool = True) -> None:
    global _ACTIVE, _START_TS
    with _LOCK:
        _ACTIVE = active
        _START_TS = time.time()
        _STATS.clear()
        _API_STATS.clear()
        _CACHE_STATS.clear()


def stop() -> None:
    global _ACTIVE
    with _LOCK:
        _ACTIVE = False


def record(url: str, ok: bool) -> None:
    with _LOCK:
        if not _ACTIVE:
            return
        key = (_provider(url), _endpoint(url))
        row = _STATS[key]
        row["total"] += 1
        if ok:
            row["success"] += 1
        else:
                row["failed"] += 1


def record_api_call(name: str, ok: bool, provider: str | None = None) -> None:
    with _LOCK:
        if not _ACTIVE:
            return
        endpoint = str(name or "unknown")
        row = _API_STATS[(provider or _provider_from_api_name(endpoint), endpoint)]
        row["total"] += 1
        if ok:
            row["success"] += 1
        else:
            row["failed"] += 1


def record_cache_read(kind: str, hit: bool, stale: bool = False) -> None:
    with _LOCK:
        if not _ACTIVE:
            return
        row = _CACHE_STATS[str(kind or "unknown")]
        row["reads"] += 1
        if hit:
            row["hits"] += 1
            if stale:
                row["stale_hits"] += 1
        else:
            row["misses"] += 1


def record_cache_write(kind: str, empty: bool = False) -> None:
    with _LOCK:
        if not _ACTIVE:
            return
        row = _CACHE_STATS[str(kind or "unknown")]
        row["writes"] += 1
        if empty:
            row["empty_writes"] += 1


def snapshot() -> dict:
    with _LOCK:
        elapsed_s = round(time.time() - _START_TS, 1) if _START_TS else 0.0
        provider_rows: dict[str, dict[str, int]] = {}
        endpoint_rows: list[dict] = []
        for (provider, endpoint), row in sorted(_STATS.items()):
            provider_row = provider_rows.setdefault(provider, {"total": 0, "success": 0, "failed": 0})
            for key in ("total", "success", "failed"):
                provider_row[key] += int(row[key])
            endpoint_rows.append(
                {
                    "provider": provider,
                    "endpoint": endpoint,
                    "total": int(row["total"]),
                    "success": int(row["success"]),
                    "failed": int(row["failed"]),
                }
            )
        providers = [
            {"provider": provider, **row}
            for provider, row in sorted(provider_rows.items(), key=lambda item: item[0])
        ]
        api_calls = [
            {
                "provider": provider,
                "endpoint": endpoint,
                "total": int(row["total"]),
                "success": int(row["success"]),
                "failed": int(row["failed"]),
            }
            for (provider, endpoint), row in sorted(_API_STATS.items())
        ]
        cache_rows = [
            {"kind": kind, **{key: int(value) for key, value in row.items()}}
            for kind, row in sorted(_CACHE_STATS.items(), key=lambda item: item[0])
        ]
        return {
            "active": _ACTIVE,
            "elapsed_s": elapsed_s,
            "providers": providers,
            "endpoints": endpoint_rows,
            "api_calls": api_calls,
            "cache": cache_rows,
        }


def _provider(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host.endswith("eastmoney.com"):
        return "东方财富"
    if host.endswith("qq.com") or host.endswith("gtimg.cn"):
        return "腾讯"
    if host.endswith("xueqiu.com"):
        return "雪球"
    if "10jqka" in host or "iwencai" in host:
        return "同花顺"
    if host.endswith("cninfo.com.cn"):
        return "巨潮"
    if host.endswith("sina.com.cn") or host.endswith("sinajs.cn"):
        return "新浪"
    return host or "其他"


def _provider_from_api_name(name: str) -> str:
    lower = name.lower()
    if lower.endswith("_ths") or "ths" in lower:
        return "同花顺"
    if "cninfo" in lower:
        return "巨潮"
    if "xq" in lower or "xueqiu" in lower:
        return "雪球"
    if "tx" in lower or "tencent" in lower:
        return "腾讯"
    if lower.endswith("_em") or "eastmoney" in lower:
        return "东方财富"
    if lower in {"stock_zh_a_hist", "index_zh_a_hist", "stock_info_a_code_name"}:
        return "东方财富"
    return "AKShare"


def _endpoint(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path or "/"
    if len(path) > 72:
        path = path[:69] + "..."
    return f"{host}{path}"

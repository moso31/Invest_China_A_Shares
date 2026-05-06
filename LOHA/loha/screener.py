"""Screening pipeline.

Two-step process described in Strategy.md:
  1. Hard gates: drop ST, low-liquidity, too-young.
  2. estimate.md strategy fit: 0.45D + 0.35V + 0.20T, ranked descending.

AKShare endpoints rate-limit per-IP, so the worker count is intentionally
bounded by config.SCREENER_MAX_WORKERS.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from loha import bulk, cache, config, factors, request_stats, scoring, source
from loha.scoring import StockScore

log = logging.getLogger("loha.screener")


@dataclass
class ScreenProgress:
    mode: str = "curated"
    refresh_mode: str = ""
    refresh_mode_label: str = ""
    cache_used: bool = False
    source_total: int = 0
    prefiltered_total: int = 0
    total: int = 0
    done: int = 0
    accepted: int = 0
    rejected: int = 0
    failed: int = 0
    current_symbol: str = ""
    state: str = "idle"  # idle | running | done | error
    started_ts: float = 0.0
    error: str | None = None
    results: list[dict] | None = None


_progress = ScreenProgress()

FULL_SCAN_RESULTS_KEY = "full_scan_results"
FULL_SCAN_META_KEY = "full_scan_meta"
LAST_RUN_KEY = "last_run"

# Refresh modes only control raw-cache invalidation. They must not change
# _score_one(), factors.all_factors(), or scoring.composite(), because ranking
# depends on the complete Q/D/V/T/R factor chain and all of its sub-metrics.
FULL_REFRESH_FAST = "fast"
FULL_REFRESH_COMPLETE = "complete"


def progress() -> ScreenProgress:
    return _progress


def _normalise_mode(mode: str | None) -> str:
    if mode in {"all", "curated"}:
        return mode
    return config.UNIVERSE_MODE


def invalidate_stale_score_caches() -> bool:
    """Drop persisted score rows when their scoring rule version is stale."""
    stale_versions: list[int] = []
    meta = cache.get_json("screener", FULL_SCAN_META_KEY)
    if isinstance(meta, dict):
        meta_version = int(meta.get("scoring_rule_version") or 0)
        if meta_version and meta_version != config.SCORING_RULE_VERSION:
            stale_versions.append(meta_version)

    for key in (FULL_SCAN_RESULTS_KEY, LAST_RUN_KEY):
        rows = cache.get_json("screener", key)
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row_version = int(row.get("scoring_rule_version") or 0)
                if row_version and row_version != config.SCORING_RULE_VERSION:
                    stale_versions.append(row_version)
                    break

    if not stale_versions:
        return False

    cache.delete_json("screener", FULL_SCAN_RESULTS_KEY)
    cache.delete_json("screener", LAST_RUN_KEY)
    cache.delete_json("screener", FULL_SCAN_META_KEY)
    _progress.results = None
    log.warning(
        "cleared stale screener score caches: cached scoring_rule_version=%s current=%s",
        sorted(set(stale_versions)),
        config.SCORING_RULE_VERSION,
    )
    return True


def _dirty_score_row(row: dict, error: Exception, mode: str) -> dict:
    symbol = str(row.get("symbol") or "").zfill(6)
    name = str(row.get("name") or symbol)
    msg = f"score_cache_dirty: scoring rule cache repair failed in {mode} mode: {error}"
    out = dict(row)
    out.update(
        {
            "symbol": symbol,
            "name": name,
            "score": 0.0,
            "Q": 0.0,
            "D": 0.0,
            "V": 0.0,
            "T": 0.0,
            "R": 100.0,
            "components": {},
            "flags": [msg],
            "risk_flags": [msg],
            "good_flags": [],
            "dirty": True,
            "recompute_error": str(error),
        }
    )
    return out


def _score_row_needs_repair(row: dict) -> bool:
    """Return True only for legacy cache rows that lack required schema fields.

    Empty `div_year_sources` is a valid current-schema value for stocks with no
    usable dividend/price data. Treating it as missing makes every cached scan
    repeatedly recompute those rows.
    """
    if int(row.get("scoring_rule_version") or 0) != config.SCORING_RULE_VERSION:
        return True
    components = dict(row.get("components") or {})
    d_comp = dict(components.get("D") or {})
    v_comp = dict(components.get("V") or {})
    return "div_year_sources" not in d_comp or ("bucket" not in v_comp and "bucket_effective" not in v_comp)


def _repair_legacy_dividend_row(row: dict) -> tuple[dict, bool]:
    """Upgrade cached rows after factor-rule changes when raw cache is present."""
    if not _score_row_needs_repair(row):
        return row, False

    symbol = str(row.get("symbol") or "").zfill(6)
    name = str(row.get("name") or symbol)
    if not symbol or symbol == "000000":
        return row, False
    try:
        f = factors.all_factors(symbol, name)
        selected = {k: f[k] for k in ("Q", "D", "V", "T", "R")}
        s = scoring.composite(symbol, name, selected)
        return s.to_dict(), True
    except Exception as e:  # noqa: BLE001
        mode = "cache-only" if source.is_cache_only() else "fetch-enabled"
        log.warning(
            "score cache repair failed for %s in %s mode; marked row dirty: %s",
            symbol,
            mode,
            e,
        )
        return _dirty_score_row(row, e, mode), True


def _repair_legacy_dividend_rows(rows: list[dict]) -> tuple[list[dict], bool]:
    repaired: list[dict] = []
    changed = False
    for row in rows:
        row_changed = False
        with source.cache_only():
            new_row, row_changed = _repair_legacy_dividend_row(row)
        changed = changed or row_changed
        if new_row.get("dirty"):
            new_row, row_changed = _repair_legacy_dividend_row(row)
            changed = changed or row_changed
        repaired.append(new_row)
    if changed:
        repaired.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
    return repaired, changed


def _score_from_dict(row: dict) -> StockScore:
    if _score_row_needs_repair(row):
        with source.cache_only():
            row, _changed = _repair_legacy_dividend_row(row)
        if row.get("dirty"):
            row, _changed = _repair_legacy_dividend_row(row)
    legacy_flags = list(row.get("flags") or [])
    risk_flags = list(row.get("risk_flags") or [])
    good_flags = list(row.get("good_flags") or [])
    if not risk_flags and not good_flags and legacy_flags:
        risk_flags, good_flags = scoring.split_signals(legacy_flags)
    elif not risk_flags:
        risk_flags = legacy_flags
    risk_flags = scoring.filter_active_signals(risk_flags)
    good_flags = scoring.dedupe_signals(good_flags + scoring.derive_good_flags(row))
    return StockScore(
        symbol=str(row["symbol"]),
        name=str(row.get("name", row["symbol"])),
        score=float(row.get("score", 0)),
        Q=float(row.get("Q", 0)),
        D=float(row.get("D", 0)),
        V=float(row.get("V", 0)),
        T=float(row.get("T", 0)),
        R=float(row.get("R", 0)),
        flags=risk_flags,
        components=dict(row.get("components") or {}),
        good_flags=good_flags,
    )


def full_scan_meta() -> dict | None:
    return cache.get_json("screener", FULL_SCAN_META_KEY)


def _load_cached_full_scan() -> tuple[list[StockScore], dict] | None:
    if invalidate_stale_score_caches():
        return None
    rows = cache.get_json("screener", FULL_SCAN_RESULTS_KEY)
    meta = full_scan_meta() or {}
    if not rows:
        return None
    rows, changed = _repair_legacy_dividend_rows(rows)
    if changed:
        meta = {**meta, "scoring_rule_version": config.SCORING_RULE_VERSION}
        cache.put_json("screener", rows, FULL_SCAN_RESULTS_KEY)
        cache.put_json("screener", rows, "last_run")
        cache.put_json("screener", meta, FULL_SCAN_META_KEY)
    return [_score_from_dict(row) for row in rows], meta


def _cached_full_scan_rows() -> list[dict]:
    rows = cache.get_json("screener", FULL_SCAN_RESULTS_KEY)
    return rows if isinstance(rows, list) else []


def _prefilter_all_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cheap hard gates before expensive per-stock factor calls."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    if "name" in out.columns:
        names = out["name"].astype(str)
        out = out[~names.str.contains("ST|退", regex=True, na=False) & ~names.str.startswith("N", na=False)]
    if "last" in out.columns:
        last = pd.to_numeric(out["last"], errors="coerce")
        if last.notna().any():
            out = out[last.isna() | (last > 0)]
    if "amount" in out.columns:
        amount = pd.to_numeric(out["amount"], errors="coerce")
        if amount.notna().any():
            out = out[amount.isna() | (amount >= config.LIQUIDITY_MIN_TURNOVER_CNY)]
    if "total_mv" in out.columns:
        total_mv = pd.to_numeric(out["total_mv"], errors="coerce")
        if total_mv.notna().any():
            out = out[total_mv.isna() | (total_mv >= config.MIN_TOTAL_MV_CNY)]
    return out


def _universe_row(row: pd.Series | dict) -> dict:
    def value(key: str, default=None):
        if isinstance(row, dict):
            item = row.get(key, default)
        else:
            item = row[key] if key in row else default
        return None if pd.isna(item) else item

    symbol = str(value("symbol", "")).zfill(6)
    name = str(value("name", symbol) or symbol)
    return {
        "symbol": symbol,
        "name": name,
        "amount": value("amount"),
        "total_mv": value("total_mv"),
        "turnover_rate": value("turnover_rate"),
        "last": value("last"),
    }


def _build_universe(mode: str | None = None) -> tuple[list[dict], int, int]:
    """Return spot-like rows for the configured universe."""
    mode = _normalise_mode(mode)
    if mode == "all":
        df = source.realtime_snapshot()
        degraded_to_code_name_universe = False
        if df is None or df.empty:
            df = source.universe_all()
            degraded_to_code_name_universe = df is not None and not df.empty
            if degraded_to_code_name_universe:
                log.warning(
                    "realtime snapshot unavailable; degraded to stock_info_a_code_name universe with no spot prefilter fields"
                )
        if df is None or df.empty:
            return [], 0, 0
        if degraded_to_code_name_universe:
            df = df.copy()
            for column in ("amount", "total_mv", "turnover_rate", "last"):
                if column not in df.columns:
                    df[column] = None
        source_total = len(df)
        df = _prefilter_all_universe(df)
        universe = [_universe_row(r) for _, r in df.iterrows()]
        return universe, source_total, len(universe)

    # Curated list - look up names from the global universe if available
    name_lookup: dict[str, str] = {}
    df = source.universe_all()
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            name_lookup[str(r["symbol"]).zfill(6)] = str(r["name"])
    universe = [_universe_row({"symbol": s, "name": name_lookup.get(s, s)}) for s in config.DEFAULT_UNIVERSE]
    return universe, len(universe), len(universe)


def _passes_hard_gates(symbol: str, name: str) -> tuple[bool, list[str]]:
    """ST/退/N filter on the name; liquidity gate is checked alongside T-factor."""
    flags: list[str] = []
    if config.EXCLUDE_ST and (any(tag in name for tag in ("ST", "*ST", "退")) or name.startswith("N")):
        flags.append("ST/退市风险")
        return False, flags
    return True, flags


def _preheat_bulk_caches(force: bool = False) -> None:
    """Prepare market-wide bulk tables used by the normal Q/D/V/T/R factor chain.

    Fast full-market refresh calls this without clearing per-symbol caches, so
    existing single-stock raw data can be reused. Complete full-market refresh
    clears all raw data first, including these bulk caches, and then rebuilds
    them from AKShare.
    """
    periods = bulk._recent_periods(n_years=5)
    if force:
        cache.clear_kind("bulk")
    bulk.bulk_yjbb(periods)
    bulk.bulk_fhps(periods)


def _refresh_spot_snapshot_preserving_stale() -> None:
    """Refresh the candidate-pool snapshot without discarding the stale fallback.

    The fast full-market refresh must still use the same scoring logic as a
    complete refresh. Its speed comes from not purging per-symbol caches; this
    snapshot refresh is only for the cheap prefilter and spot liquidity gates.
    """
    stale = cache.get_stale("price_hist", "realtime_snapshot")
    cache.delete("price_hist", "realtime_snapshot")
    fresh = source.realtime_snapshot()
    if fresh is None or fresh.empty:
        if source.is_complete_realtime_snapshot(stale):
            log.warning("fast full-market refresh: realtime snapshot fetch failed; restored stale snapshot cache")
            cache.put("price_hist", stale, "realtime_snapshot")
        else:
            log.warning(
                "fast refresh: realtime snapshot fetch failed and no stale cache available; downstream will degrade to code-name universe"
            )


def _normalise_refresh_mode(refresh_mode: str | None) -> str:
    if refresh_mode in {FULL_REFRESH_FAST, FULL_REFRESH_COMPLETE}:
        return refresh_mode
    return FULL_REFRESH_FAST


def _spot_amount_too_low(spot: dict) -> bool:
    amount = pd.to_numeric(pd.Series([spot.get("amount")]), errors="coerce").fillna(float("nan")).iloc[0]
    return pd.notna(amount) and float(amount) < config.LIQUIDITY_MIN_TURNOVER_CNY * 0.5


def _score_one(spot: dict) -> tuple[str, StockScore | None]:
    symbol = str(spot.get("symbol") or "").zfill(6)
    name = str(spot.get("name") or symbol)
    ok, _gate_flags = _passes_hard_gates(symbol, name)
    if not ok:
        return "rejected", None
    if _spot_amount_too_low(spot):
        log.info("drop %s by spot liquidity gate: %s", symbol, spot.get("amount"))
        return "rejected", None
    try:
        f = factors.all_factors(symbol, name)
        turnover = f["T"].components.get("avg_turnover_cny")
        if turnover is not None and turnover < config.LIQUIDITY_MIN_TURNOVER_CNY:
            log.info("drop %s by liquidity gate: %.0f", symbol, turnover)
            return "rejected", None
        s = scoring.composite(symbol, name, {k: f[k] for k in ("Q", "D", "V", "T", "R")})
        return "accepted", s
    except Exception as e:  # noqa: BLE001
        log.warning("scoring failed for %s: %s", symbol, e)
        return "failed", None


def run(
    top_n: int | None = None,
    on_progress: Callable[[ScreenProgress], None] | None = None,
    mode: str | None = None,
    force_refresh: bool = False,
    refresh_mode: str = FULL_REFRESH_FAST,
) -> list[StockScore]:
    scan_mode = _normalise_mode(mode)
    refresh_mode = _normalise_refresh_mode(refresh_mode)
    request_stats_active = scan_mode == "all" and force_refresh
    if request_stats_active:
        request_stats.reset(active=True)
    elif scan_mode == "all":
        request_stats.reset(active=False)
    _progress.mode = scan_mode
    if scan_mode == "all" and force_refresh:
        _progress.refresh_mode = refresh_mode
        _progress.refresh_mode_label = "完全更新" if refresh_mode == FULL_REFRESH_COMPLETE else "快速更新"
    elif scan_mode == "all":
        _progress.refresh_mode = "cached_or_normal"
        _progress.refresh_mode_label = "复用本地结果"
    else:
        _progress.refresh_mode = ""
        _progress.refresh_mode_label = ""
    _progress.cache_used = False
    _progress.source_total = 0
    _progress.prefiltered_total = 0
    _progress.total = 0
    _progress.done = 0
    _progress.accepted = 0
    _progress.rejected = 0
    _progress.failed = 0
    _progress.current_symbol = "准备扫描范围"
    _progress.state = "running"
    _progress.started_ts = time.time()
    _progress.error = None
    _progress.results = None

    results: list[StockScore] = []
    try:
        if scan_mode == "all" and not force_refresh:
            cached = _load_cached_full_scan()
            if cached is not None:
                results, meta = cached
                result_rows = [r.to_dict() for r in results]
                _progress.cache_used = True
                _progress.source_total = int(meta.get("source_total", 0))
                _progress.prefiltered_total = int(meta.get("prefiltered_total", 0))
                _progress.total = int(meta.get("total", len(result_rows)))
                _progress.done = int(meta.get("done", _progress.total))
                _progress.accepted = int(meta.get("accepted", len(result_rows)))
                _progress.rejected = int(meta.get("rejected", 0))
                _progress.failed = int(meta.get("failed", 0))
                _progress.current_symbol = "已复用本地全盘扫描结果"
                _progress.state = "done"
                _progress.results = result_rows
                cache.put_json("screener", result_rows, LAST_RUN_KEY)
                return results
            _progress.state = "error"
            _progress.error = "暂无本地全盘扫描结果，请先点击“强制更新全 A 股数据”。"
            _progress.results = []
            return []

        # Fast and complete full-market refreshes share the exact same scoring
        # path below. Fast keeps per-symbol raw caches and refreshes only the
        # candidate/bulk inputs; complete deletes raw caches and rebuilds them.
        if scan_mode == "all" and force_refresh and refresh_mode == FULL_REFRESH_COMPLETE:
            _progress.current_symbol = "完整强刷：清理所有原始数据缓存"
            cache.clear_data_sources()
            _progress.current_symbol = "完整强刷：预热全市场批量宽表"
            _preheat_bulk_caches(force=False)
        elif scan_mode == "all" and force_refresh:
            _progress.current_symbol = "快速强刷：保留单股缓存，刷新候选池快照"
            _refresh_spot_snapshot_preserving_stale()
            _progress.current_symbol = "快速强刷：确认全市场批量宽表缓存"
            _preheat_bulk_caches(force=False)

        universe, source_total, prefiltered_total = _build_universe(scan_mode)
        if scan_mode == "all" and force_refresh and not universe:
            raise RuntimeError("全市场快照与代码名单均获取失败，疑似网络/代理故障。")
        _progress.source_total = source_total
        _progress.prefiltered_total = prefiltered_total
        _progress.total = len(universe)
        _progress.current_symbol = "开始评分"
        if on_progress:
            on_progress(_progress)

        with ThreadPoolExecutor(max_workers=config.SCREENER_MAX_WORKERS) as ex:
            futures = {ex.submit(_score_one, row): row for row in universe}
            for fut in as_completed(futures):
                row = futures[fut]
                sym = str(row.get("symbol") or "").zfill(6)
                _progress.current_symbol = sym
                try:
                    status, s = fut.result()
                except Exception as e:  # noqa: BLE001
                    log.warning("worker error %s: %s", sym, e)
                    status = "failed"
                    s = None
                if s is not None:
                    results.append(s)
                if status == "accepted":
                    _progress.accepted += 1
                elif status == "rejected":
                    _progress.rejected += 1
                else:
                    _progress.failed += 1
                _progress.done += 1
                if on_progress:
                    on_progress(_progress)
        results.sort(key=lambda r: r.score, reverse=True)
        if top_n:
            results = results[:top_n]
        _progress.state = "done"
        _progress.results = [r.to_dict() for r in results]
        if scan_mode == "all":
            meta = {
                "status": "扫描完毕",
                "success_ts": time.time(),
                "success_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "scoring_rule_version": config.SCORING_RULE_VERSION,
                "source_total": _progress.source_total,
                "prefiltered_total": _progress.prefiltered_total,
                "total": _progress.total,
                "done": _progress.done,
                "accepted": _progress.accepted,
                "rejected": _progress.rejected,
                "failed": _progress.failed,
                "refresh_mode": _progress.refresh_mode,
                "refresh_mode_label": _progress.refresh_mode_label,
                "duration_s": round(time.time() - _progress.started_ts, 1) if _progress.started_ts else 0,
                "request_stats": request_stats.snapshot(),
            }
            if request_stats_active:
                request_stats.stop()
            cache.put_json("screener", _progress.results, FULL_SCAN_RESULTS_KEY)
            cache.put_json("screener", meta, FULL_SCAN_META_KEY)
        # Persist last run for the UI to read on refresh
        cache.put_json("screener", _progress.results, LAST_RUN_KEY)
    except Exception as e:  # noqa: BLE001
        if request_stats_active:
            request_stats.stop()
        _progress.state = "error"
        _progress.error = str(e)
        log.exception("screener failed")
    return results


def recompute_cached_full_scan(
    on_progress: Callable[[ScreenProgress], None] | None = None,
    cache_only: bool = True,
) -> list[StockScore]:
    """Recompute full-scan scores from local cached raw data.

    This is a debug/maintenance path for factor-rule changes. It does not
    clear AKShare caches and, by default, it will not fetch missing data.
    """
    old_rows = _cached_full_scan_rows()
    old_meta = full_scan_meta() or {}

    _progress.mode = "debug_recompute"
    _progress.cache_used = True
    _progress.source_total = int(old_meta.get("source_total", len(old_rows)))
    _progress.prefiltered_total = int(old_meta.get("prefiltered_total", len(old_rows)))
    _progress.total = 0
    _progress.done = 0
    _progress.accepted = 0
    _progress.rejected = 0
    _progress.failed = 0
    _progress.current_symbol = "准备重算评分"
    _progress.state = "running"
    _progress.started_ts = time.time()
    _progress.error = None
    _progress.results = None

    if not old_rows:
        _progress.state = "error"
        _progress.error = "暂无全盘扫描结果缓存，无法执行调试重算。请先完成一次全 A 股扫描。"
        _progress.results = []
        return []

    context = source.cache_only() if cache_only else nullcontext()

    results: list[StockScore] = []
    try:
        if not cache_only:
            periods = bulk._recent_periods(n_years=5)
            bulk.bulk_yjbb(periods)
            bulk.bulk_fhps(periods)
        with context:
            universe, source_total, prefiltered_total = _build_universe("all")
            if not universe:
                universe = [
                    _universe_row(
                        {
                            "symbol": str(row["symbol"]).zfill(6),
                            "name": str(row.get("name") or row["symbol"]),
                        }
                    )
                    for row in old_rows
                    if row.get("symbol")
                ]
                source_total = int(old_meta.get("source_total", len(universe)))
                prefiltered_total = int(old_meta.get("prefiltered_total", len(universe)))

            _progress.source_total = source_total
            _progress.prefiltered_total = prefiltered_total
            _progress.total = len(universe)
            _progress.current_symbol = "开始重算评分"
            if on_progress:
                on_progress(_progress)

            with ThreadPoolExecutor(max_workers=config.SCREENER_MAX_WORKERS) as ex:
                futures = {ex.submit(_score_one, row): row for row in universe}
                for fut in as_completed(futures):
                    row = futures[fut]
                    sym = str(row.get("symbol") or "").zfill(6)
                    _progress.current_symbol = sym
                    try:
                        status, score_row = fut.result()
                    except Exception as e:  # noqa: BLE001
                        log.warning("debug recompute worker error %s: %s", sym, e)
                        status = "failed"
                        score_row = None
                    if score_row is not None:
                        results.append(score_row)
                    if status == "accepted":
                        _progress.accepted += 1
                    elif status == "rejected":
                        _progress.rejected += 1
                    else:
                        _progress.failed += 1
                    _progress.done += 1
                    if on_progress:
                        on_progress(_progress)

        results.sort(key=lambda r: r.score, reverse=True)
        result_rows = [r.to_dict() for r in results]
        meta = {
            **old_meta,
            "scoring_rule_version": config.SCORING_RULE_VERSION,
            "source_total": _progress.source_total,
            "prefiltered_total": _progress.prefiltered_total,
            "total": _progress.total,
            "done": _progress.done,
            "accepted": _progress.accepted,
            "rejected": _progress.rejected,
            "failed": _progress.failed,
            "last_recompute_ts": time.time(),
            "last_recompute_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "last_recompute_mode": "cache_only" if cache_only else "allow_fetch",
        }
        cache.put_json("screener", result_rows, FULL_SCAN_RESULTS_KEY)
        cache.put_json("screener", result_rows, LAST_RUN_KEY)
        cache.put_json("screener", meta, FULL_SCAN_META_KEY)
        _progress.state = "done"
        _progress.results = result_rows
    except Exception as e:  # noqa: BLE001
        _progress.state = "error"
        _progress.error = str(e)
        log.exception("debug full-scan recompute failed")
    return results


def last_results() -> list[dict] | None:
    if invalidate_stale_score_caches():
        return None
    if _progress.results:
        rows, changed = _repair_legacy_dividend_rows(_progress.results)
        if changed:
            _progress.results = rows
            cache.put_json("screener", rows, LAST_RUN_KEY)
            if _cached_full_scan_rows():
                meta = {**(full_scan_meta() or {}), "scoring_rule_version": config.SCORING_RULE_VERSION}
                cache.put_json("screener", rows, FULL_SCAN_RESULTS_KEY)
                cache.put_json("screener", meta, FULL_SCAN_META_KEY)
        return rows
    cached = cache.get_json("screener", LAST_RUN_KEY)
    if cached:
        rows, changed = _repair_legacy_dividend_rows(cached)
        _progress.results = rows
        if changed:
            cache.put_json("screener", rows, LAST_RUN_KEY)
            if _cached_full_scan_rows():
                meta = {**(full_scan_meta() or {}), "scoring_rule_version": config.SCORING_RULE_VERSION}
                cache.put_json("screener", rows, FULL_SCAN_RESULTS_KEY)
                cache.put_json("screener", meta, FULL_SCAN_META_KEY)
        return rows
    return None

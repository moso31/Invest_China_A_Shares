"""Whole-market AKShare bulk fetches and lookup helpers."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
from collections.abc import Iterable

import akshare as ak
import pandas as pd

from loha import cache, config, source

log = logging.getLogger("loha.bulk")

_YJBB_ALIAS_COLUMNS = {
    "OPERATE_INCOME": (
        "OPERATE_INCOME",
        "营业总收入-营业总收入",
        "营业总收入",
        "营业收入",
    ),
    "OPERATE_INCOME_YOY": (
        "OPERATE_INCOME_YOY",
        "营业总收入-同比增长",
        "营业收入-同比增长",
        "营业总收入同比增长",
    ),
    "PARENT_NETPROFIT": (
        "PARENT_NETPROFIT",
        "归母净利润",
        "归属于母公司股东的净利润",
        "净利润-净利润",
        "净利润",
    ),
    "PARENT_NETPROFIT_YOY": (
        "PARENT_NETPROFIT_YOY",
        "归母净利润同比增长",
        "净利润-同比增长",
        "净利润同比增长",
    ),
    "BASIC_EPS": ("BASIC_EPS", "每股收益", "基本每股收益"),
    "ROE_AVG": ("ROE_AVG", "WEIGHTAVG_ROE", "净资产收益率", "加权净资产收益率"),
    "GROSS_PROFIT_MARGIN": ("GROSS_PROFIT_MARGIN", "销售毛利率", "毛利率"),
}

_FHPS_ALIAS_COLUMNS = {
    "PRETAX_BONUS_RMB": (
        "PRETAX_BONUS_RMB",
        "现金分红-现金分红比例",
        "现金分红比例",
        "派息比例",
        "派息",
    ),
}

_NUMERIC_ALIASES = set(_YJBB_ALIAS_COLUMNS) | set(_FHPS_ALIAS_COLUMNS)


def _normalise_periods(periods: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in periods:
        period = re.sub(r"\D", "", str(item or ""))
        if len(period) != 8:
            continue
        if period not in seen:
            out.append(period)
            seen.add(period)
    out.sort(reverse=True)
    return out


def _period_cache_key(periods: list[str]) -> str:
    joined = "_".join(periods)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    first = periods[0] if periods else "none"
    last = periods[-1] if periods else "none"
    return f"{len(periods)}_{first}_{last}_{digest}"


def _report_deadline(year: int, month_day: str) -> dt.date:
    if month_day == "1231":
        return dt.date(year + 1, 4, 30)
    month, day = int(month_day[:2]), int(month_day[2:])
    if month_day == "0630":
        return dt.date(year, 8, 31)
    if month_day == "0930":
        return dt.date(year, 10, 31)
    if month_day == "0331":
        return dt.date(year, 4, 30)
    return dt.date(year, month, day)


def _recent_periods(n_years: int = 5, n_quarters: int = 4) -> list[str]:
    """Return disclosed report periods covering the recent five-year window."""
    today = dt.date.today()
    annual_years = [
        year for year in range(today.year, today.year - n_years - 4, -1) if _report_deadline(year, "1231") <= today
    ][:n_years]
    if not annual_years:
        return []

    latest_annual_year = annual_years[0]
    periods: list[str] = []

    recent_quarters = 0
    for year in range(today.year, latest_annual_year, -1):
        for month_day in ("0930", "0630", "0331"):
            if recent_quarters >= n_quarters:
                break
            if _report_deadline(year, month_day) <= today:
                periods.append(f"{year}{month_day}")
                recent_quarters += 1

    for year in annual_years:
        for month_day in ("1231", "0930", "0630", "0331"):
            if _report_deadline(year, month_day) <= today:
                periods.append(f"{year}{month_day}")

    return _normalise_periods(periods)


def _find_code_column(df: pd.DataFrame) -> str | None:
    for candidate in ("SECURITY_CODE", "股票代码", "代码"):
        if candidate in df.columns:
            return candidate
    best_col: str | None = None
    best_ratio = 0.0
    for col in df.columns:
        ser = df[col].dropna().astype(str).str.strip()
        if ser.empty:
            continue
        ratio = ser.str.fullmatch(r"\d{6}").mean()
        if ratio > best_ratio:
            best_col = str(col)
            best_ratio = float(ratio)
    return best_col if best_ratio >= 0.5 else None


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    columns = [str(c) for c in df.columns]
    for candidate in candidates:
        if candidate in df.columns:
            return str(candidate)
    for candidate in candidates:
        for col in columns:
            if candidate and candidate in col:
                return col
    return None


def _normalise_report_date(value: object) -> str | None:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) >= 8:
        return digits[:8]
    if len(digits) >= 4:
        return f"{digits[:4]}1231"
    return None


def _with_common_aliases(df: pd.DataFrame, period: str, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    out = source._fix_mojibake_df(df)  # type: ignore[attr-defined]
    out = out.copy()
    code_col = _find_code_column(out)
    if code_col is not None:
        out["SECURITY_CODE"] = out[code_col].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    else:
        out["SECURITY_CODE"] = ""
    out["REPORT_DATE"] = period

    for alias, candidates in aliases.items():
        if alias not in out.columns:
            source_col = _find_column(out, candidates)
            if source_col is not None:
                out[alias] = out[source_col]
    for col in _NUMERIC_ALIASES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _sort_bulk_slice(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["REPORT_DATE"] = out["REPORT_DATE"].map(_normalise_report_date)
    out = out.dropna(subset=["REPORT_DATE"]).sort_values("REPORT_DATE", ascending=False)
    return out.reset_index(drop=True)


def _default_cached(kind: str) -> pd.DataFrame | None:
    periods = _recent_periods(n_years=5)
    return cache.get("bulk", kind, _period_cache_key(periods))


def bulk_yjbb(periods: list[str] | None = None) -> pd.DataFrame | None:
    """Fetch recent whole-market performance reports and cache one long table."""
    periods = _normalise_periods(periods or _recent_periods(n_years=5))
    if not periods:
        log.warning("bulk_yjbb: no valid periods")
        return None

    key = _period_cache_key(periods)
    cached = cache.get("bulk", "yjbb", key)
    if cached is not None:
        log.info("bulk_yjbb: cache hit %s rows for %s", len(cached), key)
        return cached

    frames: list[pd.DataFrame] = []
    for period in periods:
        df = source._retry(ak.stock_yjbb_em, date=period)  # type: ignore[attr-defined]
        if df is None:
            log.warning("bulk_yjbb: %s fetch failed, skipped", period)
            continue
        if df.empty:
            log.info("bulk_yjbb: %s fetched 0 rows", period)
            continue
        normalised = _with_common_aliases(df, period, _YJBB_ALIAS_COLUMNS)
        frames.append(normalised)
        log.info("bulk_yjbb: %s fetched %s rows", period, len(normalised))

    if not frames:
        log.warning("bulk_yjbb: all periods failed or empty: %s", ",".join(periods))
        return None

    out = pd.concat(frames, ignore_index=True)
    out = _sort_bulk_slice(out)
    cache.put(
        "bulk",
        out,
        "yjbb",
        key,
        ttl_hours=config.CACHE_TTL_HOURS.get("financial", 24 * 7),
    )
    return out


def bulk_fhps(periods: list[str] | None = None) -> pd.DataFrame | None:
    """Fetch recent whole-market dividend/bonus reports and cache one long table."""
    periods = _normalise_periods(periods or _recent_periods(n_years=5))
    if not periods:
        log.warning("bulk_fhps: no valid periods")
        return None

    key = _period_cache_key(periods)
    cached = cache.get("bulk", "fhps", key)
    if cached is not None:
        log.info("bulk_fhps: cache hit %s rows for %s", len(cached), key)
        return cached

    frames: list[pd.DataFrame] = []
    for period in periods:
        df = source._retry(ak.stock_fhps_em, date=period)  # type: ignore[attr-defined]
        if df is None:
            log.warning("bulk_fhps: %s fetch failed, skipped", period)
            continue
        if df.empty:
            log.info("bulk_fhps: %s fetched 0 rows", period)
            continue
        normalised = _with_common_aliases(df, period, _FHPS_ALIAS_COLUMNS)
        frames.append(normalised)
        log.info("bulk_fhps: %s fetched %s rows", period, len(normalised))

    if not frames:
        log.warning("bulk_fhps: all periods failed or empty: %s", ",".join(periods))
        return None

    out = pd.concat(frames, ignore_index=True)
    out = _sort_bulk_slice(out)
    cache.put(
        "bulk",
        out,
        "fhps",
        key,
        ttl_hours=config.CACHE_TTL_HOURS.get("financial", 24 * 7),
    )
    return out


def get_bulk_yjbb_for(symbol: str) -> pd.DataFrame | None:
    """Return cached performance-report rows for one stock, newest period first."""
    df = _default_cached("yjbb")
    if df is None or "SECURITY_CODE" not in df.columns:
        return None
    code = str(symbol).zfill(6)
    mask = df["SECURITY_CODE"].astype(str).str.zfill(6) == code
    return _sort_bulk_slice(df.loc[mask].copy())


def get_bulk_fhps_for(symbol: str) -> pd.DataFrame | None:
    """Return cached dividend-report rows for one stock, newest period first."""
    df = _default_cached("fhps")
    if df is None or "SECURITY_CODE" not in df.columns:
        return None
    code = str(symbol).zfill(6)
    mask = df["SECURITY_CODE"].astype(str).str.zfill(6) == code
    return _sort_bulk_slice(df.loc[mask].copy())


def _period_value_map(df: pd.DataFrame, col: str, annual_only: bool) -> dict[int | str, float]:
    if df.empty or col not in df.columns or "REPORT_DATE" not in df.columns:
        return {}
    out: dict[int | str, float] = {}
    rows = df[["REPORT_DATE", col]].copy()
    rows["REPORT_DATE"] = rows["REPORT_DATE"].map(_normalise_report_date)
    rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows = rows.dropna(subset=["REPORT_DATE", col]).sort_values("REPORT_DATE", ascending=False)
    for _, row in rows.iterrows():
        period = str(row["REPORT_DATE"])
        if annual_only and period[4:8] != "1231":
            continue
        key: int | str = int(period[:4]) if annual_only else period
        if key not in out:
            out[key] = float(row[col])
    return out


def quick_yjbb_metrics(symbol: str) -> dict | None:
    """Extract the subset of financial metrics covered by the YJBB bulk table."""
    df = get_bulk_yjbb_for(symbol)
    if df is None or df.empty:
        return None
    metrics = {
        "revenue_by_year": _period_value_map(df, "OPERATE_INCOME", annual_only=True),
        "revenue_yoy_by_year": _period_value_map(df, "OPERATE_INCOME_YOY", annual_only=True),
        "revenue_by_period": _period_value_map(df, "OPERATE_INCOME", annual_only=False),
        "ni_by_year": _period_value_map(df, "PARENT_NETPROFIT", annual_only=True),
        "ni_yoy_by_year": _period_value_map(df, "PARENT_NETPROFIT_YOY", annual_only=True),
        "ni_by_period": _period_value_map(df, "PARENT_NETPROFIT", annual_only=False),
        "roe_by_year": _period_value_map(df, "ROE_AVG", annual_only=True),
        "roe_by_period": _period_value_map(df, "ROE_AVG", annual_only=False),
        "eps_by_year": _period_value_map(df, "BASIC_EPS", annual_only=True),
        "eps_by_period": _period_value_map(df, "BASIC_EPS", annual_only=False),
    }
    return metrics if any(metrics.values()) else None

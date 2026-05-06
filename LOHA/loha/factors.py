"""Factor calculators - Q / D / V / T / R.

Each factor returns a `FactorResult` with a 0-100 score, a dict of
explanatory metrics (shown in the UI), and a list of flag strings.
The composite score in `scoring.py` uses estimate.md's strategy fit:
   Score = 0.45D + 0.35V + 0.20T

Robustness over precision: AKShare endpoints return inconsistent column
layouts. We try several column-name variants for each indicator and fall
back to neutral scores when data is missing.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from loha import bulk, config, source
from loha import valuation as valuation_engine

log = logging.getLogger("loha.factors")


@dataclass
class FactorResult:
    score: float  # 0-100, higher = better (R is inverted in scoring)
    components: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    good_flags: list[str] = field(default_factory=list)


# ---- helpers -------------------------------------------------------------


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if x is None or math.isnan(x):
        return 50.0
    return float(max(lo, min(hi, x)))


def _to_num(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if not (isinstance(v, float) and math.isnan(v)) else None
    s = str(v).strip().replace(",", "")
    if not s or s in {"--", "-", "nan", "None"}:
        return None
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    mult = 1.0
    if s.endswith("亿"):
        mult, s = 1e8, s[:-1]
    elif s.endswith("万"):
        mult, s = 1e4, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _find_indicator_row(df: pd.DataFrame, *candidates: str) -> pd.Series | None:
    """Find a row whose label column matches any candidate substring.

    The financial_abstract frame has TWO label columns: 选项 (high-level
    category) and 指标 (specific indicator name). We must search the
    指标 column - searching 选项 only finds category headers like '常用指标'.
    """
    if df is None or df.empty:
        return None
    label_col = None
    for preferred in ("指标", "项目"):
        for c in df.columns:
            if str(c) == preferred or preferred in str(c):
                label_col = c
                break
        if label_col is not None:
            break
    if label_col is None:
        for c in df.columns:
            if any(k in str(c) for k in ("指标", "选项", "项目")):
                label_col = c
                break
    if label_col is None:
        return None
    labels = df[label_col].astype(str)
    for cand in candidates:
        mask = labels.str.contains(cand, regex=False, na=False)
        if mask.any():
            return df[mask].iloc[0]
    return None


def _row_period_values(row: pd.Series | None) -> list[tuple[str, float]]:
    """Return [(period, value)] from a financial-abstract row, newest first."""
    if row is None:
        return []
    out: list[tuple[str, float]] = []
    for col, val in row.items():
        col_s = str(col)
        if re.fullmatch(r"\d{8}", col_s) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", col_s):
            num = _to_num(val)
            if num is not None:
                out.append((col_s, num))
    out.sort(key=lambda x: x[0], reverse=True)
    return out


def _annual_only(periods: list[tuple[str, float]]) -> list[tuple[str, float]]:
    return [p for p in periods if p[0][4:8] == "1231" or p[0].endswith("-12-31")]


def _latest_annual_yoy(row: pd.Series | None) -> float | None:
    if row is None:
        return None
    annual = _annual_only(_row_period_values(row))[:2]
    if len(annual) < 2 or annual[1][1] == 0:
        return None
    return (annual[0][1] - annual[1][1]) / abs(annual[1][1]) * 100


def _is_financial_company(symbol: str) -> tuple[bool, str]:
    info = source.individual_info(symbol) or {}
    industry = str(info.get("行业", "") or "")
    is_financial = any(k in industry for k in ("银行", "保险", "证券", "多元金融", "金融"))
    return is_financial, industry


def _quick_metrics(symbol: str) -> dict | None:
    return bulk.quick_yjbb_metrics(symbol)


def _quick_year_values(metrics: dict | None, key: str, limit: int | None = None) -> list[tuple[str, float]]:
    if not metrics:
        return []
    values = metrics.get(key) or {}
    out = [(f"{int(year)}1231", float(value)) for year, value in values.items()]
    out.sort(key=lambda item: item[0], reverse=True)
    return out[:limit] if limit else out


def _quick_period_values(metrics: dict | None, key: str, limit: int | None = None) -> list[tuple[str, float]]:
    if not metrics:
        return []
    values = metrics.get(key) or {}
    out = [(str(period), float(value)) for period, value in values.items()]
    out.sort(key=lambda item: item[0], reverse=True)
    return out[:limit] if limit else out


def _quick_latest_annual_yoy(metrics: dict | None, yoy_key: str, value_key: str) -> float | None:
    if not metrics:
        return None
    yoy_values = metrics.get(yoy_key) or {}
    if yoy_values:
        latest_year = max(int(year) for year in yoy_values)
        value = yoy_values.get(latest_year)
        return float(value) if value is not None else None
    annual = _quick_year_values(metrics, value_key, limit=2)
    if len(annual) < 2 or annual[1][1] == 0:
        return None
    return (annual[0][1] - annual[1][1]) / abs(annual[1][1]) * 100


def _load_financial_if_needed(symbol: str, fin: pd.DataFrame | None) -> pd.DataFrame | None:
    return fin if fin is not None else source.financial_abstract(symbol)


def _dividend_by_year_from_bulk_fhps(df: pd.DataFrame | None) -> dict[int, float]:
    if df is None or df.empty or "REPORT_DATE" not in df.columns:
        return {}
    amount_col = "PRETAX_BONUS_RMB" if "PRETAX_BONUS_RMB" in df.columns else None
    if amount_col is None:
        amount_col = next((c for c in ("派息比例", "派息(元/10股)", "派息") if c in df.columns), None)
    if amount_col is None:
        return {}
    work = df[["REPORT_DATE", amount_col]].copy()
    work["_year"] = work["REPORT_DATE"].astype(str).str.extract(r"(\d{4})", expand=False)
    work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce")
    work = work.dropna(subset=["_year", amount_col])
    if work.empty:
        return {}
    yearly = (work.groupby("_year")[amount_col].sum() / 10.0).to_dict()
    return {int(k): float(v) for k, v in yearly.items()}


def _dividend_hist_columns(div_hist: pd.DataFrame) -> tuple[str | None, str | None]:
    amount_col = next((c for c in ("派息比例", "派息(元/10股)", "派息") if c in div_hist.columns), None)
    date_col = next(
        (
            c
            for c in (
                "实施分红年度",
                "分红年度",
                "报告期",
                "报告时间",
                "实施方案公告日期",
                "除权除息日期",
                "公告日期",
                "实施日期",
            )
            if c in div_hist.columns
        ),
        None,
    )
    return amount_col, date_col


def _dividend_by_year_from_hist(div_hist: pd.DataFrame | None) -> dict[int, float]:
    if div_hist is None or div_hist.empty:
        return {}
    amount_col, date_col = _dividend_hist_columns(div_hist)
    if amount_col is None or date_col is None:
        return {}
    df = div_hist.copy()
    year_from_text = df[date_col].astype(str).str.extract(r"(\d{4})", expand=False)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df["_year"] = pd.to_numeric(year_from_text, errors="coerce")
    missing_year = df["_year"].isna() & df[date_col].notna()
    df.loc[missing_year, "_year"] = df.loc[missing_year, date_col].dt.year
    df = df.dropna(subset=["_year", amount_col])
    if df.empty:
        return {}
    yearly = (df.groupby("_year")[amount_col].sum() / 10.0).to_dict()
    return {int(k): float(v) for k, v in yearly.items()}


def _dividend_ttm_from_rows(df: pd.DataFrame | None, amount_col: str | None, date_col: str | None) -> float:
    if df is None or df.empty or amount_col is None or date_col is None:
        return 0.0
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work[amount_col] = pd.to_numeric(work[amount_col], errors="coerce")
    work = work.dropna(subset=[date_col, amount_col])
    if work.empty:
        return 0.0
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=1)
    ttm = work[work[date_col] >= cutoff]
    return float(ttm[amount_col].sum()) / 10.0 if not ttm.empty else 0.0


# ===========================================================================
# Q - Enterprise Quality
# ===========================================================================


def quality(symbol: str) -> FactorResult:
    quick = _quick_metrics(symbol)
    fin = None if quick else source.financial_abstract(symbol)
    components: dict = {}
    flags: list[str] = []
    good_flags: list[str] = []

    if quick is None and (fin is None or fin.empty):
        return FactorResult(50.0, {"note": "no financial data"}, ["缺少财务数据"])

    fin = _load_financial_if_needed(symbol, fin)
    roe_row = _find_indicator_row(fin, "净资产收益率", "ROE")
    margin_row = _find_indicator_row(fin, "销售净利率", "净利率")
    gross_row = _find_indicator_row(fin, "毛利率")
    debt_row = _find_indicator_row(fin, "资产负债率")
    ocf_row = _find_indicator_row(fin, "经营现金流量净额", "经营现金流", "经营活动产生的现金流量净额")
    ni_row = _find_indicator_row(fin, "归母净利润", "归属于母公司", "净利润")
    rev_row = _find_indicator_row(fin, "营业总收入", "营业收入")
    is_financial, industry = _is_financial_company(symbol)
    if industry:
        components["industry"] = industry

    # ROE 3-year median
    roe_score = 50.0
    roe_annual = _quick_year_values(quick, "roe_by_year", limit=3)
    if not roe_annual and roe_row is not None:
        roe_annual = _annual_only(_row_period_values(roe_row))[:3]
    if roe_annual:
        vals = [v for _, v in roe_annual]
        med = float(np.median(vals))
        components["ROE_median_3y"] = round(med, 2)
        # Map: 0% -> 0 score, 10% -> 60, 15% -> 80, 20%+ -> 100
        roe_score = _clip(20 + med * 4)
        if med < config.ROE_MEDIAN_3Y_MIN:
            flags.append(f"ROE 三年中位数 {med:.1f}% 低于 {config.ROE_MEDIAN_3Y_MIN}% 门槛")
        elif med >= 15:
            good_flags.append(f"ROE 三年中位数 {med:.1f}% 较高")

    # Net margin (latest)
    margin_score = 50.0
    if margin_row is not None:
        periods = _row_period_values(margin_row)
        if periods:
            latest = periods[0][1]
            components["net_margin"] = round(latest, 2)
            margin_score = _clip(40 + latest * 2)

    # Gross margin (latest)
    gross_score = 50.0
    if gross_row is not None:
        periods = _row_period_values(gross_row)
        if periods:
            latest = periods[0][1]
            components["gross_margin"] = round(latest, 2)
            gross_score = _clip(30 + latest * 1.5)

    # OCF / NI - 3 year mean (proxy for cash quality)
    cash_score = 50.0
    quick_ni_periods = dict(_quick_year_values(quick, "ni_by_year", limit=3))
    if ocf_row is not None and (ni_row is not None or quick_ni_periods):
        ocf_periods = dict(_annual_only(_row_period_values(ocf_row))[:3])
        ni_periods = quick_ni_periods or dict(_annual_only(_row_period_values(ni_row))[:3])
        ratios = []
        for k, ocf in ocf_periods.items():
            ni = ni_periods.get(k)
            if ni and ni > 0:
                ratios.append(ocf / ni)
        if ratios:
            mean = float(np.mean(ratios))
            components["OCF_over_NI_3y"] = round(mean, 2)
            cash_score = _clip(30 + mean * 50)
            if mean < config.OCF_OVER_NI_3Y_MIN:
                flags.append(f"经营现金流 / 净利润 三年均值 {mean:.2f} 低于 {config.OCF_OVER_NI_3Y_MIN}")
            elif mean >= 1.0:
                good_flags.append(f"经营现金流 / 净利润 三年均值 {mean:.2f}，现金流覆盖利润良好")

    # Growth (latest annual YoY), used as a soft quality component.
    growth_score = 50.0
    growth_values = []
    rev_yoy = _quick_latest_annual_yoy(quick, "revenue_yoy_by_year", "revenue_by_year")
    ni_yoy = _quick_latest_annual_yoy(quick, "ni_yoy_by_year", "ni_by_year")
    if rev_yoy is None:
        rev_yoy = _latest_annual_yoy(rev_row)
    if ni_yoy is None:
        ni_yoy = _latest_annual_yoy(ni_row)
    if rev_yoy is not None:
        components["revenue_yoy"] = round(rev_yoy, 2)
        growth_values.append(rev_yoy)
        if rev_yoy < -5:
            flags.append(f"营业收入最近年度同比 {rev_yoy:.1f}% 下滑")
    if ni_yoy is not None:
        components["net_profit_yoy"] = round(ni_yoy, 2)
        growth_values.append(ni_yoy)
        if ni_yoy < -10:
            flags.append(f"净利润最近年度同比 {ni_yoy:.1f}% 下滑")
    if growth_values:
        avg_growth = float(np.mean(growth_values))
        # -20% -> 10, 0% -> 50, 15% -> 80, 25%+ -> 100
        growth_score = _clip(50 + avg_growth * 2)

    # Debt ratio (latest). Banks and insurers naturally have high liability
    # ratios, so the normal industrial-company rule would create false alarms.
    debt_score = 50.0
    if debt_row is not None:
        periods = _row_period_values(debt_row)
        if periods:
            latest = periods[0][1]
            components["debt_ratio"] = round(latest, 2)
            if is_financial:
                debt_score = 65.0
                components["debt_note"] = "金融行业负债率不按普通企业阈值扣分"
            else:
                debt_score = _clip(120 - latest)
            if latest > config.DEBT_RATIO_MAX_HARD and not is_financial:
                flags.append(f"资产负债率 {latest:.0f}% 偏高")
            elif latest < 50 and not is_financial:
                good_flags.append(f"资产负债率 {latest:.0f}% 较稳健")

    # Composite Q
    q = (
        0.28 * roe_score
        + 0.25 * cash_score
        + 0.17 * margin_score
        + 0.10 * gross_score
        + 0.10 * debt_score
        + 0.10 * growth_score
    )
    return FactorResult(round(q, 1), components, flags, good_flags)


# ===========================================================================
# D - Dividend Quality
# ===========================================================================


def weighted_dividend_metrics(
    div_by_year: dict[int, float],
    last_close: float | None,
    listing_year: int | None,
) -> dict:
    """Compute the current D-factor 5-year weighted dividend metrics.

    `div_by_year` is 元/股 by calendar year. Missing listed years and
    pre-listing years both score as 0% yield; pre-listing years are tagged
    separately for display.
    """
    div_year_yields: dict[str, float] = {}
    div_year_sources: dict[str, str] = {}
    weighted_score = 0.0
    yield_pct: float | None = None

    if last_close and last_close > 0:
        current_year = pd.Timestamp.today().year
        target_years = [current_year - 1 - i for i in range(config.DIV_LOOKBACK_YEARS)]
        raw_year_yields: dict[int, float] = {}
        weighted_yield = 0.0
        for year in target_years:
            if listing_year is not None and year < listing_year:
                yr_yield = 0.0
                div_year_sources[str(year)] = "pre_listing_zero"
            elif year in div_by_year:
                yr_yield = div_by_year[year] / last_close * 100
                div_year_sources[str(year)] = "actual_dividend"
            else:
                yr_yield = 0.0
                div_year_sources[str(year)] = "listed_no_dividend"
            raw_year_yields[year] = yr_yield
            div_year_yields[str(year)] = round(yr_yield, 2)
        for year, w in zip(target_years, config.DIV_YEAR_WEIGHTS):
            weighted_yield += w * raw_year_yields.get(year, 0.0)
        weighted_score = _clip(weighted_yield / config.DIV_YIELD_FULL_CREDIT * 60, hi=60.0)
        yield_pct = weighted_yield

    return {
        "score": round(weighted_score, 1),
        "dividend_yield": round(yield_pct, 2) if yield_pct is not None else None,
        "div_year_yields": div_year_yields,
        "div_year_sources": div_year_sources,
    }


def dividend(symbol: str) -> FactorResult:
    components: dict = {}
    flags: list[str] = []
    good_flags: list[str] = []

    bulk_fhps = bulk.get_bulk_fhps_for(symbol)
    ind = source.indicator_hist(symbol)

    # Latest close
    last_close: float | None = None
    if ind is not None and "close" in ind.columns and not ind.empty:
        try:
            last_close = float(ind["close"].dropna().iloc[-1])
        except Exception:
            last_close = None

    # Determine listing year from individual_info (上市时间 field)
    listing_year: int | None = None
    info = source.individual_info(symbol) or {}
    listing_date_str = str(info.get("上市时间", "") or "")
    if listing_date_str:
        try:
            ts = pd.to_datetime(listing_date_str, errors="coerce")
            if pd.notna(ts):
                listing_year = int(ts.year)
        except Exception:
            pass

    # Build per-year dividend map (year -> 元/股) and TTM total
    div_by_year: dict[int, float] = {}
    div_per_share_ttm = 0.0

    if bulk_fhps is not None and (bulk_fhps.empty or "PRETAX_BONUS_RMB" in bulk_fhps.columns):
        div_by_year = _dividend_by_year_from_bulk_fhps(bulk_fhps)
        div_per_share_ttm = _dividend_ttm_from_rows(bulk_fhps, "PRETAX_BONUS_RMB", "REPORT_DATE")
    else:
        if bulk_fhps is not None:
            log.warning("bulk dividend schema missing PRETAX_BONUS_RMB for %s; fallback to dividend_hist", symbol)
        div_hist = source.dividend_hist(symbol)
        amount_col, date_col = _dividend_hist_columns(div_hist) if div_hist is not None else (None, None)
        div_by_year = _dividend_by_year_from_hist(div_hist)
        div_per_share_ttm = _dividend_ttm_from_rows(div_hist, amount_col, date_col)

    # Fall back listing_year to earliest dividend year if info unavailable
    if listing_year is None and div_by_year:
        listing_year = min(div_by_year.keys())

    components["dividend_per_share_ttm"] = round(div_per_share_ttm, 4)

    metrics = weighted_dividend_metrics(div_by_year, last_close, listing_year)
    current_year = pd.Timestamp.today().year
    target_years = [current_year - 1 - i for i in range(config.DIV_LOOKBACK_YEARS)]
    dps_new_to_old = [float(div_by_year.get(year, 0.0)) for year in target_years]
    dps_old_to_new = list(reversed(dps_new_to_old))
    latest_dps = next((v for v in dps_new_to_old if v > 0), 0.0)
    current_yield = latest_dps / last_close * 100 if last_close and last_close > 0 else None

    components["dividend_yield"] = round(current_yield, 2) if current_yield is not None else None
    components["weighted_dividend_yield"] = metrics["dividend_yield"]
    components["weighted_dividend_score"] = metrics["score"]
    components["div_year_yields"] = metrics["div_year_yields"]
    components["div_year_sources"] = metrics["div_year_sources"]
    components["div_by_year"] = {str(k): round(v, 4) for k, v in sorted(div_by_year.items(), reverse=True)}

    quick = _quick_metrics(symbol)
    fin: pd.DataFrame | None = None
    eps_by_year = {int(year): float(eps) for year, eps in (quick or {}).get("eps_by_year", {}).items()}
    if not eps_by_year:
        fin = _load_financial_if_needed(symbol, fin)
        eps_row = _find_indicator_row(fin, "基本每股收益")
        for period, eps in _annual_only(_row_period_values(eps_row)):
            try:
                eps_by_year[int(period[:4])] = eps
            except ValueError:
                continue
    payout_ratios: list[float] = []
    for year in target_years[:3]:
        dps = float(div_by_year.get(year, 0.0))
        eps = eps_by_year.get(year)
        if dps > 0 and eps and eps > 0:
            payout_ratios.append(dps / eps * 100)
    payout_ratio_3y_avg = float(np.mean(payout_ratios)) if payout_ratios else None
    components["payout_ratio_3y_avg"] = round(payout_ratio_3y_avg, 2) if payout_ratio_3y_avg is not None else None

    payout_score = 0.0
    if payout_ratio_3y_avg is None:
        payout_score = 5.0
    elif payout_ratio_3y_avg <= 80:
        payout_score = 10.0

    fin = _load_financial_if_needed(symbol, fin)
    ocf_row = _find_indicator_row(fin, "经营现金流量净额", "经营现金流", "经营活动产生的现金流量净额")
    ni_row = _find_indicator_row(fin, "归母净利润", "归属于母公司", "净利润")
    ocf_latest = (_row_period_values(ocf_row) or [(None, None)])[0][1]
    ni_latest = (_quick_period_values(quick, "ni_by_period", limit=1) or _row_period_values(ni_row) or [(None, None)])[
        0
    ][1]
    cash_coverage_score = 0.0
    cash_coverage: float | None = None
    if ocf_latest is not None and ni_latest is not None and ni_latest > 0 and payout_ratio_3y_avg is not None:
        required_cash = ni_latest * (payout_ratio_3y_avg / 100)
        if required_cash > 0:
            cash_coverage = ocf_latest / required_cash
            if cash_coverage >= 1.2:
                cash_coverage_score = 15
            elif cash_coverage >= 1.0:
                cash_coverage_score = 8
    components["dividend_cash_coverage"] = round(cash_coverage, 2) if cash_coverage is not None else None

    roe_vals = [v for _, v in _quick_year_values(quick, "roe_by_year", limit=5)]
    if not roe_vals:
        roe_row = _find_indicator_row(fin, "净资产收益率", "ROE")
        roe_vals = [v for _, v in _annual_only(_row_period_values(roe_row))[:5]]
    roe_5y_std = float(np.std(roe_vals, ddof=0)) if len(roe_vals) >= 2 else None
    components["roe_5y_std"] = round(roe_5y_std, 2) if roe_5y_std is not None else None
    roe_stability_score = 0.0
    if roe_5y_std is not None:
        if roe_5y_std < 3:
            roe_stability_score = 15
        elif roe_5y_std < 6:
            roe_stability_score = 8

    components.update(
        {
            "latest_dps": round(latest_dps, 4),
            "payout_score": round(payout_score, 1),
            "cash_coverage_score": round(cash_coverage_score, 1),
            "roe_stability_score": round(roe_stability_score, 1),
            "dps_history": [round(v, 4) for v in dps_old_to_new],
        }
    )

    weighted_yield = metrics["dividend_yield"]
    if weighted_yield is not None and weighted_yield >= config.DIV_YIELD_FULL_CREDIT:
        good_flags.append(f"5年加权股息率 {weighted_yield:.1f}% 达到策略核心区间")
    if payout_score >= 10 and cash_coverage_score >= 8:
        good_flags.append("分红率与经营现金流覆盖较稳定")

    listed_years = [year for year in target_years if listing_year is None or year >= listing_year]
    if any(float(div_by_year.get(year, 0.0)) <= 0 for year in listed_years):
        flags.append("dividend_unstable：近 5 年存在上市后未分红年份")
    positive_path = [v for v in dps_old_to_new if v > 0]
    if len(positive_path) >= 2:
        for prev, cur in zip(positive_path, positive_path[1:]):
            if prev > 0 and cur / prev < 0.5:
                flags.append("dividend_unstable：近 5 年每股分红出现大幅下调")
                break

    d = float(metrics["score"]) + payout_score + cash_coverage_score + roe_stability_score
    return FactorResult(round(d, 1), components, flags, good_flags)


# ===========================================================================
# V - Valuation safety margin
# ===========================================================================


def valuation(symbol: str) -> FactorResult:
    try:
        est = valuation_engine.evaluate(symbol)
    except Exception as exc:  # noqa: BLE001
        return FactorResult(50.0, {"note": f"valuation failed: {exc}"}, [f"估值计算失败：{exc}"])
    return FactorResult(
        est.value_score,
        est.components,
        est.flags,
        est.good_flags,
    )


# ===========================================================================
# T - Tradability (turnover, ATR, amplitude)
# ===========================================================================


def trading(symbol: str) -> FactorResult:
    components: dict = {}
    flags: list[str] = []
    good_flags: list[str] = []
    px = source.price_hist(symbol, years=1)
    if px is None or px.empty or "close" not in px.columns:
        return FactorResult(50.0, {"note": "no price data"}, ["缺少行情数据"])

    df = px.tail(max(80, config.TURNOVER_WINDOW, config.ATR_WINDOW + 5)).copy()

    # Mean daily turnover (CNY)
    if "amount" in df.columns:
        avg_amount = float(df["amount"].tail(20).mean())
        components["avg_turnover_cny"] = round(avg_amount, 0)
    else:
        avg_amount = 0.0

    # ATR(20) %
    high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(config.ATR_WINDOW).mean().iloc[-1]
    last_close = float(close.iloc[-1])
    atr_pct = float(atr / last_close * 100) if atr and last_close else 0.0
    components["ATR20"] = round(float(atr) if not pd.isna(atr) else 0.0, 3)
    components["ATR20_pct"] = round(atr_pct, 2)
    components["last_close"] = round(last_close, 2)

    # Mean amplitude (high-low)/prev_close
    amp = ((high - low) / prev_close * 100).tail(config.TURNOVER_WINDOW).mean()
    components["avg_amplitude_pct"] = round(float(amp) if not pd.isna(amp) else 0.0, 2)

    # estimate.md §6.3 t_score.
    returns = close.pct_change().dropna()
    vol_60d_annual = None
    if len(returns) >= 60:
        vol_60d_annual = float(returns.tail(60).std() * math.sqrt(252))
    components["vol_60d_annual_pct"] = round(vol_60d_annual * 100, 2) if vol_60d_annual is not None else None

    vol_score = 0.0
    if vol_60d_annual is not None:
        if 0.20 <= vol_60d_annual <= 0.40:
            vol_score = 40
            good_flags.append(f"60 日年化波动 {vol_60d_annual * 100:.1f}%，适合小幅做T")
        elif 0.15 <= vol_60d_annual <= 0.50:
            vol_score = 25
        elif 0.10 <= vol_60d_annual <= 0.60:
            vol_score = 10
        else:
            flags.append("60 日年化波动过低或过高，做T适配度弱")

    liquidity_score = 0.0
    if avg_amount > 5e8:
        liquidity_score = 30
    elif avg_amount > 1e8:
        liquidity_score = 20
    elif avg_amount > 3e7:
        liquidity_score = 10
    if avg_amount < 3e7:
        flags.append(f"low_liquidity：20 日均成交额 {avg_amount / 1e8:.2f} 亿低于 3000 万")
    elif avg_amount >= 1e8:
        good_flags.append(f"20 日均成交额 {avg_amount / 1e8:.1f} 亿，流动性满足做T")

    beta_score = 0.0
    beta_60d = None
    idx = source.index_hist("000300", years=1)
    if idx is not None and not idx.empty and "close" in idx.columns:
        left = df[["date", "close"]].copy() if "date" in df.columns else df[["close"]].copy()
        right = idx[["date", "close"]].copy() if "date" in idx.columns else idx[["close"]].copy()
        if "date" in left.columns and "date" in right.columns:
            merged = left.merge(right, on="date", how="inner", suffixes=("_stock", "_index"))
            stock_ret = pd.to_numeric(merged["close_stock"], errors="coerce").pct_change()
            index_ret = pd.to_numeric(merged["close_index"], errors="coerce").pct_change()
        else:
            stock_ret = close.pct_change()
            index_ret = pd.to_numeric(right["close"], errors="coerce").pct_change()
        beta_df = pd.DataFrame({"s": stock_ret, "i": index_ret}).dropna().tail(60)
        if len(beta_df) >= 30 and beta_df["i"].var() > 0:
            beta_60d = float(beta_df["s"].cov(beta_df["i"]) / beta_df["i"].var())
            if 0.7 <= beta_60d <= 1.3:
                beta_score = 20
            elif 0.5 <= beta_60d <= 1.6:
                beta_score = 10
    else:
        components["beta_note"] = "缺少沪深300行情，Beta 子分按 0"
    components["beta_60d"] = round(beta_60d, 3) if beta_60d is not None else None

    trend_score = 0.0
    ret_60d = None
    if len(close.dropna()) >= 60:
        ret_60d = float(close.iloc[-1] / close.iloc[-60] - 1)
        if abs(ret_60d) < 0.15:
            trend_score = 10
        elif abs(ret_60d) < 0.25:
            trend_score = 5
        else:
            flags.append("60 日价格单边趋势较强，均值回归做T信号较弱")
    components["ret_60d_pct"] = round(ret_60d * 100, 2) if ret_60d is not None else None

    components.update(
        {
            "volatility_score": round(vol_score, 1),
            "liquidity_score": round(liquidity_score, 1),
            "beta_score": round(beta_score, 1),
            "trend_score": round(trend_score, 1),
        }
    )

    t = vol_score + liquidity_score + beta_score + trend_score
    return FactorResult(round(t, 1), components, flags, good_flags)


# ===========================================================================
# R - Risk (penalty, higher = worse)
# ===========================================================================


def risk(symbol: str) -> FactorResult:
    components: dict = {}
    flags: list[str] = []
    quick = _quick_metrics(symbol)
    fin = None if quick else source.financial_abstract(symbol)
    penalty = 0.0

    if quick is not None or (fin is not None and not fin.empty):
        fin = _load_financial_if_needed(symbol, fin)
        rev_row = _find_indicator_row(fin, "营业总收入", "营业收入")
        ni_row = _find_indicator_row(fin, "归母净利润", "归属于母公司", "净利润")
        ocf_row = _find_indicator_row(fin, "经营现金流量净额", "经营现金流", "经营活动产生的现金流量净额")
        ar_row = _find_indicator_row(fin, "应收账款", "应收票据及应收账款")
        inv_row = _find_indicator_row(fin, "存货")

        def _consecutive_decline(row: pd.Series | None, n: int) -> bool:
            if row is None:
                return False
            periods = _row_period_values(row)[: n + 1]
            if len(periods) < n + 1:
                return False
            vals = [v for _, v in periods]
            return all(vals[i] < vals[i + 1] for i in range(n))

        def _consecutive_decline_values(periods: list[tuple[str, float]], n: int) -> bool:
            if len(periods) < n + 1:
                return False
            vals = [v for _, v in periods[: n + 1]]
            return all(vals[i] < vals[i + 1] for i in range(n))

        rev_decline = _consecutive_decline_values(
            _quick_period_values(quick, "revenue_by_period", limit=config.REVENUE_DECLINE_QUARTERS + 1),
            config.REVENUE_DECLINE_QUARTERS,
        ) or _consecutive_decline(rev_row, config.REVENUE_DECLINE_QUARTERS)
        ni_decline = _consecutive_decline_values(
            _quick_period_values(quick, "ni_by_period", limit=config.NI_DECLINE_QUARTERS + 1),
            config.NI_DECLINE_QUARTERS,
        ) or _consecutive_decline(ni_row, config.NI_DECLINE_QUARTERS)

        if rev_decline:
            penalty += 35
            flags.append(f"营业收入连续 {config.REVENUE_DECLINE_QUARTERS} 期下滑")
            components["revenue_decline"] = True
        if ni_decline:
            penalty += 30
            flags.append(f"净利润连续 {config.NI_DECLINE_QUARTERS} 期下滑")
            components["net_profit_decline"] = True

        # loss_streak: ≥2 loss years in the last 3 annual periods
        ni_annual = _quick_year_values(quick, "ni_by_year", limit=3)
        if not ni_annual and ni_row is not None:
            ni_annual = _annual_only(_row_period_values(ni_row))[:3]
        if ni_annual:
            loss_years = sum(1 for _, v in ni_annual if v < 0)
            components["loss_years_in_3y"] = loss_years
            if loss_years >= 2:
                penalty += 35
                flags.append(f"loss_streak：近 {len(ni_annual)} 年中 {loss_years} 年净利润亏损")

        # ocf_disparity: OCF/NI < threshold for 2 consecutive annual periods
        if ocf_row is not None and ni_row is not None:
            ocf_annual = _annual_only(_row_period_values(ocf_row))[:2]
            ni_annual_d = {p: v for p, v in _annual_only(_row_period_values(ni_row))[:2]}
            valid = [(p, ocf_v / ni_annual_d[p]) for p, ocf_v in ocf_annual if p in ni_annual_d and ni_annual_d[p] > 0]
            if valid:
                components["latest_OCF_over_NI"] = round(valid[0][1], 2)
            below = [(p, r) for p, r in valid if r < config.OCF_OVER_NI_HARD_FLOOR]
            if len(below) >= 2:
                penalty += 20
                flags.append(f"ocf_disparity：连续 2 年经营现金流/净利润低于 {config.OCF_OVER_NI_HARD_FLOOR}")
            elif below and valid and below[0][0] == valid[0][0]:
                penalty += 10
                flags.append(f"最近年度经营现金流/净利润 {below[0][1]:.2f} 偏低")

        # AR / inventory abnormal jump (>40% YoY proxy: latest vs 4 periods ago)
        for label, row in (("应收账款", ar_row), ("存货", inv_row)):
            if row is None:
                continue
            periods = _row_period_values(row)[:5]
            if len(periods) >= 5 and periods[4][1] > 0:
                growth = (periods[0][1] - periods[4][1]) / periods[4][1]
                if growth > 0.4:
                    penalty += 10
                    flags.append(f"{label}同比增长 {growth * 100:.0f}%，需关注")
                    components[f"{label}_yoy"] = round(growth, 2)

    # tiny_cap: total market cap < 30 亿
    info = source.individual_info(symbol) or {}
    total_mv = _to_num(info.get("总市值"))
    components["total_mv"] = round(total_mv, 0) if total_mv is not None else None
    if total_mv is not None and total_mv < config.MIN_TOTAL_MV_CNY:
        penalty += 10
        flags.append(f"tiny_cap：总市值 {total_mv / 1e8:.1f} 亿，低于 {config.MIN_TOTAL_MV_CNY / 1e8:.0f} 亿门槛")

    # goodwill_heavy: goodwill / equity > 30%
    bs = source.balance_sheet(symbol)
    if bs is not None and not bs.empty and "GOODWILL" in bs.columns and "TOTAL_PARENT_EQUITY" in bs.columns:
        gw = bs["GOODWILL"].iloc[0]
        eq = bs["TOTAL_PARENT_EQUITY"].iloc[0]
        if pd.notna(gw) and pd.notna(eq) and float(eq) > 0:
            gw_ratio = float(gw) / float(eq)
            components["goodwill_to_equity"] = round(gw_ratio, 3)
            if gw_ratio > 0.30:
                penalty += 20
                flags.append(f"goodwill_heavy：商誉占归母净资产 {gw_ratio * 100:.0f}%，超过 30%")

    # Add ST/退 flag from name (set by screener via components if available)
    if components.get("name_st"):
        penalty += 100
        flags.append("ST 标识，已被剔除")

    score = round(_clip(penalty), 1)  # higher = worse
    return FactorResult(score, components, flags)


# ===========================================================================
# Aggregate fetch - run all factors for a single symbol
# ===========================================================================


def all_factors(symbol: str, name: str | None = None) -> dict:
    out = {
        "symbol": symbol,
        "name": name or "",
        "Q": quality(symbol),
        "D": dividend(symbol),
        "V": valuation(symbol),
        "T": trading(symbol),
        "R": risk(symbol),
    }
    return out

"""Valuation engine based on ../estimate.md.

The project keeps Q/D/V/T/R as UI-facing factor names, but V is now an
industry-bucket estimate instead of a flat PE/PB percentile average.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from loha import bulk, config, source

log = logging.getLogger("loha.valuation")


@dataclass
class ValuationEstimate:
    symbol: str
    industry: str
    bucket: str
    method: str
    fair_low: float | None
    fair_mid: float | None
    fair_high: float | None
    margin_of_safety: float | None
    pe_percentile_5y: float | None
    pb_percentile_5y: float | None
    primary_percentile: float | None
    value_score: float
    flags: list[str] = field(default_factory=list)
    good_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    components: dict = field(default_factory=dict)


PE_STABLE_INDUSTRIES = {
    "白酒Ⅱ",
    "非白酒",
    "饮料乳品",
    "食品加工",
    "调味发酵品Ⅱ",
    "休闲食品",
    "农产品加工",
    "个护用品",
    "化妆品",
    "一般零售",
    "旅游零售Ⅱ",
    "专业连锁Ⅱ",
    "互联网电商",
    "贸易Ⅱ",
    "白色家电",
    "黑色家电",
    "厨卫电器",
    "小家电",
    "家电零部件Ⅱ",
    "照明设备Ⅱ",
    "化学制药",
    "中药Ⅱ",
    "生物制品",
    "医疗服务",
    "医疗美容",
    "医疗器械",
    "医药商业",
    "动物保健Ⅱ",
    "服装家纺",
    "家居用品",
    "文娱用品",
    "旅游及景区",
    "酒店餐饮",
    "教育",
    "饰品",
    "汽车服务",
    "摩托车及其他",
    "软件开发",
    "计算机设备",
    "通信设备",
    "元件",
    "光学光电子",
    "消费电子",
    "电子化学品Ⅱ",
    "其他电子Ⅱ",
    "出版",
    "广告营销",
    "电视广播Ⅱ",
    "数字媒体",
    "游戏Ⅱ",
    "体育Ⅱ",
    "影视院线",
    "通用设备",
    "专用设备",
    "自动化设备",
    "轨交设备Ⅱ",
    "工程咨询服务Ⅱ",
    "专业服务",
    "综合Ⅱ",
    "装修建材",
    "装修装饰Ⅱ",
    "包装印刷",
    "农化制品",
    "非金属材料Ⅱ",
    "金属新材料",
    "燃气Ⅱ",
    "环保设备Ⅱ",
    "环境治理",
}

PB_CYCLICAL_INDUSTRIES = {
    "普钢",
    "特钢Ⅱ",
    "冶钢原料",
    "化学原料",
    "化学制品",
    "化学纤维",
    "橡胶",
    "塑料",
    "造纸",
    "玻璃玻纤",
    "水泥",
    "工业金属",
    "小金属",
    "工程机械",
    "商用车",
    "乘用车",
    "汽车零部件",
    "纺织制造",
    "焦炭Ⅱ",
    "油服工程",
    "炼化及贸易",
    "半导体",
    "电池",
    "光伏设备",
    "风电设备",
    "电机Ⅱ",
    "其他电源设备Ⅱ",
    "能源金属",
    "房屋建设Ⅱ",
    "基础建设",
    "专业工程",
}

DIV_INCOME_INDUSTRIES = {
    "电力",
    "电网设备",
    "煤炭开采",
    "油气开采Ⅱ",
    "铁路公路",
    "航空机场",
    "航运港口",
    "物流",
    "通信服务",
    "多元金融",
}

NAV_RE_INDUSTRIES = {"房地产开发", "房地产服务"}
CYCLICAL_AG_INDUSTRIES = {"养殖业", "种植业", "渔业", "饲料", "农业综合Ⅱ", "林业Ⅱ"}
DEFENSE_INDUSTRIES = {"航空装备Ⅱ", "航天装备Ⅱ", "地面兵装Ⅱ", "航海装备Ⅱ", "军工电子Ⅱ"}

EXACT_BUCKETS: dict[str, str] = {}
for _name in PE_STABLE_INDUSTRIES:
    EXACT_BUCKETS[_name] = "PE_STABLE"
for _name in PB_CYCLICAL_INDUSTRIES:
    EXACT_BUCKETS[_name] = "PB_CYCLICAL"
for _name in DIV_INCOME_INDUSTRIES:
    EXACT_BUCKETS[_name] = "DIV_INCOME"
for _name in NAV_RE_INDUSTRIES:
    EXACT_BUCKETS[_name] = "NAV_RE"
for _name in CYCLICAL_AG_INDUSTRIES:
    EXACT_BUCKETS[_name] = "CYCLICAL_AG"
for _name in DEFENSE_INDUSTRIES:
    EXACT_BUCKETS[_name] = "DEFENSE"
EXACT_BUCKETS.update(
    {
        "银行Ⅱ": "BANK",
        "保险Ⅱ": "INSURANCE",
        "证券Ⅱ": "SECURITIES",
    }
)


KEYWORD_BUCKETS: list[tuple[str, str]] = [
    (r"银行", "BANK"),
    (r"保险", "INSURANCE"),
    (r"证券|券商", "SECURITIES"),
    (r"房地产|地产", "NAV_RE"),
    (r"航空装备|航天|兵装|航海装备|军工|国防", "DEFENSE"),
    (r"煤炭|电力|水电|火电|核电|公路|铁路|港口|机场|航运|油气|石油|电信运营|运营商|通信服务|多元金融", "DIV_INCOME"),
    (r"养殖|种植|渔业|饲料|农业综合|林业", "CYCLICAL_AG"),
    (
        r"钢铁|化工|化纤|橡胶|塑料|造纸|玻璃|水泥|工业金属|小金属|工程机械|汽车|乘用车|商用车|半导体|电池|光伏|风电|锂|能源金属|建筑|基建|焦炭|油服",
        "PB_CYCLICAL",
    ),
    (
        r"白酒|酿酒|食品|饮料|乳品|调味|零售|电商|家电|医药|医疗|服装|家居|旅游|酒店|教育|软件|计算机|通信设备|电子|传媒|出版|游戏|设备|环保|燃气",
        "PE_STABLE",
    ),
]


CYCLE_PE_MID: list[tuple[str, float]] = [
    (r"钢铁|普钢|特钢|冶钢", 8.0),
    (r"化学原料|化学制品|化工", 12.0),
    (r"水泥|玻璃", 10.0),
    (r"工业金属|小金属|有色", 12.0),
    (r"造纸", 10.0),
    (r"工程机械", 13.0),
    (r"乘用车|整车", 14.0),
    (r"半导体", 35.0),
    (r"电池|光伏|风电|锂", 18.0),
    (r"焦炭|油服", 10.0),
    (r"房屋建设|基础建设|基建|建筑", 7.0),
]


def _to_num(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        val = float(v)
        return None if math.isnan(val) else val
    s = str(v).strip().replace(",", "")
    if not s or s in {"--", "-", "nan", "None"}:
        return None
    if s.endswith("%"):
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


def _clip(x: float | None, lo: float = 0.0, hi: float = 100.0) -> float:
    if x is None or math.isnan(float(x)):
        return 50.0
    return float(max(lo, min(hi, x)))


def _latest_positive(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    ser = pd.to_numeric(series, errors="coerce").dropna()
    ser = ser[ser > 0]
    return float(ser.iloc[-1]) if not ser.empty else None


def _prepare_indicator(ind: pd.DataFrame | None) -> pd.DataFrame | None:
    if ind is None or ind.empty:
        return None
    out = ind.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
        out = out.sort_values("trade_date")
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=config.VALUATION_LOOKBACK_YEARS)
        out = out[out["trade_date"] >= cutoff]
    return out


def _series(ind: pd.DataFrame | None, col: str, positive: bool = True) -> pd.Series | None:
    if ind is None or col not in ind.columns:
        return None
    ser = pd.to_numeric(ind[col], errors="coerce").dropna()
    if positive:
        ser = ser[ser > 0]
    return ser if not ser.empty else None


def _percentile(ser: pd.Series | None, current: float | None = None) -> float | None:
    if ser is None or len(ser) < 60:
        return None
    latest = current if current is not None else float(ser.iloc[-1])
    if latest is None or latest <= 0:
        return None
    return float((ser <= latest).sum() / len(ser) * 100)


def _quantile(ser: pd.Series | None, pct: float) -> float | None:
    if ser is None or len(ser) < 60:
        return None
    return float(np.percentile(ser, pct))


def _find_indicator_row(df: pd.DataFrame | None, *candidates: str) -> pd.Series | None:
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
        return None
    labels = df[label_col].astype(str)
    for cand in candidates:
        mask = labels.str.contains(cand, regex=False, na=False)
        if mask.any():
            return df[mask].iloc[0]
    return None


def _row_period_values(row: pd.Series | None) -> list[tuple[str, float]]:
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


def _annual_values(fin: pd.DataFrame | None, *candidates: str, limit: int = 5) -> list[tuple[str, float]]:
    row = _find_indicator_row(fin, *candidates)
    return _annual_only(_row_period_values(row))[:limit]


def _latest_value(fin: pd.DataFrame | None, *candidates: str) -> float | None:
    row = _find_indicator_row(fin, *candidates)
    values = _row_period_values(row)
    return values[0][1] if values else None


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


def _dividend_by_year_from_bulk_fhps(df: pd.DataFrame | None) -> dict[int, float]:
    if df is None or df.empty or "REPORT_DATE" not in df.columns or "PRETAX_BONUS_RMB" not in df.columns:
        return {}
    work = df[["REPORT_DATE", "PRETAX_BONUS_RMB"]].copy()
    work["_year"] = work["REPORT_DATE"].astype(str).str.extract(r"(\d{4})", expand=False)
    work["PRETAX_BONUS_RMB"] = pd.to_numeric(work["PRETAX_BONUS_RMB"], errors="coerce")
    work = work.dropna(subset=["_year", "PRETAX_BONUS_RMB"])
    if work.empty:
        return {}
    yearly = (work.groupby("_year")["PRETAX_BONUS_RMB"].sum() / 10.0).to_dict()
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


def _dividend_by_year(symbol: str) -> dict[int, float]:
    bulk_fhps = bulk.get_bulk_fhps_for(symbol)
    if bulk_fhps is not None and (bulk_fhps.empty or "PRETAX_BONUS_RMB" in bulk_fhps.columns):
        return _dividend_by_year_from_bulk_fhps(bulk_fhps)
    if bulk_fhps is not None:
        log.warning("bulk dividend schema missing PRETAX_BONUS_RMB for %s; fallback to dividend_hist", symbol)
    return _dividend_by_year_from_hist(source.dividend_hist(symbol))


def _mean_profit_ratio_from_quick(metrics: dict | None) -> float | None:
    annual = _quick_year_values(metrics, "ni_by_year", limit=5)
    latest = (_quick_period_values(metrics, "ni_by_period", limit=1) or [(None, None)])[0][1]
    if latest is None or latest <= 0 or not annual:
        return None
    positives = [v for _, v in annual if v > 0]
    if not positives:
        return None
    return float(np.mean(positives) / latest)


def _latest_from_quick(metrics: dict | None, key: str) -> float | None:
    values = _quick_period_values(metrics, key, limit=1)
    return values[0][1] if values else None


def _latest_year_from_quick(metrics: dict | None, key: str) -> float | None:
    values = _quick_year_values(metrics, key, limit=1)
    return values[0][1] if values else None


def map_bucket(industry: str | None) -> tuple[str, bool]:
    text = str(industry or "").strip()
    if text in EXACT_BUCKETS:
        return EXACT_BUCKETS[text], False
    for pattern, bucket in KEYWORD_BUCKETS:
        if re.search(pattern, text):
            return bucket, False
    return "PE_STABLE", True


def _cycle_pe_mid(industry: str) -> float:
    for pattern, pe in CYCLE_PE_MID:
        if re.search(pattern, industry):
            return pe
    return 12.0


def _target_price(price: float | None, current_multiple: float | None, target: float | None) -> float | None:
    if price is None or current_multiple is None or target is None:
        return None
    if price <= 0 or current_multiple <= 0 or target <= 0:
        return None
    return price * target / current_multiple


def _ordered_band(
    low: float | None,
    mid: float | None,
    high: float | None,
) -> tuple[float | None, float | None, float | None]:
    vals = [v for v in (low, mid, high) if v is not None and v > 0]
    if not vals:
        return None, None, None
    if len(vals) == 1:
        v = vals[0]
        return v * 0.9, v, v * 1.1
    if mid is None or mid <= 0:
        mid = float(np.median(vals))
    low = min(v for v in vals if v <= mid) if any(v <= mid for v in vals) else min(vals)
    high = max(v for v in vals if v >= mid) if any(v >= mid for v in vals) else max(vals)
    if low > mid:
        low = mid * 0.9
    if high < mid:
        high = mid * 1.1
    return float(low), float(mid), float(high)


def _multiple_band(
    price: float | None,
    current_multiple: float | None,
    series: pd.Series | None,
    q_low: float,
    q_mid: float,
    q_high: float,
) -> tuple[float | None, float | None, float | None]:
    low = _target_price(price, current_multiple, _quantile(series, q_low))
    mid = _target_price(price, current_multiple, _quantile(series, q_mid))
    high = _target_price(price, current_multiple, _quantile(series, q_high))
    return _ordered_band(low, mid, high)


def _mean_profit_ratio(fin: pd.DataFrame | None) -> float | None:
    vals = _annual_values(fin, "归母净利润", "归属于母公司", "净利润", limit=5)
    latest = _latest_value(fin, "归母净利润", "归属于母公司", "净利润")
    if latest is None or latest <= 0 or not vals:
        return None
    positives = [v for _, v in vals if v > 0]
    if not positives:
        return None
    return float(np.mean(positives) / latest)


def _dividend_context(symbol: str, price: float | None) -> dict:
    div_by_year = _dividend_by_year(symbol)
    current_year = pd.Timestamp.today().year
    target_years = [current_year - 1 - i for i in range(config.DIV_LOOKBACK_YEARS)]
    dps_new_to_old = [float(div_by_year.get(year, 0.0)) for year in target_years]
    dps_history = list(reversed(dps_new_to_old))
    latest_dps = next((v for v in dps_new_to_old if v > 0), 0.0)
    current_yield = latest_dps / price * 100 if price and price > 0 and latest_dps > 0 else 0.0
    yearly_yields = [d / price * 100 for d in dps_new_to_old if price and price > 0 and d > 0]
    return {
        "div_by_year": div_by_year,
        "dps_history": dps_history,
        "latest_dps": latest_dps,
        "dividend_yield_pct": current_yield,
        "dividend_yield_5y_median_pct": float(np.median(yearly_yields)) if yearly_yields else None,
    }


def _dividend_growth(dps_history: list[float]) -> float:
    vals = [v for v in dps_history if v > 0]
    if len(vals) < 2:
        return 0.0
    rates: list[float] = []
    for prev, cur in zip(vals, vals[1:]):
        if prev > 0:
            rates.append(cur / prev - 1)
    if not rates:
        return 0.0
    return min(0.05, float(np.mean(rates)))


def _fair_div_income(
    symbol: str,
    industry: str,
    price: float | None,
    pe_pct: float | None,
) -> tuple[str, float | None, float | None, float | None, list[str], dict]:
    ctx = _dividend_context(symbol, price)
    latest_dps = ctx["latest_dps"]
    extra: dict = dict(ctx)
    notes: list[str] = []
    if latest_dps <= 0:
        notes.append("缺少最近年度现金分红，DDM 估值不可用")
        return "ddm_plus_div_yield", None, None, None, notes, extra

    g_div = _dividend_growth(ctx["dps_history"])
    if re.search(r"铁路公路|电力|水电|运营商|通信服务|电信", industry):
        required_return = 0.07
    else:
        required_return = 0.08
    if re.search(r"航空机场", industry):
        g_div = 0.0
        notes.append("航空机场按强周期处理，股息增长率按 0 估计")
    if g_div >= required_return:
        g_div = required_return - 0.01

    fair_mid = latest_dps * (1 + g_div) / (required_return - g_div)
    fair_low = fair_mid * 0.85
    fair_high = fair_mid * 1.20
    fair_high_check = latest_dps / 0.04
    fair_low_check = latest_dps / 0.06
    fair_mid = (fair_mid + (fair_high_check + fair_low_check) / 2) / 2
    if re.search(r"煤炭|油气|石油", industry) and pe_pct is not None and pe_pct > 70:
        fair_mid *= 0.85
        notes.append("煤炭/油气 PE 分位偏高，按商品周期顶部风险折价 15%")
    extra.update(
        {
            "g_dividend": round(g_div * 100, 2),
            "required_return": round(required_return * 100, 2),
        }
    )
    return "ddm_plus_div_yield", fair_low, fair_mid, fair_high, notes, extra


def evaluate(symbol: str, name: str | None = None) -> ValuationEstimate:
    symbol = symbol.zfill(6)
    info = source.individual_info(symbol) or {}
    industry = str(info.get("行业", "") or "")
    bucket, fallback = map_bucket(industry)
    method = ""
    flags: list[str] = []
    good_flags: list[str] = []
    notes: list[str] = []
    components: dict = {"industry": industry, "bucket": bucket}

    if fallback:
        flags.append("bucket_fallback：行业未命中 estimate.md 映射，默认按 PE_STABLE")

    raw_ind = source.indicator_hist(symbol)
    ind = _prepare_indicator(raw_ind)
    if ind is None or ind.empty:
        return ValuationEstimate(
            symbol,
            industry,
            bucket,
            "unknown",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            50.0,
            ["缺少估值数据"],
            [],
            [],
            components,
        )

    price = _latest_positive(_series(ind, "close", positive=True))
    pe_ser = _series(ind, "pe_ttm", positive=True)
    if pe_ser is None:
        pe_ser = _series(ind, "pe", positive=True)
    pb_ser = _series(ind, "pb", positive=True)
    current_pe = _latest_positive(pe_ser)
    current_pb = _latest_positive(pb_ser)
    pe_pct = _percentile(pe_ser, current_pe)
    pb_pct = _percentile(pb_ser, current_pb)

    components.update(
        {
            "price": round(price, 3) if price else None,
            "pe_ttm_latest": round(current_pe, 2) if current_pe else None,
            "pb_latest": round(current_pb, 2) if current_pb else None,
            "pe_percentile_5y": round(pe_pct, 1) if pe_pct is not None else None,
            "pb_percentile_5y": round(pb_pct, 1) if pb_pct is not None else None,
        }
    )

    if len(ind) < 1000:
        flags.append("data_insufficient：近 5 年估值历史不足 1000 个交易日")

    quick = _quick_metrics(symbol)
    fin = source.financial_abstract(symbol)
    roe_vals = [v for _, v in _quick_year_values(quick, "roe_by_year", limit=5)]
    if not roe_vals:
        roe_vals = [v for _, v in _annual_values(fin, "净资产收益率", "ROE", limit=5)]
    roe_3y_avg = float(np.mean(roe_vals[:3])) if roe_vals[:3] else None
    roe_5y_std = float(np.std(roe_vals, ddof=0)) if len(roe_vals) >= 2 else None
    roe_latest = _latest_from_quick(quick, "roe_by_period")
    if roe_latest is None:
        roe_latest = _latest_value(fin, "净资产收益率", "ROE")
    debt_ratio = _latest_value(fin, "资产负债率")
    net_profit_latest = _latest_from_quick(quick, "ni_by_period")
    if net_profit_latest is None:
        net_profit_latest = _latest_value(fin, "归母净利润", "归属于母公司", "净利润")
    components.update(
        {
            "roe_3y_avg": round(roe_3y_avg, 2) if roe_3y_avg is not None else None,
            "roe_5y_std": round(roe_5y_std, 2) if roe_5y_std is not None else None,
            "debt_to_asset": round(debt_ratio, 2) if debt_ratio is not None else None,
            "net_profit_latest": round(net_profit_latest, 2) if net_profit_latest is not None else None,
        }
    )
    if net_profit_latest is not None and net_profit_latest < 0:
        flags.append("negative_profit：最近一期净利润为负")
    if debt_ratio is not None and debt_ratio > 70 and bucket not in {"BANK", "INSURANCE", "SECURITIES"}:
        flags.append(f"over_leverage：非金融股资产负债率 {debt_ratio:.0f}% 偏高")

    effective_bucket = bucket
    if bucket == "PE_STABLE" and roe_5y_std is not None and roe_5y_std > 8:
        effective_bucket = "PB_CYCLICAL"
        components["bucket_effective"] = effective_bucket
        notes.append("ROE 5 年波动超过 8，按 PB_CYCLICAL 退化估值")
    else:
        components["bucket_effective"] = effective_bucket

    fair_low: float | None = None
    fair_mid: float | None = None
    fair_high: float | None = None
    extra_components: dict = {}

    if effective_bucket == "PE_STABLE":
        method = "historical_pe_band"
        fair_low, fair_mid, fair_high = _multiple_band(price, current_pe, pe_ser, 25, 50, 75)

    elif effective_bucket == "PB_CYCLICAL":
        method = "pb_band_with_cycle_pe"
        low, mid_pb, high = _multiple_band(price, current_pb, pb_ser, 20, 50, 80)
        ratio = _mean_profit_ratio_from_quick(quick)
        if ratio is None:
            ratio = _mean_profit_ratio(fin)
        cycle_pe = _cycle_pe_mid(industry)
        mid_cycle = None
        if current_pe and ratio is not None:
            mid_cycle = price * cycle_pe / current_pe * ratio if price else None
        fair_mid = (
            min(v for v in (mid_pb, mid_cycle) if v is not None)
            if any(v is not None for v in (mid_pb, mid_cycle))
            else None
        )
        fair_low, fair_mid, fair_high = _ordered_band(low, fair_mid, high)
        extra_components.update(
            {
                "cycle_pe_mid": cycle_pe,
                "cycle_profit_ratio": round(ratio, 3) if ratio is not None else None,
                "fair_mid_pb": round(mid_pb, 3) if mid_pb else None,
                "fair_mid_cycle_pe": round(mid_cycle, 3) if mid_cycle else None,
            }
        )
        if current_pe is not None and current_pe < 5 and pb_pct is not None and pb_pct > 70:
            flags.append("cycle_top_risk：低 PE 叠加 PB 高分位，疑似周期顶部")

    elif effective_bucket == "BANK":
        method = "pb_roe"
        if current_pb and price and roe_3y_avg is not None:
            implied_pb = max(0.0, (roe_3y_avg / 100 - 0.03) / (0.10 - 0.03))
            fair_mid = price * implied_pb / current_pb
            fair_low = fair_mid * 0.85
            fair_high = fair_mid * 1.15
            extra_components["implied_pb"] = round(implied_pb, 3)
        if current_pb is not None and current_pb < 0.5 and roe_latest is not None and roe_latest < 8:
            flags.append("state_bank_value_trap：PB<0.5 且 ROE<8%，低估可能是常态")

    elif effective_bucket == "INSURANCE":
        method = "pb"
        if current_pb and price and roe_3y_avg is not None:
            implied_pb = max(0.0, (roe_3y_avg / 100 - 0.02) / (0.11 - 0.02))
            fair_mid = price * implied_pb / current_pb
            fair_low = fair_mid * 0.85
            fair_high = fair_mid * 1.15
            extra_components["implied_pb"] = round(implied_pb, 3)
            notes.append("缺少 P/EV 数据，按寿险 PB-ROE 退化估值")

    elif effective_bucket == "SECURITIES":
        method = "pb_band"
        pb_low = max(0.9, _quantile(pb_ser, 15) or 0.9)
        pb_mid = _quantile(pb_ser, 50)
        pb_high_raw = _quantile(pb_ser, 85)
        pb_high = min(2.5, pb_high_raw) if pb_high_raw is not None else None
        fair_low = _target_price(price, current_pb, pb_low)
        fair_mid = _target_price(price, current_pb, pb_mid)
        fair_high = _target_price(price, current_pb, pb_high)
        fair_low, fair_mid, fair_high = _ordered_band(fair_low, fair_mid, fair_high)

    elif effective_bucket == "DIV_INCOME":
        method, fair_low, fair_mid, fair_high, div_notes, extra_components = _fair_div_income(
            symbol,
            industry,
            price,
            pe_pct,
        )
        notes.extend(div_notes)

    elif effective_bucket == "NAV_RE":
        method = "nav_pb_floor"
        pb_mid, pb_low, pb_high = 0.4, 0.25, 0.7
        if name and re.search(r"保利|招商|华润|华发|越秀|陆家嘴|金融街|首开", name):
            pb_mid, pb_low, pb_high = 0.7, 0.5, 1.0
            extra_components["central_soe_proxy"] = True
        else:
            flags.append("re_private_developer_risk：未识别央国企地产标签，按民营开发商折价")
        fair_low = _target_price(price, current_pb, pb_low)
        fair_mid = _target_price(price, current_pb, pb_mid)
        fair_high = _target_price(price, current_pb, pb_high)
        if debt_ratio is not None and debt_ratio > 75:
            flags.append("re_high_leverage：资产负债率超过 75%")
        # re_liquidity_risk: cash/short-debt ratio (estimate.md §5.7)
        bs = source.balance_sheet(symbol)
        if bs is not None and not bs.empty:

            def _bs_val(col):
                import math

                if col in bs.columns:
                    v = bs[col].iloc[0]
                    try:
                        fv = float(v)
                        return fv if not math.isnan(fv) else None
                    except (TypeError, ValueError):
                        pass
                return None

            cash = _bs_val("MONETARYFUNDS")
            st_loan = _bs_val("SHORT_LOAN")
            st_due = _bs_val("NONCURRENT_LIAB_1YEAR")
            if cash is not None:
                st_debt = (st_loan or 0.0) + (st_due or 0.0)
                if st_debt > 0:
                    is_central_soe = extra_components.get("central_soe_proxy", False)
                    restriction_factor = 0.2 if is_central_soe else 0.5
                    effective_cash = cash * (1 - restriction_factor)
                    ratio = effective_cash / st_debt
                    extra_components["cash_short_debt_ratio"] = round(ratio, 2)
                    extra_components["re_restriction_factor"] = restriction_factor
                    if ratio < 1.0:
                        flags.append(
                            f"re_liquidity_risk：现金短债比 {ratio:.2f}"
                            f"（受限折扣 {restriction_factor * 100:.0f}%），低于安全线 1.0"
                        )

    elif effective_bucket == "CYCLICAL_AG":
        method = "pb_cyclical_fallback"
        notes.append("缺少年出栏量/头均市值数据，农业周期股退化为 PB 周期估值")
        low, mid, high = _multiple_band(price, current_pb, pb_ser, 20, 50, 80)
        fair_low, fair_mid, fair_high = _ordered_band(low, mid, high)

    elif effective_bucket == "DEFENSE":
        method = "pe_band_high_tolerance"
        pe_low = max(25.0, _quantile(pe_ser, 25) or 25.0)
        pe_mid = max(35.0, _quantile(pe_ser, 50) or 35.0)
        pe_high_raw = _quantile(pe_ser, 75)
        pe_high = min(60.0, pe_high_raw) if pe_high_raw is not None else 60.0
        fair_low = _target_price(price, current_pe, pe_low)
        fair_mid = _target_price(price, current_pe, pe_mid)
        fair_high = _target_price(price, current_pe, pe_high)
        fair_low, fair_mid, fair_high = _ordered_band(fair_low, fair_mid, fair_high)

    fair_low, fair_mid, fair_high = _ordered_band(fair_low, fair_mid, fair_high)
    components.update(extra_components)
    components.update(
        {
            "method": method,
            "fair_low": round(fair_low, 3) if fair_low else None,
            "fair_mid": round(fair_mid, 3) if fair_mid else None,
            "fair_high": round(fair_high, 3) if fair_high else None,
        }
    )

    margin = (fair_mid - price) / price if fair_mid and price else None
    components["margin_of_safety_pct"] = round(margin * 100, 1) if margin is not None else None

    primary_by_bucket = {
        "PE_STABLE": pe_pct,
        "PB_CYCLICAL": pb_pct,
        "BANK": pb_pct,
        "INSURANCE": pb_pct,
        "SECURITIES": pb_pct,
        "DIV_INCOME": pe_pct,
        "NAV_RE": pb_pct,
        "CYCLICAL_AG": pb_pct,
        "DEFENSE": pe_pct,
    }
    primary_pct = primary_by_bucket.get(effective_bucket)
    if primary_pct is None:
        candidates = [v for v in (pe_pct, pb_pct) if v is not None]
        primary_pct = float(np.mean(candidates)) if candidates else None

    score = 50.0 if primary_pct is None else 100.0 * (1 - primary_pct / 100)
    if primary_pct is not None and primary_pct < 5:
        if (roe_3y_avg is not None and roe_3y_avg < 5) or (net_profit_latest is not None and net_profit_latest < 0):
            score *= 0.5
            flags.append("value_trap：估值极低但盈利质量弱")
        elif debt_ratio is not None and debt_ratio > 70 and effective_bucket != "BANK":
            score *= 0.7
            flags.append("value_trap：估值极低但杠杆偏高")

    if effective_bucket == "DIV_INCOME":
        div_ctx = extra_components or _dividend_context(symbol, price)
        dy_now = div_ctx.get("dividend_yield_pct")
        dy_median = div_ctx.get("dividend_yield_5y_median_pct")
        if dy_now is not None:
            components["dividend_yield_pct"] = round(float(dy_now), 2)
        if dy_median is not None:
            components["dividend_yield_5y_median_pct"] = round(float(dy_median), 2)
        if dy_now and dy_median and dy_now > dy_median * 1.1:
            score = min(100.0, score + 10.0)
            good_flags.append("当前股息率高于近 5 年中位数 10% 以上")

    score = _clip(score)
    components.update(
        {
            "primary_percentile": round(primary_pct, 1) if primary_pct is not None else None,
            "valuation_percentile": round(primary_pct, 1) if primary_pct is not None else None,
            "value_score": round(score, 1),
        }
    )

    if primary_pct is not None:
        if primary_pct >= config.PERCENTILE_EXPENSIVE:
            flags.append(f"估值分位 {primary_pct:.0f}% 已进入偏贵带")
        elif primary_pct <= config.PERCENTILE_CHEAP:
            good_flags.append(f"估值分位 {primary_pct:.0f}% 处于低估带")
    if margin is not None and margin >= 0.20:
        good_flags.append(f"估值安全边际 {margin * 100:.0f}%")
    elif margin is not None and margin <= -0.10:
        flags.append(f"当前价高于估值中枢 {abs(margin) * 100:.0f}%")

    components["notes"] = notes
    return ValuationEstimate(
        symbol=symbol,
        industry=industry,
        bucket=bucket,
        method=method,
        fair_low=fair_low,
        fair_mid=fair_mid,
        fair_high=fair_high,
        margin_of_safety=margin,
        pe_percentile_5y=pe_pct,
        pb_percentile_5y=pb_pct,
        primary_percentile=primary_pct,
        value_score=round(score, 1),
        flags=flags,
        good_flags=good_flags,
        notes=notes,
        components=components,
    )

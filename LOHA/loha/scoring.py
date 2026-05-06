"""Composite score & ranking.

estimate.md strategy fit:
    Score = 0.45D + 0.35V + 0.20T
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loha import config
from loha.factors import FactorResult

GOOD_SIGNAL_MARKERS = (
    "处于低估带",
    "估值安全边际",
    "股息率",
    "加权股息率",
    "分红率与经营现金流覆盖",
    "现金流覆盖利润良好",
    "ROE 三年中位数",
    "负债率",
    "流动性充足",
    "波动适合网格交易",
)

NEGATIVE_SIGNAL_MARKERS = (
    "低于",
    "偏低",
    "偏高",
    "下滑",
    "不足",
    "缺少",
    "失败",
    "风险",
    "仅",
    "过低",
    "过高",
    "偏贵",
    "剔除",
    "value_trap",
    "cycle_top_risk",
    "data_insufficient",
    "dividend_unstable",
)

DEPRECATED_SIGNAL_MARKERS = ()


def split_signals(flags: list[str]) -> tuple[list[str], list[str]]:
    """Split legacy mixed flags into (risk_flags, good_flags)."""
    risk_flags: list[str] = []
    good_flags: list[str] = []
    for flag in flags:
        text = str(flag)
        if any(marker in text for marker in DEPRECATED_SIGNAL_MARKERS):
            continue
        is_good = any(marker in text for marker in GOOD_SIGNAL_MARKERS)
        is_negative = any(marker in text for marker in NEGATIVE_SIGNAL_MARKERS)
        if is_good and not is_negative:
            good_flags.append(text)
        elif "处于低估带" in text:
            good_flags.append(text)
        else:
            risk_flags.append(text)
    return risk_flags, good_flags


def derive_good_flags(row: dict) -> list[str]:
    """Derive positive signals from stored factor components.

    This upgrades old full-scan cache rows that only had mixed `flags`.
    """
    components = row.get("components") or {}
    q = components.get("Q") or {}
    d = components.get("D") or {}
    v = components.get("V") or {}
    t = components.get("T") or {}
    out: list[str] = []

    roe = q.get("ROE_median_3y")
    if isinstance(roe, (int, float)) and roe >= 15:
        out.append(f"[Q] ROE 三年中位数 {roe:.1f}% 较高")

    cash = q.get("OCF_over_NI_3y")
    if isinstance(cash, (int, float)) and cash >= 1.0:
        out.append(f"[Q] 经营现金流 / 净利润 三年均值 {cash:.2f}，现金流覆盖利润良好")

    debt = q.get("debt_ratio")
    if isinstance(debt, (int, float)) and debt < 50 and not q.get("debt_note"):
        out.append(f"[Q] 资产负债率 {debt:.0f}% 较稳健")

    div_yield = d.get("weighted_dividend_yield")
    if isinstance(div_yield, (int, float)) and div_yield >= config.DIV_YIELD_FULL_CREDIT:
        out.append(f"[D] 5年加权股息率 {div_yield:.1f}% 达到策略目标")

    val_pct = v.get("valuation_percentile")
    if isinstance(val_pct, (int, float)) and val_pct <= config.PERCENTILE_CHEAP:
        bucket = v.get("bucket_effective") or v.get("bucket") or "估值桶"
        out.append(f"[V] {bucket} 估值分位 {val_pct:.0f}% 处于低估带")

    mos = v.get("margin_of_safety_pct")
    if isinstance(mos, (int, float)) and mos >= 20:
        out.append(f"[V] 估值安全边际 {mos:.0f}%")

    vol = t.get("vol_60d_annual_pct")
    if isinstance(vol, (int, float)) and 20 <= vol <= 40:
        out.append(f"[T] 60 日年化波动 {vol:.1f}%，适合小幅做T")

    amount = t.get("avg_turnover_cny")
    if isinstance(amount, (int, float)) and amount >= config.LIQUIDITY_MIN_TURNOVER_CNY * 3:
        out.append(f"[T] 日均成交额 {amount / 1e8:.1f} 亿，流动性充足")

    return out


def dedupe_signals(flags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for flag in flags:
        if any(marker in str(flag) for marker in DEPRECATED_SIGNAL_MARKERS):
            continue
        if flag not in seen:
            out.append(flag)
            seen.add(flag)
    return out


def filter_active_signals(flags: list[str]) -> list[str]:
    return [flag for flag in flags if not any(marker in str(flag) for marker in DEPRECATED_SIGNAL_MARKERS)]


@dataclass
class StockScore:
    symbol: str
    name: str
    score: float
    Q: float
    D: float
    V: float
    T: float
    R: float
    flags: list[str] = field(default_factory=list)
    components: dict = field(default_factory=dict)
    good_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "score": round(self.score, 1),
            "Q": round(self.Q, 1),
            "D": round(self.D, 1),
            "V": round(self.V, 1),
            "T": round(self.T, 1),
            "R": round(self.R, 1),
            "flags": self.flags,
            "risk_flags": self.flags,
            "good_flags": self.good_flags,
            "components": self.components,
            "scoring_rule_version": config.SCORING_RULE_VERSION,
        }


def composite(symbol: str, name: str, factors: dict[str, FactorResult]) -> StockScore:
    Q = factors["Q"].score
    D = factors["D"].score
    V = factors["V"].score
    T = factors["T"].score
    R = factors["R"].score

    score = (
        config.W_QUALITY * Q + config.W_DIVIDEND * D + config.W_VALUATION * V + config.W_TRADING * T - config.W_RISK * R
    )

    raw_risk_flags: list[str] = []
    good_flags: list[str] = []
    components: dict = {}
    for key, res in factors.items():
        for f in res.flags:
            raw_risk_flags.append(f"[{key}] {f}")
        for f in res.good_flags:
            good_flags.append(f"[{key}] {f}")
        components[key] = res.components

    risk_flags, legacy_good_flags = split_signals(raw_risk_flags)
    good_flags = dedupe_signals(legacy_good_flags + good_flags)

    return StockScore(
        symbol,
        name,
        score,
        Q,
        D,
        V,
        T,
        R,
        flags=risk_flags,
        components=components,
        good_flags=good_flags,
    )

"""Trading interval calculation based on estimate.md.

Primary valuation comes from the industry bucket engine. The T band follows
estimate.md §6.3: 60-day Bollinger-style band clipped by fair_mid ±15%.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from loha import config, source
from loha import valuation as valuation_engine

log = logging.getLogger("loha.interval")

Band = Literal["cheap", "neutral", "expensive", "unknown"]


@dataclass
class TradingInterval:
    symbol: str
    last_price: float
    band: Band  # 当前估值带
    buy_low: float | None  # 买入下沿
    buy_mid: float | None  # 中性带下沿（视情况展示）
    sell_high: float | None  # 卖出上沿
    atr: float | None
    atr_pct: float | None
    valuation_pct: float | None  # bucket primary valuation percentile
    components: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def lot_size(self) -> int:
        if self.last_price and self.last_price < config.LOT_PRICE_BREAKPOINT:
            return config.LOT_SIZE_LOW_PRICE
        return config.LOT_SIZE_HIGH_PRICE

    def shares_per_ladder(self) -> int:
        return self.lot_size

    def action_hint(self) -> str:
        if not self.last_price:
            return "数据不足"
        if self.sell_high is not None and self.last_price >= self.sell_high:
            return "减仓或只减不加"
        if self.band == "expensive":
            return "停止加仓"
        if self.buy_low is not None and self.last_price <= self.buy_low:
            return "可分批加仓"
        if self.band == "cheap":
            return "低估观察，等待网格买点"
        return "持有观察"

    def budget_plan(self) -> dict:
        lot_value = self.last_price * self.lot_size if self.last_price else 0
        core_budget = config.PER_STOCK_BUDGET_CNY * config.CORE_POSITION_RATIO
        tactical_budget = config.PER_STOCK_BUDGET_CNY * config.TACTICAL_POSITION_RATIO
        return {
            "per_stock_budget": config.PER_STOCK_BUDGET_CNY,
            "core_budget": round(core_budget, 2),
            "tactical_budget": round(tactical_budget, 2),
            "lot_size": self.lot_size,
            "lot_value": round(lot_value, 2) if lot_value else None,
            "core_lots": math.floor(core_budget / lot_value) if lot_value else 0,
            "tactical_lots": math.floor(tactical_budget / lot_value) if lot_value else 0,
        }

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "last_price": round(self.last_price, 3) if self.last_price else None,
            "band": self.band,
            "buy_low": round(self.buy_low, 3) if self.buy_low else None,
            "buy_mid": round(self.buy_mid, 3) if self.buy_mid else None,
            "sell_high": round(self.sell_high, 3) if self.sell_high else None,
            "atr": round(self.atr, 3) if self.atr else None,
            "atr_pct": round(self.atr_pct, 2) if self.atr_pct else None,
            "valuation_pct": round(self.valuation_pct, 1) if self.valuation_pct is not None else None,
            "lot_size": self.lot_size,
            "action_hint": self.action_hint(),
            "budget_plan": self.budget_plan(),
            "components": self.components,
            "notes": self.notes,
        }


def _atr(price_df: pd.DataFrame) -> tuple[float | None, float | None]:
    if price_df is None or price_df.empty or "close" not in price_df.columns:
        return None, None
    df = price_df.tail(config.ATR_WINDOW + 5).copy()
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
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
    last = float(close.iloc[-1])
    if atr is None or pd.isna(atr) or last <= 0:
        return None, None
    return float(atr), float(atr) / last * 100


def compute(symbol: str, last_price: float | None = None) -> TradingInterval:
    """Build the estimate.md valuation + T-band interval for a symbol."""
    # ATR and current price only need recent daily bars. Using 1-year data
    # avoids forcing a slow 5-year行情 fetch on every detail page.
    px = source.price_hist(symbol, years=1)
    if px is None or px.empty or "close" not in px.columns:
        px = source.price_hist(symbol, years=config.VALUATION_LOOKBACK_YEARS)
    notes: list[str] = []
    px_len = len(px) if px is not None else 0

    if last_price is None:
        if px is not None and not px.empty and "close" in px.columns:
            last_price = float(px["close"].astype(float).iloc[-1])
        else:
            ind = source.indicator_hist(symbol)
            if ind is not None and not ind.empty and "close" in ind.columns:
                closes = ind["close"].dropna()
                if not closes.empty:
                    last_price = float(closes.iloc[-1])
        if last_price is None:
            return TradingInterval(symbol, 0.0, "unknown", None, None, None, None, None, None, {}, ["缺少行情"])

    atr, atr_pct = _atr(px) if px is not None else (None, None)

    est = valuation_engine.evaluate(symbol)
    val_components = dict(est.components)
    val_pct = est.primary_percentile
    fair_mid = est.fair_mid

    boll_low = boll_high = None
    if px is not None and not px.empty and "close" in px.columns:
        close = pd.to_numeric(px["close"], errors="coerce").dropna().tail(60)
        if len(close) >= 20:
            ma60 = float(close.mean())
            std60 = float(close.std())
            boll_low = ma60 - 1.5 * std60
            boll_high = ma60 + 1.5 * std60
            val_components["boll_low_60"] = round(boll_low, 3)
            val_components["boll_high_60"] = round(boll_high, 3)
    if boll_low is None or boll_high is None:
        notes.append("无法计算 60 日波动带")

    valuation_buy = fair_mid * 0.85 if fair_mid else None
    valuation_sell = fair_mid * 1.15 if fair_mid else None
    val_components["valuation_buy_anchor"] = round(valuation_buy, 3) if valuation_buy else None
    val_components["valuation_sell_anchor"] = round(valuation_sell, 3) if valuation_sell else None

    buy_low = None
    sell_high = None
    if boll_low is not None and boll_high is not None and fair_mid:
        clipped_low = max(boll_low, valuation_buy)
        clipped_high = min(boll_high, valuation_sell)
        if clipped_low <= clipped_high:
            buy_low = clipped_low
            sell_high = clipped_high
        else:
            buy_low = boll_low
            sell_high = boll_high
            notes.append("估值裁剪与 60 日波动带无交集，暂按波动带展示并结合估值带判断")
    elif boll_low is not None and boll_high is not None:
        buy_low = boll_low
        sell_high = boll_high
        notes.append("缺少估值中枢，交易区间暂按 60 日波动带")
    elif fair_mid:
        buy_low = valuation_buy
        sell_high = valuation_sell
        notes.append("缺少波动带，交易区间暂按估值中枢 ±15%")

    # Mid (entry into core position) = average of (buy_low, last_price) when below current.
    buy_mid: float | None = None
    if buy_low is not None and last_price > buy_low:
        buy_mid = (buy_low + last_price) / 2

    # Band classification
    margin = est.margin_of_safety
    if val_pct is None and margin is None:
        band: Band = "unknown"
    elif (margin is not None and margin >= 0.15) or (val_pct is not None and val_pct <= config.PERCENTILE_CHEAP):
        band = "cheap"
    elif (margin is not None and margin <= -0.10) or (val_pct is not None and val_pct >= config.PERCENTILE_EXPENSIVE):
        band = "expensive"
    else:
        band = "neutral"

    if band == "expensive":
        notes.append("当前价格相对估值中枢偏贵，策略上以观察或减仓为主")
    elif band == "cheap":
        notes.append("当前价格具备估值安全边际，可结合做T区间与风险信号分批处理")
    notes.extend(est.notes)
    log.debug(
        "interval diagnostics symbol=%s px_is_none=%s px_len=%s fair_mid_is_none=%s "
        "boll_low_is_none=%s boll_high_is_none=%s atr_is_none=%s buy_mid_is_none=%s",
        symbol,
        px is None,
        px_len,
        fair_mid is None,
        boll_low is None,
        boll_high is None,
        atr is None,
        buy_mid is None,
    )

    return TradingInterval(
        symbol=symbol,
        last_price=last_price,
        band=band,
        buy_low=buy_low,
        buy_mid=buy_mid,
        sell_high=sell_high,
        atr=atr,
        atr_pct=atr_pct,
        valuation_pct=val_pct,
        components=val_components,
        notes=notes,
    )

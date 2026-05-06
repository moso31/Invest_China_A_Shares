"""AKShare wrappers - one function per data kind, each cached.

Every function returns a DataFrame (possibly empty) or None on hard failure.
Callers must handle None gracefully; AKShare endpoints fail intermittently
and we never want one stock's bad fetch to abort an entire screen.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from contextlib import contextmanager
from typing import Any

from loha.eastmoney import EastmoneyCooldownError, install_requests_cookie_patch

install_requests_cookie_patch()

import akshare as ak  # noqa: E402
import pandas as pd  # noqa: E402

from loha import cache, request_stats  # noqa: E402

log = logging.getLogger("loha.source")
_CACHE_ONLY_DEPTH = 0
_FAILED_FETCH_TTL_HOURS = 1.0
_MIN_FULL_MARKET_SNAPSHOT_ROWS = 1000


@contextmanager
def cache_only():
    """Use local cache only; never call AKShare inside this context."""
    global _CACHE_ONLY_DEPTH
    _CACHE_ONLY_DEPTH += 1
    try:
        with cache.allow_stale_reads():
            yield
    finally:
        _CACHE_ONLY_DEPTH = max(0, _CACHE_ONLY_DEPTH - 1)


def is_cache_only() -> bool:
    return _CACHE_ONLY_DEPTH > 0


def _retry(fn, *args, attempts: int = 2, sleep: float = 0.4, **kwargs):
    if _CACHE_ONLY_DEPTH > 0:
        log.info("skip akshare call in cache-only mode: %s args=%s", fn.__name__, args)
        return None
    last_err: Exception | None = None
    name = getattr(fn, "__name__", str(fn))
    for i in range(attempts):
        try:
            log.debug("akshare call: %s", name)
            result = fn(*args, **kwargs)
            request_stats.record_api_call(name, _result_ok(result))
            return result
        except EastmoneyCooldownError as e:
            request_stats.record_api_call(name, False)
            last_err = e
            break
        except Exception as e:  # noqa: BLE001
            request_stats.record_api_call(name, False)
            last_err = e
            time.sleep(sleep * (i + 1))
    log.warning("akshare call failed: %s args=%s err=%s", name, args, last_err)
    return None


def _result_ok(result: object) -> bool:
    if result is None:
        return False
    empty = getattr(result, "empty", None)
    if isinstance(empty, bool):
        return not empty
    return True


def _fix_mojibake_str(s):
    """Recover GBK strings that were mis-decoded as Latin-1.

    AKShare's financial endpoints occasionally return responses where the
    GBK bytes are decoded as latin-1, leaving us with strings like 'ѡ��'
    instead of '选项'. We detect that by the absence of CJK and the presence
    of high-bit Latin-1 characters, then re-encode/decode through GBK.
    """
    if not isinstance(s, str):
        return s
    if any("一" <= c <= "鿿" for c in s):
        return s  # real Chinese already
    if not any(0x80 <= ord(c) < 0x500 for c in s):
        return s  # pure ASCII / digits
    for wrong_encoding in ("latin-1", "cp1252", "cp1251"):
        try:
            recovered = s.encode(wrong_encoding).decode("gbk")
            if any("一" <= c <= "鿿" for c in recovered):
                return recovered
        except (UnicodeError, UnicodeDecodeError):
            pass
    return s


def _fix_mojibake_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [_fix_mojibake_str(c) for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(_fix_mojibake_str)
    return df


# ---- Universe -------------------------------------------------------------


def universe_all() -> pd.DataFrame | None:
    """Full A-share code/name list."""
    cached = cache.get("universe", "all_a")
    if cached is not None:
        return cached
    df = _retry(ak.stock_info_a_code_name)
    if df is None or df.empty:
        return None
    df = df.rename(columns={"code": "symbol", "name": "name"})
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    cache.put("universe", df, "all_a")
    return df


# ---- Individual info ------------------------------------------------------


def individual_info(symbol: str) -> dict[str, Any] | None:
    """Returns a dict with keys like 名称, 行业, 总市值, 流通市值."""
    cached = cache.get_json("info", symbol)
    if cached is not None:
        return cached
    df = _retry(ak.stock_individual_info_em, symbol=symbol)
    if df is None or df.empty:
        stale = cache.get_stale_json("info", symbol)
        if stale is not None:
            log.warning("individual_info fetch failed for %s; using stale cached data", symbol)
            return stale
        cache.put_json("info", {}, symbol, ttl_hours=_FAILED_FETCH_TTL_HOURS)
        return None
    out: dict[str, Any] = {}
    # df columns: item, value
    try:
        for _, row in df.iterrows():
            out[str(row["item"])] = row["value"]
    except Exception:
        return None
    cache.put_json("info", out, symbol)
    return out


# ---- Daily price history --------------------------------------------------


def _tx_symbol(symbol: str) -> str | None:
    symbol = str(symbol).zfill(6)
    if symbol.startswith("6"):
        return f"sh{symbol}"
    if symbol.startswith(("0", "3")):
        return f"sz{symbol}"
    return None


def _tx_price_hist(symbol: str, start: dt.date, end: dt.date, adjust: str = "qfq") -> pd.DataFrame | None:
    tx_symbol = _tx_symbol(symbol)
    if tx_symbol is None:
        return None
    try:
        from akshare.stock_feature.stock_hist_tx import stock_zh_a_hist_tx
    except Exception:  # noqa: BLE001
        return None
    try:
        df = stock_zh_a_hist_tx(
            symbol=tx_symbol,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust=adjust,
            timeout=15,
        )
        request_stats.record_api_call("stock_zh_a_hist_tx", _result_ok(df), provider="腾讯")
    except Exception as e:  # noqa: BLE001
        request_stats.record_api_call("stock_zh_a_hist_tx", False, provider="腾讯")
        log.warning("Tencent price_hist fallback failed for %s: %s", symbol, e)
        return None
    if df is None or df.empty:
        return None
    out = _normalise_tx_price_hist(df)
    return out


def _normalise_tx_price_hist(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalise Tencent kline data.

    AKShare's Tencent fallback names its lot volume column `amount`. The rest
    of LOHA expects `amount` to be CNY turnover, so convert lots * 100 * close.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    for col in ("open", "close", "high", "low", "amount", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "amount" in out.columns and "close" in out.columns:
        lots = pd.to_numeric(out["amount"], errors="coerce")
        close = pd.to_numeric(out["close"], errors="coerce")
        out["volume"] = lots
        out["amount"] = lots * 100 * close
    return out


def _repair_cached_tx_price_hist(df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, bool]:
    if df is None or df.empty or "amount" not in df.columns or "close" not in df.columns:
        return df, False
    if "volume" in df.columns:
        return df, False
    cols = {str(c).lower() for c in df.columns}
    looks_like_tx = {"date", "open", "close", "high", "low", "amount"}.issubset(cols)
    if not looks_like_tx:
        return df, False
    amount = pd.to_numeric(df["amount"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    if amount.dropna().empty or close.dropna().empty:
        return df, False
    # Tencent lot-volume values are usually far below CNY turnover values.
    if float(amount.tail(20).median()) >= 50_000_000:
        return df, False
    repaired = df.copy()
    repaired["volume"] = amount
    repaired["amount"] = amount * 100 * close
    return repaired, True


def _tx_index_symbol(symbol: str) -> str:
    symbol = str(symbol).zfill(6)
    if symbol.startswith("399"):
        return f"sz{symbol}"
    return f"sh{symbol}"


def _tx_index_hist(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame | None:
    try:
        from akshare.stock_feature.stock_hist_tx import stock_zh_a_hist_tx
    except Exception:  # noqa: BLE001
        return None
    try:
        df = stock_zh_a_hist_tx(
            symbol=_tx_index_symbol(symbol),
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
            timeout=15,
        )
        request_stats.record_api_call("stock_zh_a_hist_tx", _result_ok(df), provider="腾讯")
    except Exception as e:  # noqa: BLE001
        request_stats.record_api_call("stock_zh_a_hist_tx", False, provider="腾讯")
        log.warning("Tencent index_hist fallback failed for %s: %s", symbol, e)
        return None
    if df is None or df.empty:
        return None
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    for col in ("open", "close", "high", "low", "amount"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def price_hist(symbol: str, years: int = 5) -> pd.DataFrame | None:
    """Daily OHLCV (qfq adjusted) for the last `years` years."""
    cached = cache.get("price_hist", symbol, str(years))
    if cached is not None:
        cached, repaired = _repair_cached_tx_price_hist(cached)
        if repaired and cached is not None:
            cache.put("price_hist", cached, symbol, str(years))
        return cached
    end = dt.date.today()
    start = end.replace(year=end.year - years)
    df = _retry(
        ak.stock_zh_a_hist,
        symbol=symbol,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    if df is None or df.empty:
        tx_df = _tx_price_hist(symbol, start, end, adjust="qfq")
        if tx_df is not None and not tx_df.empty:
            log.warning("price_hist fetch failed from Eastmoney for %s; using Tencent fallback", symbol)
            cache.put("price_hist", tx_df, symbol, str(years))
            return tx_df
        stale = cache.get_stale("price_hist", symbol, str(years))
        if stale is not None and not stale.empty:
            log.warning(
                "price_hist fetch failed for %s years=%s; using stale cached data",
                symbol,
                years,
            )
            return stale
        cache.put_empty("price_hist", symbol, str(years), ttl_hours=_FAILED_FETCH_TTL_HOURS)
        return None
    rename = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
    }
    df = df.rename(columns=rename)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    cache.put("price_hist", df, symbol, str(years))
    return df


def index_hist(symbol: str = "000300", years: int = 1) -> pd.DataFrame | None:
    """Daily index OHLCV, used for 60-day beta in the T factor."""
    cached = cache.get("price_hist", "index", symbol, str(years))
    if cached is not None:
        return cached
    end = dt.date.today()
    start = end.replace(year=end.year - years)
    df = _retry(
        ak.index_zh_a_hist,
        symbol=symbol,
        period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        attempts=1,
    )
    if df is None or df.empty:
        tx_df = _tx_index_hist(symbol, start, end)
        if tx_df is not None and not tx_df.empty:
            log.warning("index_hist fetch failed from Eastmoney for %s; using Tencent fallback", symbol)
            cache.put("price_hist", tx_df, "index", symbol, str(years))
            return tx_df
        stale = cache.get_stale("price_hist", "index", symbol, str(years))
        if stale is not None:
            return stale
        cache.put_empty("price_hist", "index", symbol, str(years), ttl_hours=_FAILED_FETCH_TTL_HOURS)
        return None
    rename = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
    }
    df = df.rename(columns=rename)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    for col in ("open", "close", "high", "low", "volume", "amount", "amplitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    cache.put("price_hist", df, "index", symbol, str(years))
    return df


# ---- Daily indicator (PE/PB/total mv) ------------------------------------

_VALUE_EM_RENAME = {
    "数据日期": "trade_date",
    "当日收盘价": "close",
    "当日涨跌幅": "change_pct",
    "总市值": "total_mv",
    "流通市值": "float_mv",
    "总股本": "total_shares",
    "流通股本": "float_shares",
    "PE(TTM)": "pe_ttm",
    "PE(静)": "pe",
    "市净率": "pb",
    "PEG值": "peg",
    "市现率": "pcf",
    "市销率": "ps",
}

_VALUE_EM_NUMERIC_COLUMNS = [
    "close",
    "change_pct",
    "total_mv",
    "float_mv",
    "total_shares",
    "float_shares",
    "pe_ttm",
    "pe",
    "pb",
    "peg",
    "pcf",
    "ps",
]


def _normalise_indicator_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    df = _fix_mojibake_df(df)
    df = df.rename(columns=_VALUE_EM_RENAME)
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date.astype(str)
    for col in _VALUE_EM_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def indicator_hist(symbol: str) -> pd.DataFrame | None:
    """Daily PE TTM / static PE / PB / market-cap series.

    Backed by东方财富's valuation analysis endpoint (`ak.stock_value_em`),
    returning ~5000 trading days with renamed English column names so the
    factor / interval modules can reference them directly.
    """
    cached = cache.get("indicator", symbol)
    if cached is not None:
        return _normalise_indicator_df(cached)
    df = _retry(ak.stock_value_em, symbol=symbol)
    if df is None or df.empty:
        stale = cache.get_stale("indicator", symbol)
        if stale is not None and not stale.empty:
            log.warning("indicator_hist fetch failed for %s; using stale cached data", symbol)
            return _normalise_indicator_df(stale)
        cache.put_empty("indicator", symbol, ttl_hours=_FAILED_FETCH_TTL_HOURS)
        return None
    df = _normalise_indicator_df(df)
    cache.put("indicator", df, symbol)
    return df


_SPOT_RENAME = {
    "代码": "symbol",
    "名称": "name",
    "最新价": "last",
    "成交额": "amount",
    "成交量": "volume",
    "振幅": "amplitude",
    "换手率": "turnover_rate",
    "总市值": "total_mv",
    "流通市值": "float_mv",
}

_SPOT_NUMERIC_COLUMNS = [
    "last",
    "amount",
    "volume",
    "amplitude",
    "turnover_rate",
    "total_mv",
    "float_mv",
]


def _normalise_spot_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    df = _fix_mojibake_df(df)
    df = df.rename(columns=_SPOT_RENAME)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    for col in _SPOT_NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def is_complete_realtime_snapshot(df: pd.DataFrame | None) -> bool:
    return df is not None and not df.empty and len(df) >= _MIN_FULL_MARKET_SNAPSHOT_ROWS


def _tx_realtime_snapshot() -> pd.DataFrame | None:
    try:
        from akshare.stock.stock_zh_a_tx import stock_zh_a_spot_tx
    except Exception:  # noqa: BLE001
        return None

    try:
        df = stock_zh_a_spot_tx()
        request_stats.record_api_call("stock_zh_a_spot_tx", _result_ok(df), provider="腾讯")
    except Exception:  # noqa: BLE001
        request_stats.record_api_call("stock_zh_a_spot_tx", False, provider="腾讯")
        return None
    if df is None or df.empty:
        return None

    out = pd.DataFrame(
        {
            "代码": df.get("code", pd.Series(dtype=object)).astype(str).str[-6:],
            "名称": df.get("name"),
            "最新价": df.get("zxj"),
            "换手率": df.get("hsl"),
            "总市值": pd.to_numeric(df.get("zsz"), errors="coerce") * 1e8,
            "流通市值": pd.to_numeric(df.get("ltsz"), errors="coerce") * 1e8,
            "成交额": pd.Series(pd.NA, index=df.index, dtype="object"),
        }
    )
    out = _normalise_spot_df(out)
    if not is_complete_realtime_snapshot(out):
        log.warning(
            "Tencent spot fallback returned only %s rows; treating it as incomplete for full-market scan",
            0 if out is None else len(out),
        )
        return None
    return out


# ---- Financial abstract ---------------------------------------------------


def financial_abstract(symbol: str) -> pd.DataFrame | None:
    """Quarterly financial summary: wide table, indicators as rows."""
    cached = cache.get("financial", symbol)
    if cached is not None:
        return _fix_mojibake_df(cached)
    df = _retry(ak.stock_financial_abstract, symbol=symbol)
    if df is None or df.empty:
        df = _retry(ak.stock_financial_abstract_ths, symbol=symbol, indicator="按报告期")
    if df is None or df.empty:
        stale = cache.get_stale("financial", symbol)
        if stale is not None and not stale.empty:
            log.warning("financial_abstract fetch failed for %s; using stale cached data", symbol)
            return _fix_mojibake_df(stale)
        cache.put_empty("financial", symbol, ttl_hours=_FAILED_FETCH_TTL_HOURS)
        return None
    df = _fix_mojibake_df(df)
    cache.put("financial", df, symbol)
    return df


# ---- Balance sheet --------------------------------------------------------


def balance_sheet(symbol: str) -> pd.DataFrame | None:
    """Balance sheet via Eastmoney, wide table (rows=periods newest-first, cols=items).

    Column names are English (GOODWILL, MONETARYFUNDS, SHORT_LOAN,
    NONCURRENT_LIAB_1YEAR, TOTAL_PARENT_EQUITY, …).
    """
    key = symbol + "_bs"
    cached = cache.get("financial", key)
    if cached is not None:
        return cached
    prefix = "SH" if symbol.startswith("6") else ("BJ" if symbol.startswith(("8", "4")) else "SZ")
    df = _retry(ak.stock_balance_sheet_by_yearly_em, symbol=f"{prefix}{symbol}")
    if df is None or df.empty:
        stale = cache.get_stale("financial", key)
        if stale is not None:
            return stale
        cache.put_empty("financial", key, ttl_hours=6)
        return None
    if "REPORT_DATE" in df.columns:
        df = df.sort_values("REPORT_DATE", ascending=False).reset_index(drop=True)
    cache.put("financial", df, key)
    return df


# ---- Dividend history -----------------------------------------------------


def dividend_hist(symbol: str) -> pd.DataFrame | None:
    cached = cache.get("dividend", symbol)
    if cached is not None:
        return _fix_mojibake_df(cached)
    df = _retry(ak.stock_dividend_cninfo, symbol=symbol)
    if df is None or df.empty:
        stale = cache.get_stale("dividend", symbol)
        if stale is not None and not stale.empty:
            log.warning("dividend_hist fetch failed for %s; using stale cached data", symbol)
            return _fix_mojibake_df(stale)
        cache.put_empty("dividend", symbol, ttl_hours=_FAILED_FETCH_TTL_HOURS)
        return None
    df = _fix_mojibake_df(df)
    cache.put("dividend", df, symbol)
    return df


# ---- Convenience: spot snapshot for a list of codes -----------------------


def realtime_snapshot() -> pd.DataFrame | None:
    """Whole-market spot quote, used for current price lookups."""
    cached = cache.get("price_hist", "realtime_snapshot")
    if cached is not None:
        cached = _normalise_spot_df(cached)
        if is_complete_realtime_snapshot(cached):
            return cached
        log.warning(
            "cached realtime snapshot has only %s rows; deleting incomplete cache",
            0 if cached is None else len(cached),
        )
        cache.delete("price_hist", "realtime_snapshot")
    df = _retry(ak.stock_zh_a_spot_em)
    df = _normalise_spot_df(df)
    if df is not None and not df.empty and not is_complete_realtime_snapshot(df):
        log.warning("Eastmoney realtime snapshot returned only %s rows; treating it as failed", len(df))
        df = None
    if df is None or df.empty:
        tx_df = _tx_realtime_snapshot()
        if tx_df is not None and not tx_df.empty:
            log.warning("realtime_snapshot fetch failed from Eastmoney; using Tencent spot fallback")
            cache.put("price_hist", tx_df, "realtime_snapshot")
            return tx_df
        stale = cache.get_stale("price_hist", "realtime_snapshot")
        stale = _normalise_spot_df(stale)
        if is_complete_realtime_snapshot(stale):
            log.warning("realtime_snapshot fetch failed; using stale cached snapshot")
            return stale
        return None
    cache.put("price_hist", df, "realtime_snapshot")
    return df


def refresh_symbol(symbol: str) -> dict[str, bool]:
    """Force-refresh one symbol's local AKShare caches.

    Existing cache entries are restored when a refresh attempt fails, so a
    temporary network error does not make the stock detail page worse.
    """
    symbol = symbol.zfill(6)
    status: dict[str, bool] = {}

    def refresh_frame(label: str, kind: str, key_parts: tuple[str, ...], fn) -> None:
        old = cache.get(kind, *key_parts)
        cache.delete(kind, *key_parts)
        fresh = fn()
        ok = fresh is not None and not fresh.empty
        if not ok and old is not None and not old.empty:
            cache.put(kind, old, *key_parts)
        status[label] = bool(ok)

    def refresh_json(label: str, kind: str, key_parts: tuple[str, ...], fn) -> None:
        old = cache.get_json(kind, *key_parts)
        cache.delete_json(kind, *key_parts)
        fresh = fn()
        ok = fresh is not None
        if not ok and old is not None:
            cache.put_json(kind, old, *key_parts)
        status[label] = bool(ok)

    refresh_frame("price_1y", "price_hist", (symbol, "1"), lambda: price_hist(symbol, years=1))
    refresh_frame("price_5y", "price_hist", (symbol, "5"), lambda: price_hist(symbol, years=5))
    refresh_frame("indicator", "indicator", (symbol,), lambda: indicator_hist(symbol))
    refresh_frame("financial", "financial", (symbol,), lambda: financial_abstract(symbol))
    refresh_frame("dividend", "dividend", (symbol,), lambda: dividend_hist(symbol))
    refresh_json("info", "info", (symbol,), lambda: individual_info(symbol))
    return status

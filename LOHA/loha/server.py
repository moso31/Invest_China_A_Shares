"""FastAPI application: page views + JSON API.

Pages are server-rendered with Jinja2 (light JS only for the screener
progress poll & pool actions). The same /api routes back the HTML pages
and any external client.
"""

from __future__ import annotations

import datetime
import math
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from loha import config, factors, interval, pool, request_stats, scoring, screener, source
from loha.eastmoney import EASTMONEY_NID18_ENV_VAR, load_persisted_nid18, nid18_configured, set_nid18_for_process

PKG_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(PKG_DIR / "templates"))

app = FastAPI(title="LOHA", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(PKG_DIR / "static")), name="static")


@app.on_event("startup")
def _startup_cache_maintenance() -> None:
    load_persisted_nid18()
    screener.invalidate_stale_score_caches()


def _lookup_name(symbol: str, fallback: str | None = None) -> str:
    name_lookup = source.universe_all()
    if name_lookup is not None and not name_lookup.empty:
        m = name_lookup[name_lookup["symbol"] == symbol]
        if not m.empty:
            return str(m.iloc[0]["name"])
    return fallback or symbol


def _score_symbol(symbol: str, name: str | None = None) -> scoring.StockScore:
    name = name or _lookup_name(symbol)
    factor_values = factors.all_factors(symbol, name)
    return scoring.composite(symbol, name, {k: factor_values[k] for k in ("Q", "D", "V", "T", "R")})


def _normalise_score_row(row: dict | None) -> dict:
    if not row:
        return {}
    out = dict(row)
    legacy_flags = list(out.get("flags") or [])
    risk_flags = list(out.get("risk_flags") or [])
    good_flags = list(out.get("good_flags") or [])
    if not risk_flags and not good_flags and legacy_flags:
        risk_flags, good_flags = scoring.split_signals(legacy_flags)
    elif not risk_flags:
        risk_flags = legacy_flags
    risk_flags = scoring.filter_active_signals(risk_flags)
    good_flags = scoring.dedupe_signals(good_flags + scoring.derive_good_flags(out))
    out["risk_flags"] = risk_flags
    out["good_flags"] = good_flags
    out["flags"] = risk_flags
    return out


def _json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _action_summary(score_row: dict, interval_row: dict | None) -> dict:
    score_row = _normalise_score_row(score_row)
    flags = score_row.get("risk_flags") or []
    score = float(score_row.get("score") or 0)
    risk_penalty = float(score_row.get("R") or 0)
    interval_action = (interval_row or {}).get("action_hint") or "数据不足"

    if risk_penalty >= 50 or score < 50:
        action = "停止加仓"
        reason = "评分或风险扣分不满足加仓条件"
    elif interval_row and (
        interval_row.get("band") == "expensive"
        or (
            interval_row.get("last_price") is not None
            and interval_row.get("sell_high") is not None
            and interval_row["last_price"] >= interval_row["sell_high"]
        )
    ):
        action = "减仓或只减不加"
        reason = "价格已接近卖出触发位或估值偏贵"
    elif interval_row and (
        interval_row.get("last_price") is not None
        and interval_row.get("buy_low") is not None
        and interval_row["last_price"] <= interval_row["buy_low"]
    ):
        action = "可分批加仓"
        reason = "价格进入估值/做T买入触发区"
    else:
        action = interval_action
        reason = "未触发强买入或强卖出条件"

    return {
        "action": action,
        "reason": reason,
        "current_flags": flags,
    }


def _confirmation_reasons(score: float, flags: list[str]) -> list[str]:
    reasons = list(flags)
    if score < 50:
        reasons.insert(0, f"综合评分 {score:.1f} 低于 50")
    return reasons


LARGE_INFO_KEYS = {"总股本", "流通股", "总市值", "流通市值"}


def _format_wan(value) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(num) >= 10_000_000:
        return f"{num / 10_000:.1f}万"
    if num.is_integer():
        return str(int(num))
    return f"{num:.2f}".rstrip("0").rstrip(".")


def _display_info(info: dict) -> dict:
    out = {}
    for key, value in (info or {}).items():
        out[key] = _format_wan(value) if key in LARGE_INFO_KEYS else value
    return out


def _build_dividends(score_row: dict) -> dict:
    d_comp = ((score_row.get("components") or {}).get("D")) or {}
    div_year_yields: dict = d_comp.get("div_year_yields") or {}
    div_year_sources: dict = d_comp.get("div_year_sources") or {}
    div_by_year: dict = d_comp.get("div_by_year") or {}

    current_year = datetime.date.today().year
    # Show 5 lookback years (same window as scoring), newest first
    target_years = [current_year - 1 - i for i in range(config.DIV_LOOKBACK_YEARS)]
    # Extend with any additional historical years from actual data
    extra = sorted(
        (int(y) for y in div_by_year if int(y) not in target_years),
        reverse=True,
    )
    all_years = target_years + [y for y in extra if y < target_years[-1]]

    rows = []
    for year in all_years:
        year_str = str(year)
        div_per_share = float(div_by_year[year_str]) if year_str in div_by_year else 0.0
        score_yield = float(div_year_yields[year_str]) if year_str in div_year_yields else None
        source = div_year_sources.get(year_str, "actual_dividend" if div_per_share > 0 else "listed_no_dividend")
        yield_pct = score_yield if source == "actual_dividend" else None
        rows.append(
            {
                "year": year,
                "div_per_share": round(div_per_share, 4),
                "yield_pct": round(yield_pct, 2) if yield_pct is not None else None,
                "score_yield_pct": round(score_yield, 2) if score_yield is not None else None,
                "yield_source": source,
                "above_4pct": score_yield is not None and score_yield >= 4.0,
            }
        )

    return {
        "rows": rows,
        "weighted_dividend_yield": d_comp.get("weighted_dividend_yield"),
        "weighted_dividend_score": d_comp.get("weighted_dividend_score"),
    }


def _stock_payload(symbol: str) -> dict:
    symbol = symbol.zfill(6)
    name = _lookup_name(symbol)
    s = _score_symbol(symbol, name)
    iv = interval.compute(symbol)
    info = source.individual_info(symbol) or {}
    info_display = _display_info(info)
    score_row = s.to_dict()
    score_row = _normalise_score_row(score_row)
    interval_row = iv.to_dict()
    return {
        "symbol": symbol,
        "name": name,
        "info": info_display,
        "info_display": info_display,
        "info_raw": info,
        "score": score_row,
        "interval": interval_row,
        "action": _action_summary(score_row, interval_row),
        "dividends": _build_dividends(score_row),
        "in_pool": pool.has(symbol),
    }


def _pool_plan(entry: pool.PoolEntry, interval_row: dict | None) -> dict:
    interval_row = interval_row or {}
    current_price = interval_row.get("last_price")
    entry_price = entry.entry_price or current_price or 0
    atr = interval_row.get("atr")
    if not atr and entry_price:
        atr = entry_price * 0.02
    sell_fly_line = entry_price + 2 * atr if entry_price and atr else None
    stop_loss_line = max(0, entry_price - 2 * atr) if entry_price and atr else None
    lot_size = interval_row.get("lot_size") or config.LOT_SIZE_HIGH_PRICE
    lot_value = current_price * lot_size if current_price else None
    grid_budget = config.PER_STOCK_BUDGET_CNY * config.TACTICAL_POSITION_RATIO
    core_budget = config.PER_STOCK_BUDGET_CNY * config.CORE_POSITION_RATIO
    return {
        "entry_price": round(entry_price, 3) if entry_price else None,
        "sell_fly_line": round(sell_fly_line, 3) if sell_fly_line else None,
        "stop_loss_line": round(stop_loss_line, 3) if stop_loss_line else None,
        "core_budget": round(core_budget, 2),
        "grid_budget": round(grid_budget, 2),
        "core_ratio": config.CORE_POSITION_RATIO,
        "grid_ratio": config.TACTICAL_POSITION_RATIO,
        "grid_lots": int(grid_budget // lot_value) if lot_value else 0,
        "core_lots": int(core_budget // lot_value) if lot_value else 0,
        "note": "25% 底仓长期持有；75% 网格仓围绕加入价和 ATR 做高抛低吸。卖飞线以上不追高，止损线以下复核风险或停止加仓。",
    }


# ---- Page views ----------------------------------------------------------


def _eastmoney_notice() -> dict | None:
    if nid18_configured():
        return None
    return {
        "title": "未指定 Eastmoney nid18",
        "message": "东方财富数据可能降级；扫描流程会继续，并优先使用本地缓存、腾讯/同花顺等可用兜底数据。",
    }


def _eastmoney_settings_context() -> dict:
    configured = nid18_configured()
    return {
        "configured": configured,
        "env_var": EASTMONEY_NID18_ENV_VAR,
        "title": "Eastmoney nid18 已设置" if configured else "未指定 Eastmoney nid18",
        "message": (
            "可粘贴新值更新当前服务进程。"
            if configured
            else "东方财富数据可能降级；扫描流程会继续，并优先使用本地缓存、腾讯/同花顺等可用兜底数据。"
        ),
    }


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "active": "home",
            "eastmoney_notice": _eastmoney_notice(),
            "eastmoney_settings": _eastmoney_settings_context(),
            "config": {
                "universe_mode": config.UNIVERSE_MODE,
                "universe_size": (
                    len(config.DEFAULT_UNIVERSE) if config.UNIVERSE_MODE == "curated" else "all A-shares"
                ),
                "weights": {
                    "Q": config.W_QUALITY,
                    "D": config.W_DIVIDEND,
                    "V": config.W_VALUATION,
                    "T": config.W_TRADING,
                    "R": config.W_RISK,
                },
                "budget": config.PER_STOCK_BUDGET_CNY,
                "core_ratio": config.CORE_POSITION_RATIO,
                "grid_ratio": config.TACTICAL_POSITION_RATIO,
            },
        },
    )


@app.get("/screener", response_class=HTMLResponse)
def screener_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request,
        "screener.html",
        {
            "active": "screener",
            "eastmoney_notice": _eastmoney_notice(),
            "eastmoney_settings": _eastmoney_settings_context(),
        },
    )


@app.get("/pool", response_class=HTMLResponse)
def pool_page(request: Request):
    return TEMPLATES.TemplateResponse(request, "pool.html", {"active": "pool"})


@app.get("/debug", response_class=HTMLResponse)
def debug_page(request: Request):
    return TEMPLATES.TemplateResponse(request, "debug.html", {"active": "debug"})


@app.get("/stock/{symbol}", response_class=HTMLResponse)
def stock_detail(symbol: str, request: Request):
    return TEMPLATES.TemplateResponse(request, "detail.html", {"active": "", "symbol": symbol})


# ---- JSON API ------------------------------------------------------------

_screen_lock = threading.Lock()
_screen_thread: threading.Thread | None = None


@app.post("/api/screener/run")
def api_screener_run(top_n: int | None = None, mode: str | None = None):
    global _screen_thread
    with _screen_lock:
        prog = screener.progress()
        if prog.state == "running":
            return {"ok": False, "msg": "screening already in progress", "progress": prog.__dict__}
        _screen_thread = threading.Thread(
            target=screener.run, kwargs={"top_n": top_n, "mode": mode, "force_refresh": False}, daemon=True
        )
        _screen_thread.start()
    return {"ok": True}


@app.post("/api/screener/full-scan/run")
def api_full_scan_run():
    global _screen_thread
    with _screen_lock:
        prog = screener.progress()
        if prog.state == "running":
            return {"ok": False, "msg": "screening already in progress", "progress": prog.__dict__}
        _screen_thread = threading.Thread(
            target=screener.run,
            kwargs={"mode": "all", "force_refresh": True, "refresh_mode": screener.FULL_REFRESH_COMPLETE},
            daemon=True,
        )
        _screen_thread.start()
    return {"ok": True, "refresh_mode": screener.FULL_REFRESH_COMPLETE}


@app.post("/api/debug/recompute-scores/run")
def api_debug_recompute_scores_run():
    global _screen_thread
    with _screen_lock:
        prog = screener.progress()
        if prog.state == "running":
            return {"ok": False, "msg": "screening already in progress", "progress": prog.__dict__}
        _screen_thread = threading.Thread(
            target=screener.recompute_cached_full_scan,
            kwargs={"cache_only": True},
            daemon=True,
        )
        _screen_thread.start()
    return {"ok": True}


@app.get("/api/screener/progress")
def api_screener_progress():
    p = screener.progress()
    elapsed_s = round(time.time() - p.started_ts, 1) if p.started_ts else 0
    stats = request_stats.snapshot()
    current_full_scan = None
    if p.mode == "all" and p.state in {"running", "done", "error"}:
        status = "扫描中"
        if p.state == "done":
            status = "扫描完毕"
        elif p.state == "error":
            status = "扫描失败"
        current_full_scan = {
            "status": status,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "refresh_mode": p.refresh_mode,
            "refresh_mode_label": p.refresh_mode_label,
            "duration_s": elapsed_s,
            "source_total": p.source_total,
            "prefiltered_total": p.prefiltered_total,
            "total": p.total,
            "done": p.done,
            "accepted": p.accepted,
            "rejected": p.rejected,
            "failed": p.failed,
            "current_symbol": p.current_symbol,
            "request_stats": stats,
        }
    return {
        "state": p.state,
        "mode": p.mode,
        "refresh_mode": p.refresh_mode,
        "refresh_mode_label": p.refresh_mode_label,
        "cache_used": p.cache_used,
        "source_total": p.source_total,
        "prefiltered_total": p.prefiltered_total,
        "total": p.total,
        "done": p.done,
        "accepted": p.accepted,
        "rejected": p.rejected,
        "failed": p.failed,
        "current_symbol": p.current_symbol,
        "error": p.error,
        "elapsed_s": elapsed_s,
        "request_stats": stats,
        "current_full_scan": current_full_scan,
    }


@app.get("/api/screener/full-scan/meta")
def api_full_scan_meta():
    return {"meta": screener.full_scan_meta()}


@app.get("/api/screener/results")
def api_screener_results(limit: int = 50):
    rows = screener.last_results() or []
    rows = [_json_safe(_normalise_score_row(r)) for r in rows]
    return {"results": rows[:limit], "total": len(rows)}


class EastmoneyNid18Request(BaseModel):
    nid18: str = ""


def _validate_nid18(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if len(value) > 256 or any(ch.isspace() or ch in {';', ',', '"', "'"} for ch in value):
        raise HTTPException(status_code=400, detail="nid18 格式不正确")
    return value


@app.get("/api/settings/eastmoney-nid18")
def api_get_eastmoney_nid18():
    return _eastmoney_settings_context()


@app.post("/api/settings/eastmoney-nid18")
def api_set_eastmoney_nid18(req: EastmoneyNid18Request):
    set_nid18_for_process(_validate_nid18(req.nid18))
    return {"ok": True, **_eastmoney_settings_context()}


@app.get("/api/pool")
def api_pool_list():
    entries = pool.list_all()
    enriched = []
    for e in entries:
        score_row = None
        try:
            iv = interval.compute(e.symbol)
        except Exception:
            iv = None
        try:
            score_row = _normalise_score_row(_score_symbol(e.symbol, e.name).to_dict())
        except Exception:
            score_row = None
        interval_row = iv.to_dict() if iv else None
        action = (
            _action_summary(score_row, interval_row)
            if score_row
            else {
                "action": "数据不足",
                "reason": "无法刷新当前评分",
                "current_flags": [],
            }
        )
        enriched.append(
            {
                **e.to_dict(),
                "interval": interval_row,
                "current_score": score_row,
                "action": action,
                "pool_plan": _pool_plan(e, interval_row),
            }
        )
    return {"pool": enriched}


class AddRequest(BaseModel):
    symbol: str
    name: str | None = None
    notes: str = ""
    confirm_low_score: bool = False


@app.post("/api/pool/add")
def api_pool_add(req: AddRequest):
    symbol = req.symbol.zfill(6)
    last = screener.last_results() or []
    rec = next((r for r in last if r["symbol"] == symbol), None)
    rec = _normalise_score_row(rec) if rec else None
    name = req.name or (rec["name"] if rec else _lookup_name(symbol))
    if rec:
        score = float(rec["score"])
        flags = list(rec["risk_flags"])
    else:
        try:
            calculated = _normalise_score_row(_score_symbol(symbol, name).to_dict())
            name = calculated["name"]
            score = float(calculated["score"])
            flags = list(calculated["risk_flags"])
        except Exception as e:  # noqa: BLE001
            score = 0.0
            flags = [f"无法计算当前评分：{e}"]

    reasons = _confirmation_reasons(score, flags)

    risk_high = bool(reasons)
    if risk_high and not req.confirm_low_score:
        return JSONResponse(
            status_code=409,
            content={
                "ok": False,
                "needs_confirm": True,
                "reasons": reasons,
                "score": score,
            },
        )

    entry_price = 0.0
    try:
        iv = interval.compute(symbol)
        entry_price = float(iv.last_price or 0)
    except Exception:
        entry_price = 0.0
    added = pool.add(symbol, name, score, flags, req.notes, entry_price=entry_price)
    return {"ok": added, "already_in_pool": not added}


class RemoveRequest(BaseModel):
    symbol: str


@app.post("/api/pool/remove")
def api_pool_remove(req: RemoveRequest):
    return {"ok": pool.remove(req.symbol.zfill(6))}


@app.get("/api/stock/{symbol}")
def api_stock_detail(symbol: str):
    try:
        return _stock_payload(symbol)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stock/{symbol}/refresh")
def api_stock_refresh(symbol: str):
    symbol = symbol.zfill(6)
    refresh_status = source.refresh_symbol(symbol)
    try:
        payload = _stock_payload(symbol)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))
    return {"ok": any(refresh_status.values()), "refresh": refresh_status, "stock": payload}

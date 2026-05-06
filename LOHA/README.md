# LOHA

A-share long-term stock screening and trading interval planner. Implements the four-layer system from `../Strategy.md`: data, scoring, risk sentinel, trading plan.

## Install

From the repository root:

```bash
.venv\Scripts\activate           # or: source .venv/Scripts/activate
pip install -e ./akshare         # vendored AKShare
pip install -e ./LOHA            # this project
```

## Run

```bash
python -m loha
```

Then open http://127.0.0.1:8000 .

## Eastmoney nid18

Eastmoney's `push2` / `push2his` APIs may close direct Python requests unless the
request carries a valid anonymous `nid18` cookie. LOHA now supports injecting
that cookie through the `LOHA_EASTMONEY_NID18` environment variable.

PowerShell:

```powershell
$env:LOHA_EASTMONEY_NID18 = "<your nid18>"
python -m loha
```

You can also start from the repository root with:

```bat
start_server_with_nid18.bat
```

The batch file prompts for `nid18`, stores it in `LOHA/data/eastmoney_nid18.txt`,
and sets it for that server process. After the web app is already running, the
overview and screener pages also provide a local-only `nid18` update control;
updates there also refresh the same local file. The file is ignored by Git.

`cmd.exe`:

```bat
set LOHA_EASTMONEY_NID18=<your nid18>
python -m loha
```

How to obtain `nid18`:

1. Open `https://quote.eastmoney.com/` in a normal browser window.
2. Press `F12` to open Developer Tools.
3. Open the `Application` / `Storage` panel, then `Cookies`.
4. Select the Eastmoney site entry such as `https://quote.eastmoney.com`.
5. Copy the value of the `nid18` cookie.

Notes:

- `nid18` is an anonymous anti-bot cookie; the current LOHA flow does not need an Eastmoney account login.
- LOHA uses a browser-like `curl_cffi` transport for Eastmoney requests when available, because `requests` alone may be closed before an HTTP response is returned.
- A current `nid18` may be enough to restore some Eastmoney calls used by deep scoring, but this is not guaranteed for every session or endpoint.
- Even when the browser shows a `nid18` value, Eastmoney may still reject some programmatic requests; LOHA will cool down repeatedly failing endpoint families instead of hammering them during a full scan.
- If Eastmoney's full-market spot snapshot is still blocked, LOHA now falls back to Tencent's spot snapshot automatically for universe building.
- If Eastmoney rotates or expires the cookie, repeat the steps above and update the environment variable.
- Do not commit your real `nid18` value into tracked files.

Optional controls:

- `LOHA_EASTMONEY_COOLDOWN_SECONDS` controls how long a failing Eastmoney endpoint family is paused; default is `300`.
- `LOHA_EASTMONEY_FAILURE_THRESHOLD` controls how many consecutive failures open the cooldown; default is `2`.
- `LOHA_EASTMONEY_TRANSPORT=requests` disables the `curl_cffi` Eastmoney transport for comparison/debugging.
- `LOHA_IGNORE_PROXY_ENV=0` lets `requests` read system proxy settings again; by default LOHA ignores proxy environment/settings inside the app process to avoid accidental `ProxyError` noise.

## What it does

1. **Machine screening** — pulls fundamentals + price history for a configurable universe, applies hard gates (ST / liquidity), computes the estimate.md strategy fit `0.45D + 0.35V + 0.20T`, ranks candidates. D is based on 5-year weighted yield, payout-ratio risk filtering, cash-flow coverage, and ROE stability; Q and R remain review/risk signals.
2. **Manual review** — top-N table where you add stocks to the local pool. Adding a low-scoring or risk-flagged stock prompts a confirmation with reasons.
3. **Local pool** — SQLite-backed list of holdings. Each row shows the current estimate.md valuation band, T-band reference interval, risk sentinel status, and recommended action (add / hold / trim).

## Universe

The screener page provides two modes:

- `精选池（快速）`: the dividend-friendly industries from Strategy.md (banks, utilities, coal, telecom, highways, consumer leaders).
- `全 A 股（流动性预筛）`: reuses the latest successful local full-market scan. If no local scan exists yet, run `强制更新全 A 股数据` first.

Use `强制更新全 A 股数据` on the screener page to ignore local data caches, fetch the full A-share snapshot from AKShare, apply cheap hard gates first (ST / delisting / new-share marker / no price / turnover below ¥100M), and refresh the full-market scan. The UI asks for confirmation because every remaining stock needs price, valuation, financial and dividend data. The "last successful full scan" timestamp is only updated after a successful run.

## Cache

Raw AKShare responses are cached as pickle/JSON files under `data/cache/`. Delete the folder to force a refetch.

## Layout

```
loha/
  config.py        constants, weights, thresholds, paths
  cache.py         pickle / JSON-backed local cache
  source.py        AKShare wrappers (one function per data kind)
  valuation.py     estimate.md bucket valuation engine
  factors.py       Q / D / V / T / R factor calculators
  scoring.py       composite score + ranking
  interval.py      dual-anchor trading interval
  screener.py      screening pipeline
  pool.py          SQLite local-pool CRUD
  server.py        FastAPI app, API routes, page views
  templates/       Jinja2 HTML
  static/          CSS
```

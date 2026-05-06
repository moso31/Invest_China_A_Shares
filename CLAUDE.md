# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a China A-shares long-term investment system using AKShare as the data source. The system scores stocks quantitatively, generates trading intervals, and manages a local stock pool. The application code lives in `LOHA/`; `akshare/` is a vendored copy of the AKShare library.

## Environment Setup

```bash
# Activate virtual environment (Windows)
.venv/Scripts/activate

# Install akshare in editable mode from the local copy
pip install -e akshare/

# Install dev tools
pip install -e "akshare/[dev]"
```

## Common Commands

```bash
# Lint and format (run from akshare/)
cd akshare && ruff check . --fix
cd akshare && ruff format .

# Run tests (from akshare/)
cd akshare && python -m pytest tests/

# Run a single test
cd akshare && python -m pytest tests/test_func.py::test_path_func
```

## AKShare Library Architecture (`akshare/akshare/`)

AKShare is a financial data spider library. All public functions are re-exported through `__init__.py`. Key modules relevant to this project:

| Module | Purpose |
|---|---|
| `stock/` | Stock quotes, dividends, ownership (东方财富, 新浪, etc.) |
| `stock_feature/` | Historical prices, PE/PB ratios, ATR, margin data, valuation |
| `stock_fundamental/` | Financial statements, IPO data, earnings forecasts |
| `economic/` | Macro data (China NBS, Fed, etc.) |
| `index/` | Index constituents and history |

**Proxy config** — use the singleton `AkshareConfig` before making requests:
```python
import akshare as ak
ak.set_proxies({"http": "...", "https": "..."})
```

**HTTP requests** — `akshare/utils/request.py` and `akshare/request.py` provide retry-enabled wrappers around `requests` and `curl_cffi`.

## Investment System Design (to be built in `LOHA/`)

The four-layer system described in `Strategy.md`:

1. **Data layer** — fetch via AKShare; cache locally to avoid repeated API calls
2. **Scoring layer** — estimate.md strategy fit: `Score = 0.45D + 0.35V + 0.20T`
   - Q = enterprise quality review signal (ROE/ROIC, margins, cash flow, leverage)
   - D = dividend_score (5-year weighted yield, payout ratio risk filter, cash flow coverage, ROE stability)
   - V = value_score (estimate.md industry bucket valuation and safety margin)
   - T = t_score (60-day annualized volatility, 20-day turnover, beta, range-bound behavior)
   - R = risk review signal (revenue decline, cash flow weakness, pledge ratio, etc.)
3. **Risk sentinel layer** — hard stops on adding to position (consecutive revenue decline, low OCF/NI ratio, abnormal AR/inventory growth)
4. **Trading interval layer** — estimate.md T-band:
   - Valuation anchor: fair_mid from the industry bucket valuation
   - Volatility anchor: 60-day Bollinger-style band
   - Final range: clip the volatility band by `fair_mid * 0.85` and `fair_mid * 1.15`

**Position model**: 25% core position (hold for dividends + growth), 75% tactical position (swing within the trading interval). Per-stock budget target ≈ ¥100,000; trade unit = 200 shares if price < ¥5, else 100 shares.

## AKShare Key APIs for This Project

```python
import akshare as ak

# Stock list and basic info
ak.stock_info_a_code_name()           # all A-share codes + names
ak.stock_zh_a_hist(symbol, period)    # historical OHLCV

# Fundamentals
ak.stock_financial_abstract_ths()     # financial summary (THS)
ak.stock_a_indicator_lg(symbol)       # ROE, PE, PB daily series

# Dividends
ak.stock_dividend_cninfo(symbol)      # dividend history

# Valuation / features
ak.stock_a_pe_and_pb()               # market-wide PE/PB
ak.stock_ttm_lyr()                   # TTM/LYR P/E table
```

## Code Conventions

- Line length: 88 characters (ruff default, matches Black)
- Quotes: double quotes
- Pre-commit hooks enforce ruff lint + format and conventional commit messages
- Commits must follow Conventional Commits format (e.g. `feat:`, `fix:`, `chore:`)

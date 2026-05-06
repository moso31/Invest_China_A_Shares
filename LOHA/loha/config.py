"""Central configuration: paths, weights, thresholds.

All numeric thresholds map back to Strategy.md / plan.md so future tuning is
done in one place.
"""

from __future__ import annotations

from pathlib import Path

# ---- Paths ----------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent  # LOHA/
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DB_PATH = DATA_DIR / "loha.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---- Web server -----------------------------------------------------------

HOST = "127.0.0.1"
PORT = 8000

# ---- Universe -------------------------------------------------------------
# Strategy.md mentions banks / utilities / coal / electricity / telecom /
# highways / consumer leaders as the dividend-friendly buckets. The default
# universe is a curated list of large names from those sectors so first runs
# stay snappy. Switch UNIVERSE_MODE to "all" to scan every A-share.

UNIVERSE_MODE = "curated"  # "curated" | "all"

DEFAULT_UNIVERSE: list[str] = [
    # Banks
    "601398",
    "601939",
    "601288",
    "601988",
    "600036",
    "601328",
    "600016",
    "601166",
    "600000",
    "601818",
    # Utilities / power
    "600900",
    "600886",
    "600795",
    "601985",
    "600025",
    "600011",
    "600021",
    # Coal
    "601088",
    "601225",
    "600188",
    "601898",
    # Telecom
    "600941",
    "601728",
    "600050",
    # Oil & gas
    "601857",
    "600028",
    "601808",
    # Highways / transport
    "600548",
    "600377",
    "601008",
    # Consumer staples (slow growers w/ stable dividends)
    "600519",
    "000858",
    "600887",
]

# ---- Hard gates -----------------------------------------------------------

LIQUIDITY_MIN_TURNOVER_CNY = 1e8  # daily turnover ≥ ¥100M
MIN_TOTAL_MV_CNY = 30e8  # total market cap ≥ ¥3B
LIQUIDITY_LOOKBACK_DAYS = 60
EXCLUDE_ST = True
SCREENER_MAX_WORKERS = 4

# ---- Scoring weights ------------------------------------------------------
# estimate.md strategy_fit = 0.45D + 0.35V + 0.20T.
# Q and R remain visible review factors; red flags are handled separately.

SCORING_RULE_VERSION = 6
W_QUALITY = 0.00
W_DIVIDEND = 0.45
W_VALUATION = 0.35
W_TRADING = 0.20
W_RISK = 0.00

# ---- Quality factor thresholds -------------------------------------------

ROE_MEDIAN_3Y_MIN = 10.0  # %
OCF_OVER_NI_3Y_MIN = 0.9
DEBT_RATIO_MAX_HARD = 75.0  # %, hard gate (industry-agnostic)
NET_MARGIN_MIN = 5.0  # %, soft (used for scoring)

# ---- Dividend factor thresholds ------------------------------------------

DIV_YIELD_FULL_CREDIT = 4.0  # ≥4% yield → full dividend score
DIV_LOOKBACK_YEARS = 5  # 加权股息率回溯年数
DIV_YEAR_WEIGHTS = [0.40, 0.27, 0.18, 0.10, 0.05]  # 由近到远，合计=1

# ---- Valuation factor -----------------------------------------------------

VALUATION_LOOKBACK_YEARS = 5
PERCENTILE_CHEAP = 20.0  # ≤ → 低估带
PERCENTILE_EXPENSIVE = 80.0  # ≥ → 偏贵带

# ---- Trading factor -------------------------------------------------------

ATR_WINDOW = 20
TURNOVER_WINDOW = 60

# ---- Risk sentinels (R penalty + hard pause-add flags) -------------------

REVENUE_DECLINE_QUARTERS = 2  # consecutive quarters
NI_DECLINE_QUARTERS = 2
OCF_OVER_NI_HARD_FLOOR = 0.5
AR_INVENTORY_YOY_WARNING = 40.0  # %, balance-sheet item YoY jump
VALUE_REVENUE_DECLINE_DISCOUNT = 0.70

# ---- Trading plan ---------------------------------------------------------

PER_STOCK_BUDGET_CNY = 100_000  # plan.md: 80k-150k, target ~100k
LOT_SIZE_LOW_PRICE = 200  # price < 5 → 200 shares per ladder
LOT_SIZE_HIGH_PRICE = 100  # price ≥ 5 → 100 shares per ladder
LOT_PRICE_BREAKPOINT = 5.0
CORE_POSITION_RATIO = 0.25
TACTICAL_POSITION_RATIO = 0.75

# ---- Cache TTL ------------------------------------------------------------

CACHE_TTL_HOURS = {
    "universe": 24,
    "price_hist": 12,
    "indicator": 24,
    "financial": 24 * 7,
    "dividend": 24 * 7,
    "bulk": 24 * 7,
    "info": 24 * 7,
    "screener": 24 * 365,
}

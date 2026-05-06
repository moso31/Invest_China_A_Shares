"""SQLite-backed local stock pool.

Schema is intentionally tiny: the pool stores user-curated picks with the
score and risk reasons captured at the moment of addition. Live data
(current price, interval) is recomputed on read so it always reflects the
freshest cache.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from loha.config import DB_PATH


@dataclass
class PoolEntry:
    symbol: str
    name: str
    added_ts: float
    score_at_add: float
    entry_price: float
    notes: str
    flags_at_add: list[str]

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "added_ts": self.added_ts,
            "added_at": time.strftime("%Y-%m-%d", time.localtime(self.added_ts)),
            "score_at_add": self.score_at_add,
            "entry_price": self.entry_price,
            "notes": self.notes,
            "flags_at_add": self.flags_at_add,
        }


def _conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS pool (
            symbol TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            added_ts REAL NOT NULL,
            score_at_add REAL NOT NULL,
            entry_price REAL NOT NULL DEFAULT 0,
            notes TEXT NOT NULL DEFAULT '',
            flags_at_add TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    cols = {r["name"] for r in c.execute("PRAGMA table_info(pool)").fetchall()}
    if "entry_price" not in cols:
        c.execute("ALTER TABLE pool ADD COLUMN entry_price REAL NOT NULL DEFAULT 0")
    return c


def list_all() -> list[PoolEntry]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM pool ORDER BY added_ts DESC").fetchall()
    return [
        PoolEntry(
            symbol=r["symbol"],
            name=r["name"],
            added_ts=r["added_ts"],
            score_at_add=r["score_at_add"],
            entry_price=float(r["entry_price"] or 0),
            notes=r["notes"] or "",
            flags_at_add=json.loads(r["flags_at_add"] or "[]"),
        )
        for r in rows
    ]


def has(symbol: str) -> bool:
    with _conn() as c:
        return c.execute("SELECT 1 FROM pool WHERE symbol = ?", (symbol,)).fetchone() is not None


def add(symbol: str, name: str, score: float, flags: list[str], notes: str = "", entry_price: float = 0.0) -> bool:
    """Insert a new pool entry. Returns False if the symbol was already present."""
    if has(symbol):
        return False
    with _conn() as c:
        c.execute(
            "INSERT INTO pool(symbol, name, added_ts, score_at_add, entry_price, notes, flags_at_add) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                symbol,
                name,
                time.time(),
                float(score),
                float(entry_price or 0),
                notes,
                json.dumps(flags, ensure_ascii=False),
            ),
        )
        c.commit()
    return True


def remove(symbol: str) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM pool WHERE symbol = ?", (symbol,))
        c.commit()
    return cur.rowcount > 0

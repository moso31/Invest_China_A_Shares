import pandas as pd

from loha import factors


def test_same_period_yoy_does_not_compare_quarter_to_yearend():
    declined, yoys = factors._same_period_yoy_streak(
        [
            ("20260331", 120.0),
            ("20251231", 1000.0),
            ("20250331", 100.0),
            ("20241231", 900.0),
        ],
        1,
    )

    assert declined is False
    assert yoys == [("20260331", 20.0)]


def test_risk_flags_ar_and_inventory_from_balance_sheet(monkeypatch):
    monkeypatch.setattr(factors, "_quick_metrics", lambda _symbol: None)
    monkeypatch.setattr(factors.source, "financial_abstract", lambda _symbol: pd.DataFrame())
    monkeypatch.setattr(factors.source, "individual_info", lambda _symbol: {"总市值": 100e8})
    monkeypatch.setattr(
        factors.source,
        "balance_sheet",
        lambda _symbol: pd.DataFrame(
            {
                "GOODWILL": [0.0],
                "TOTAL_PARENT_EQUITY": [100.0],
                "ACCOUNTS_RECE_YOY": [45.0],
                "INVENTORY_YOY": [42.0],
            }
        ),
    )

    result = factors.risk("000001")

    assert result.score == 20.0
    assert any("ar_anomaly" in flag for flag in result.flags)
    assert any("inventory_anomaly" in flag for flag in result.flags)
    assert result.components["ar_anomaly_yoy_pct"] == 45.0
    assert result.components["inventory_anomaly_yoy_pct"] == 42.0

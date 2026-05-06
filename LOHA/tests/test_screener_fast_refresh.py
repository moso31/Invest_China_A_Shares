import logging

import pandas as pd

from loha import config, screener


def test_build_universe_degrades_to_code_name_fallback(monkeypatch, caplog):
    monkeypatch.setattr(screener.source, "realtime_snapshot", lambda: None)
    monkeypatch.setattr(
        screener.source,
        "universe_all",
        lambda: pd.DataFrame(
            {
                "symbol": ["000001", "000002"],
                "name": ["平安银行", "ST测试"],
            }
        ),
    )

    with caplog.at_level(logging.WARNING, logger="loha.screener"):
        universe, source_total, prefiltered_total = screener._build_universe("all")

    assert source_total == 2
    assert prefiltered_total == 1
    assert universe == [
        {
            "symbol": "000001",
            "name": "平安银行",
            "amount": None,
            "total_mv": None,
            "turnover_rate": None,
            "last": None,
        }
    ]
    assert (
        "realtime snapshot unavailable; degraded to stock_info_a_code_name universe with no spot prefilter fields"
        in caplog.text
    )


def test_refresh_spot_snapshot_restores_stale_cache(monkeypatch, caplog):
    stale = pd.DataFrame({"symbol": [str(i).zfill(6) for i in range(1000)]})
    deleted: list[tuple[str, tuple[str, ...]]] = []
    restored: list[tuple[str, tuple[str, ...], pd.DataFrame]] = []

    def fake_get_stale(kind, *key_parts):
        assert kind == "price_hist"
        assert key_parts == ("realtime_snapshot",)
        return stale

    def fake_delete(kind, *key_parts):
        deleted.append((kind, key_parts))

    def fake_put(kind, df, *key_parts, **_kwargs):
        restored.append((kind, key_parts, df.copy()))

    monkeypatch.setattr(screener.cache, "get_stale", fake_get_stale)
    monkeypatch.setattr(screener.cache, "delete", fake_delete)
    monkeypatch.setattr(screener.cache, "put", fake_put)
    monkeypatch.setattr(screener.source, "realtime_snapshot", lambda: None)

    with caplog.at_level(logging.WARNING, logger="loha.screener"):
        screener._refresh_spot_snapshot_preserving_stale()

    assert deleted == [("price_hist", ("realtime_snapshot",))]
    assert len(restored) == 1
    assert restored[0][0] == "price_hist"
    assert restored[0][1] == ("realtime_snapshot",)
    pd.testing.assert_frame_equal(restored[0][2], stale)
    assert "restored stale snapshot cache" in caplog.text


def test_refresh_spot_snapshot_logs_degrade_without_stale(monkeypatch, caplog):
    deleted: list[tuple[str, tuple[str, ...]]] = []

    def fake_delete(kind, *key_parts):
        deleted.append((kind, key_parts))

    def fail_put(*_args, **_kwargs):
        raise AssertionError("stale cache should not be restored when none exists")

    monkeypatch.setattr(screener.cache, "get_stale", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(screener.cache, "delete", fake_delete)
    monkeypatch.setattr(screener.cache, "put", fail_put)
    monkeypatch.setattr(screener.source, "realtime_snapshot", lambda: None)

    with caplog.at_level(logging.WARNING, logger="loha.screener"):
        screener._refresh_spot_snapshot_preserving_stale()

    assert deleted == [("price_hist", ("realtime_snapshot",))]
    assert (
        "fast refresh: realtime snapshot fetch failed and no stale cache available; downstream will degrade to code-name universe"
        in caplog.text
    )


def test_run_fast_refresh_continues_with_degraded_universe(monkeypatch, caplog):
    monkeypatch.setattr(screener, "_refresh_spot_snapshot_preserving_stale", lambda: None)
    monkeypatch.setattr(screener, "_preheat_bulk_caches", lambda force=False: None)
    monkeypatch.setattr(screener.source, "realtime_snapshot", lambda: None)
    monkeypatch.setattr(
        screener.source,
        "universe_all",
        lambda: pd.DataFrame({"symbol": ["000001"], "name": ["平安银行"]}),
    )
    monkeypatch.setattr(screener, "_score_one", lambda _row: ("rejected", None))
    monkeypatch.setattr(screener.cache, "put_json", lambda *_args, **_kwargs: None)

    with caplog.at_level(logging.WARNING, logger="loha.screener"):
        results = screener.run(
            mode="all",
            force_refresh=True,
            refresh_mode=screener.FULL_REFRESH_FAST,
        )

    assert results == []
    assert screener.progress().state == "done"
    assert screener.progress().error is None
    assert screener.progress().total == 1
    assert screener.progress().done == 1
    assert screener.progress().rejected == 1
    assert "degraded to stock_info_a_code_name universe" in caplog.text


def test_run_fast_refresh_reports_error_when_all_universe_sources_fail(monkeypatch):
    monkeypatch.setattr(screener, "_refresh_spot_snapshot_preserving_stale", lambda: None)
    monkeypatch.setattr(screener, "_preheat_bulk_caches", lambda force=False: None)
    monkeypatch.setattr(screener.source, "realtime_snapshot", lambda: None)
    monkeypatch.setattr(screener.source, "universe_all", lambda: None)

    results = screener.run(
        mode="all",
        force_refresh=True,
        refresh_mode=screener.FULL_REFRESH_FAST,
    )

    assert results == []
    assert screener.progress().state == "error"
    assert screener.progress().error == "全市场快照与代码名单均获取失败，疑似网络/代理故障。"


def test_current_cache_row_with_empty_dividend_sources_does_not_repair(monkeypatch):
    def fail_all_factors(*_args, **_kwargs):
        raise AssertionError("current cache rows must not be recomputed")

    monkeypatch.setattr(screener.factors, "all_factors", fail_all_factors)
    row = {
        "symbol": "000001",
        "name": "平安银行",
        "score": 50.0,
        "Q": 50.0,
        "D": 20.0,
        "V": 50.0,
        "T": 50.0,
        "R": 0.0,
        "flags": [],
        "risk_flags": [],
        "good_flags": [],
        "components": {
            "D": {"div_year_sources": {}, "div_by_year": {}},
            "V": {"bucket": "PE_STABLE"},
        },
        "scoring_rule_version": config.SCORING_RULE_VERSION,
    }

    repaired, changed = screener._repair_legacy_dividend_row(row)
    score = screener._score_from_dict(row)

    assert repaired is row
    assert changed is False
    assert score.symbol == "000001"

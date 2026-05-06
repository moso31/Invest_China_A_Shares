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


def test_run_complete_refresh_clears_raw_cache_and_preheats_bulk(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(screener.cache, "clear_data_sources", lambda: calls.append("clear"))
    monkeypatch.setattr(screener, "_preheat_bulk_caches", lambda force=False: calls.append("preheat"))
    monkeypatch.setattr(
        screener,
        "_build_universe",
        lambda _mode: ([{"symbol": "000001", "name": "平安银行"}], 1, 1),
    )
    monkeypatch.setattr(screener, "_score_one", lambda _row: ("rejected", None))
    monkeypatch.setattr(screener.cache, "put_json", lambda *_args, **_kwargs: None)

    results = screener.run(mode="all", force_refresh=True)

    assert results == []
    assert calls == ["clear", "preheat"]
    assert screener.progress().refresh_mode == screener.FULL_REFRESH_COMPLETE
    assert screener.progress().refresh_mode_label == "完全更新"


def test_run_complete_refresh_continues_with_degraded_universe(monkeypatch, caplog):
    monkeypatch.setattr(screener.cache, "clear_data_sources", lambda: None)
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
        results = screener.run(mode="all", force_refresh=True)

    assert results == []
    assert screener.progress().state == "done"
    assert screener.progress().error is None
    assert screener.progress().total == 1
    assert screener.progress().done == 1
    assert screener.progress().rejected == 1
    assert "degraded to stock_info_a_code_name universe" in caplog.text


def test_run_complete_refresh_reports_error_when_all_universe_sources_fail(monkeypatch):
    monkeypatch.setattr(screener.cache, "clear_data_sources", lambda: None)
    monkeypatch.setattr(screener, "_preheat_bulk_caches", lambda force=False: None)
    monkeypatch.setattr(screener.source, "realtime_snapshot", lambda: None)
    monkeypatch.setattr(screener.source, "universe_all", lambda: None)

    results = screener.run(mode="all", force_refresh=True)

    assert results == []
    assert screener.progress().state == "error"
    assert "全市场快照与代码名单均获取失败" in screener.progress().error


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


def test_debug_recompute_can_rebuild_without_score_cache(monkeypatch):
    monkeypatch.setattr(screener, "_cached_full_scan_rows", lambda: [])
    monkeypatch.setattr(screener, "full_scan_meta", lambda: {})
    monkeypatch.setattr(
        screener,
        "_build_universe",
        lambda _mode: ([{"symbol": "000001", "name": "平安银行"}], 1, 1),
    )
    monkeypatch.setattr(
        screener,
        "_score_one",
        lambda _row: (
            "accepted",
            screener.StockScore(
                symbol="000001",
                name="平安银行",
                score=70.0,
                Q=50.0,
                D=80.0,
                V=60.0,
                T=70.0,
                R=0.0,
            ),
        ),
    )
    written: list[tuple[str, str, object]] = []
    monkeypatch.setattr(screener.cache, "put_json", lambda kind, data, key: written.append((kind, key, data)))

    results = screener.recompute_cached_full_scan(cache_only=True)

    assert len(results) == 1
    assert screener.progress().state == "done"
    assert screener.progress().accepted == 1
    assert any(key == screener.FULL_SCAN_RESULTS_KEY for _, key, _ in written)

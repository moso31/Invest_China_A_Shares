import pandas as pd

from loha import factors


def test_601398_bulk_and_hist_dividend_paths_match():
    bulk_fhps = pd.DataFrame(
        {
            "SECURITY_CODE": ["601398", "601398", "601398"],
            "REPORT_DATE": ["20241231", "20231231", "20221231"],
            "PRETAX_BONUS_RMB": [3.064, 3.064, 3.035],
        }
    )
    dividend_hist = pd.DataFrame(
        {
            "实施分红年度": ["2024-12-31", "2023-12-31", "2022-12-31"],
            "派息比例": [3.064, 3.064, 3.035],
        }
    )

    bulk_by_year = factors._dividend_by_year_from_bulk_fhps(bulk_fhps)
    hist_by_year = factors._dividend_by_year_from_hist(dividend_hist)

    assert bulk_by_year == hist_by_year
    for year, bulk_value in bulk_by_year.items():
        hist_value = hist_by_year[year]
        assert abs(bulk_value - hist_value) / hist_value <= 0.01

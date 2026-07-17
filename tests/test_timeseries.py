"""Unit tests for the time-series tools — pure, no network."""

from blf.timeseries import (
    Series,
    model_estimate,
    parse_fred_csv,
    parse_yahoo_json,
    trim_window,
)

FRED_SAMPLE = """observation_date,UNRATE
2024-01-01,3.7
2024-02-01,3.9
2024-03-01,.
2024-04-01,3.9
"""

# 2024-05-01, 05-02, 05-03 as UTC-midnight unix timestamps; second close is null.
YAHOO_SAMPLE = {
    "chart": {
        "result": [
            {
                "timestamp": [1714521600, 1714608000, 1714694400],
                "indicators": {"quote": [{"close": [101.5, None, 102.0]}]},
            }
        ]
    }
}


def test_parse_fred_skips_missing():
    s = parse_fred_csv(FRED_SAMPLE)
    assert s.dates == ["2024-01-01", "2024-02-01", "2024-04-01"]  # '.' dropped
    assert s.values == [3.7, 3.9, 3.9]


def test_parse_yahoo_skips_nulls():
    s = parse_yahoo_json(YAHOO_SAMPLE)
    assert s.values == [101.5, 102.0]  # null close dropped
    assert s.dates == ["2024-05-01", "2024-05-03"]


def test_parse_yahoo_malformed_returns_empty():
    assert len(parse_yahoo_json({"chart": {"result": []}})) == 0


def test_trim_window_respects_cutoff_and_window():
    dates = [f"2024-06-{d:02d}" for d in range(1, 21)]
    vals = list(range(1, 21))
    s = Series(dates, vals)
    trimmed = trim_window(s, cutoff="2024-06-15", window_days=5)
    # cutoff drops 16..20; window keeps the 5 calendar days before the 15th
    assert trimmed.dates[-1] == "2024-06-15"
    assert all(d <= "2024-06-15" for d in trimmed.dates)
    assert trimmed.dates[0] == "2024-06-10"


def test_model_estimate_threshold_monotonic():
    # Steadily rising series; last value 110.
    dates = [f"2024-07-{d:02d}" for d in range(1, 12)]
    vals = [100 + i for i in range(11)]  # 100..110
    s = Series(dates, vals)
    low = model_estimate(s, threshold=90, resolution_date="2024-07-31")
    high = model_estimate(s, threshold=200, resolution_date="2024-07-31")
    assert low["p"] > high["p"]                 # easier threshold -> higher prob
    assert 0.5 < low["p"] <= 1.0                # already above 90 and trending up
    assert high["p"] < 0.5                      # far above current level


def test_model_estimate_past_resolution_is_deterministic():
    s = Series(["2024-08-01", "2024-08-02", "2024-08-03"], [10.0, 11.0, 12.0])
    # resolution date on/before last obs -> collapses to comparison with last value
    est = model_estimate(s, threshold=11.5, resolution_date="2024-08-03")
    assert est["horizon_days"] == 0
    assert est["p"] > 0.99   # last value 12 > 11.5


def test_model_estimate_needs_enough_history():
    s = Series(["2024-09-01", "2024-09-02"], [5.0, 6.0])
    assert "error" in model_estimate(s, threshold=7, resolution_date="2024-09-10")

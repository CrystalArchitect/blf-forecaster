"""Unit tests for shrinkage (f, c) grid-search fitting — offline, deterministic."""

import json

from blf.shrinkage import (
    ShrinkageParams,
    TrialRecord,
    fit_shrinkage,
    load_trial_records,
    mean_brier,
)


def _overconfident_disagreeing(n_each=15):
    # Each question's two trials disagree and skew high (aggregates ~0.89 with no
    # shrinkage), but outcomes are 50/50 -> the optimal forecast is near 0.5, so
    # aggressive shrinkage (small alpha) should win.
    recs = []
    for i in range(2 * n_each):
        recs.append(TrialRecord(trial_ps=[0.4, 0.99], outcome=float(i < n_each), prior=0.5))
    return recs


def test_fit_prefers_more_shrinkage_when_it_helps():
    recs = _overconfident_disagreeing()
    params, report = fit_shrinkage(recs)
    assert report["fitted"] is True
    # Fitted params must not score worse than the code defaults, and here should
    # strictly improve (more shrinkage pulls the 0.89 aggregate toward 0.5).
    assert report["brier_fit"] <= report["brier_default"] + 1e-12
    assert report["brier_fit"] < report["brier_default"]
    # The fit should shrink harder than "no shrink" (alpha=1 => f=1,c=0).
    assert not (params.f == 1.0 and params.c == 0.0)


def test_fit_is_optimal_over_grid():
    recs = _overconfident_disagreeing()
    params, report = fit_shrinkage(recs)
    # No grid point should beat the reported fit.
    from blf.shrinkage import DEFAULT_GRID_C, DEFAULT_GRID_F
    best = min(mean_brier(recs, f, c) for f in DEFAULT_GRID_F for c in DEFAULT_GRID_C)
    assert abs(report["brier_fit"] - best) < 1e-12


def test_min_records_keeps_defaults():
    from blf.aggregate import DEFAULT_C, DEFAULT_FLOOR

    recs = [TrialRecord(trial_ps=[0.6, 0.7], outcome=1.0) for _ in range(5)]
    params, report = fit_shrinkage(recs)
    assert report["fitted"] is False
    assert (params.f, params.c) == (DEFAULT_FLOOR, DEFAULT_C)
    assert params.n_fit == 5


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "shrink.json"
    ShrinkageParams(f=0.05, c=1.5, n_fit=42).save(p)
    loaded = ShrinkageParams.load(p)
    assert loaded is not None
    assert loaded.f == 0.05 and loaded.c == 1.5 and loaded.n_fit == 42


def test_load_none_when_missing(tmp_path):
    assert ShrinkageParams.load(tmp_path / "nope.json") is None


def test_load_trial_records_skips_rows_without_trials(tmp_path):
    path = tmp_path / "preds.jsonl"
    path.write_text(
        json.dumps({"outcome": 1, "trial_ps": [0.3, 0.8], "prior": 0.5, "source": "fred"}) + "\n"
        + json.dumps({"outcome": 0, "trial_ps": None}) + "\n"          # skipped
        + json.dumps({"outcome": 1}) + "\n"                            # skipped
    )
    recs = load_trial_records(path)
    assert len(recs) == 1
    assert recs[0].trial_ps == [0.3, 0.8] and recs[0].source == "fred"

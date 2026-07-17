"""Unit tests for Platt calibration — deterministic, no network / API."""

from blf.calibration import (
    PlattCalibrator,
    ResolvedRecord,
    ResolvedStore,
    brier,
    fit_platt,
)


def _records(entries):
    """entries: list of (p, source, n_yes, n_total) -> flat ResolvedRecord list."""
    out = []
    for p, source, n_yes, n_total in entries:
        for i in range(n_total):
            out.append(ResolvedRecord(p=p, source=source, outcome=1 if i < n_yes else 0))
    return out


# A systematically over-confident forecaster: predictions are too extreme
# relative to how often events actually resolve YES.
OVERCONFIDENT = _records(
    [
        (0.95, "m", 75, 100),
        (0.80, "m", 65, 100),
        (0.20, "m", 35, 100),
        (0.05, "m", 25, 100),
    ]
)


def test_min_data_falls_back_to_identity():
    cal = fit_platt(_records([(0.7, "m", 3, 5)]))  # 5 records < MIN_RECORDS
    assert cal.is_identity
    assert abs(cal.transform(0.73) - 0.73) < 1e-6


def test_calibration_reduces_extremeness():
    cal = fit_platt(OVERCONFIDENT)
    # extreme predictions get pulled toward the center
    assert cal.transform(0.95) < 0.95
    assert cal.transform(0.05) > 0.05


def test_calibration_improves_brier():
    cal = fit_platt(OVERCONFIDENT)
    assert brier(OVERCONFIDENT, cal) < brier(OVERCONFIDENT, None)


def test_transform_is_monotonic():
    cal = fit_platt(OVERCONFIDENT)
    assert cal.transform(0.2) < cal.transform(0.5) < cal.transform(0.8)


def test_per_source_offsets_capture_source_bias():
    # Same raw p=0.5 from two sources, but wildly different base rates.
    recs = _records([(0.5, "hi", 42, 50), (0.5, "lo", 8, 50)])
    cal = fit_platt(recs, hierarchical=True)
    assert set(cal.deltas) == {"hi", "lo"}
    assert cal.transform(0.5, "hi") > 0.5 > cal.transform(0.5, "lo")
    # unknown source uses only the global term (delta = 0)
    assert 0.0 < cal.transform(0.5, "unknown") < 1.0


def test_save_load_roundtrip(tmp_path):
    cal = fit_platt(OVERCONFIDENT)
    path = tmp_path / "cal.json"
    cal.save(path)
    loaded = PlattCalibrator.load(path)
    assert loaded is not None
    assert abs(loaded.a - cal.a) < 1e-9
    assert abs(loaded.transform(0.7) - cal.transform(0.7)) < 1e-9


def test_store_roundtrip(tmp_path):
    store = ResolvedStore(tmp_path / "resolved.jsonl")
    store.add(ResolvedRecord(p=0.6, source="polymarket", outcome=1, question="q1"))
    store.add(ResolvedRecord(p=0.3, source="fred", outcome=0))
    loaded = store.load()
    assert len(loaded) == 2
    assert loaded[0].source == "polymarket" and loaded[0].outcome == 1
    assert loaded[1].p == 0.3

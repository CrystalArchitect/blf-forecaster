"""Tests for the ForecastBench adapter — offline, using inline fixtures that
mirror the real dataset schema."""

import json

from blf.forecastbench import load_forecastbench, load_questions

QUESTION_SET = {
    "forecast_due_date": "2024-07-12",
    "question_set": "2024-07-12-llm.json",
    "questions": [
        {
            "id": "mkt1",
            "source": "manifold",
            "question": "Will a New York team win a championship in 2025?",
            "resolution_criteria": "Resolves per the linked market.",
            "freeze_datetime": "2024-07-02T00:00:00+00:00",
            "freeze_datetime_value": "0.254",
            "resolution_dates": "N/A",
        },
        {
            "id": "fred_DTB3",
            "source": "fred",
            "question": "Will the 3-month T-bill rate have increased by {resolution_date} "
            "compared to its value on {forecast_due_date}?",
            "resolution_criteria": "Resolves to the value at fred series DTB3.",
            "freeze_datetime": "2024-07-12T00:00:00+00:00",
            "freeze_datetime_value": "5.23",
            "resolution_dates": ["2024-07-28", "2024-10-19"],
        },
    ],
}

RESOLUTION_SET = {
    "forecast_due_date": "2024-07-12",
    "question_set": "2024-07-12-llm.json",
    "resolutions": [
        {"id": "mkt1", "source": "manifold", "direction": None,
         "resolution_date": "2025-02-10", "resolved_to": 0.0, "resolved": True},
        {"id": "fred_DTB3", "source": "fred", "direction": None,
         "resolution_date": "2024-07-28", "resolved_to": 1.0, "resolved": True},
        {"id": "fred_DTB3", "source": "fred", "direction": None,
         "resolution_date": "2024-10-19", "resolved_to": 0.0, "resolved": True},
        # should be skipped: unresolved
        {"id": "fred_DTB3", "source": "fred", "direction": None,
         "resolution_date": "2027-07-21", "resolved_to": None, "resolved": False},
        # should be skipped: combo question (direction set)
        {"id": "mkt1", "source": "manifold", "direction": [1, -1],
         "resolution_date": "2025-02-10", "resolved_to": 1.0, "resolved": True},
    ],
}


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def test_forecastbench_expands_and_maps(tmp_path):
    q = _write(tmp_path, "q.json", QUESTION_SET)
    r = _write(tmp_path, "r.json", RESOLUTION_SET)
    qs = load_forecastbench(q, r)

    # 1 market + 2 fred horizons = 3; unresolved and combo entries dropped.
    assert len(qs) == 3
    ids = {x.id for x in qs}
    assert ids == {"mkt1@2025-02-10", "fred_DTB3@2024-07-28", "fred_DTB3@2024-10-19"}


def test_forecastbench_templates_and_flags(tmp_path):
    q = _write(tmp_path, "q.json", QUESTION_SET)
    r = _write(tmp_path, "r.json", RESOLUTION_SET)
    by_id = {x.id: x for x in load_forecastbench(q, r)}

    fred = by_id["fred_DTB3@2024-07-28"]
    assert "{resolution_date}" not in fred.text and "2024-07-28" in fred.text
    assert "2024-07-12" in fred.text                # forecast_due_date templated
    assert fred.timeseries is True and fred.is_market is False
    assert fred.cutoff == "2024-07-12"              # cutoff = forecast due date
    assert fred.outcome == 1.0
    assert "reference value" in fred.text.lower()   # freeze value injected

    mkt = by_id["mkt1@2025-02-10"]
    assert mkt.is_market is True and mkt.timeseries is False
    assert "crowd/market estimate was 0.254" in mkt.text


def test_forecastbench_source_filter(tmp_path):
    q = _write(tmp_path, "q.json", QUESTION_SET)
    r = _write(tmp_path, "r.json", RESOLUTION_SET)
    qs = load_forecastbench(q, r, sources={"fred"})
    assert qs and all(x.source == "fred" for x in qs)


def test_load_internal_schema_sample():
    qs = load_questions("data/sample_questions.json")
    assert len(qs) == 3
    spy = next(x for x in qs if x.id == "spy-above-400-2024-06-28")
    assert spy.timeseries is True and spy.is_market is False

"""Unit tests for the eval harness scoring + runner — offline, no LLM."""

from blf.eval import (
    EvalQuestion,
    EvalReport,
    EvalResult,
    brier_index,
    brier_score,
    run_eval,
)


def _q(qid, source, is_market):
    return EvalQuestion(
        id=qid, text="?", source=source, cutoff="2026-01-01",
        resolution_date="2026-06-01", outcome=1.0, is_market=is_market,
    )


def test_brier_index_constant_half_scores_50():
    # A constant 0.5 forecaster has Brier 0.25 -> BI 50.
    assert brier_score(0.5, 1.0) == 0.25
    assert abs(brier_index(0.25) - 50.0) < 1e-9


def test_brier_index_perfect_scores_100():
    assert abs(brier_index(0.0) - 100.0) < 1e-9


def test_overall_is_unweighted_average_of_categories():
    # 1 market question @ Brier 0.25 (BI 50); 3 dataset questions @ Brier 0.09 (BI 70).
    # Unweighted category average -> overall BI 60, NOT pooled (which would skew to 70).
    report = EvalReport(
        results=[
            EvalResult("m1", "metaculus", True, 0.5, 1.0, 0.25),
            EvalResult("d1", "fred", False, 0.7, 1.0, 0.09),
            EvalResult("d2", "fred", False, 0.7, 1.0, 0.09),
            EvalResult("d3", "fred", False, 0.7, 1.0, 0.09),
        ]
    )
    s = report.summary()
    assert abs(s["market_bi"] - 50.0) < 1e-9
    assert abs(s["dataset_bi"] - 70.0) < 1e-9
    assert abs(s["overall_bi"] - 60.0) < 1e-9  # (50 + 70) / 2, unweighted


def test_single_category_overall_equals_that_category():
    report = EvalReport(results=[EvalResult("m1", "metaculus", True, 0.5, 1.0, 0.25)])
    s = report.summary()
    assert s["dataset_bi"] is None
    assert abs(s["overall_bi"] - 50.0) < 1e-9


def test_run_eval_with_mock_predictor():
    questions = [_q("a", "metaculus", True), _q("b", "fred", False)]
    # Mock predictor: always confident-correct.
    report = run_eval(questions, predict_fn=lambda q: 1.0)
    assert len(report.results) == 2
    assert all(r.brier == 0.0 for r in report.results)
    assert abs(report.summary()["overall_bi"] - 100.0) < 1e-9


def test_run_eval_clamps_probabilities():
    q = [_q("a", "metaculus", True)]
    report = run_eval(q, predict_fn=lambda _q: 1.5)  # out of range
    assert report.results[0].p == 1.0


def test_on_result_callback_streams():
    seen = []
    run_eval([_q("a", "fred", False)], predict_fn=lambda q: 0.3, on_result=seen.append)
    assert len(seen) == 1 and seen[0].id == "a"

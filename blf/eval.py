"""ForecastBench-style evaluation harness: scoring + runner.

The scoring follows the paper's primary metric, the **Brier Index**
(Kucinskas et al. 2026):

    BI = 100 * (1 - sqrt(mean_Brier))

so higher is better and a constant 0.5 forecaster scores 50. Following the
ForecastBench methodology, the **overall** score is the unweighted average of the
market and dataset category scores (not a pooled mean), so the two question types
count equally regardless of how many of each there are.

This module is deliberately format-agnostic: questions are represented by the
internal `EvalQuestion` schema, and the ForecastBench loader/adapter lives in
`blf.forecastbench`. The runner takes an injectable `predict_fn`, so the scoring
and orchestration can be tested with no LLM/network.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class EvalQuestion:
    id: str
    text: str
    source: str
    cutoff: str                 # forecast/freeze date = knowledge cutoff for the run
    resolution_date: str
    outcome: float              # ground-truth resolution in [0, 1] (usually 0 or 1)
    is_market: bool = False     # market question (vs. dataset/time-series question)
    timeseries: bool = False    # enable the numeric time-series tools for this question
    horizon: str | None = None  # label when a dataset question has multiple horizons


@dataclass
class Prediction:
    """A predictor's output. `trial_ps`/`prior` are optional but required to
    later grid-search the shrinkage hyperparameters offline."""
    p: float
    trial_ps: list[float] | None = None
    prior: float = 0.5


@dataclass
class EvalResult:
    id: str
    source: str
    is_market: bool
    p: float
    outcome: float
    brier: float
    horizon: str | None = None
    trial_ps: list[float] | None = None
    prior: float = 0.5


def brier_score(p: float, outcome: float) -> float:
    return (p - outcome) ** 2


def brier_index(mean_brier: float) -> float:
    """BI = 100 * (1 - sqrt(mean Brier)). Undefined for an empty set."""
    return 100.0 * (1.0 - math.sqrt(mean_brier))


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    # -- category means --
    def _mean_brier(self, market: bool | None) -> float | None:
        subset = [
            r for r in self.results if market is None or r.is_market == market
        ]
        if not subset:
            return None
        return sum(r.brier for r in subset) / len(subset)

    def category_bi(self, market: bool | None) -> float | None:
        mb = self._mean_brier(market)
        return None if mb is None else brier_index(mb)

    def summary(self) -> dict:
        market_bi = self.category_bi(True)
        dataset_bi = self.category_bi(False)
        present = [b for b in (market_bi, dataset_bi) if b is not None]
        # FB overall = unweighted average of the category scores.
        overall_bi = sum(present) / len(present) if present else None
        n_market = sum(1 for r in self.results if r.is_market)
        return {
            "n": len(self.results),
            "n_market": n_market,
            "n_dataset": len(self.results) - n_market,
            "overall_bi": overall_bi,
            "market_bi": market_bi,
            "dataset_bi": dataset_bi,
            "overall_mean_brier": self._mean_brier(None),
        }

    def format(self) -> str:
        s = self.summary()

        def fmt(x):
            return f"{x:.1f}" if x is not None else "  — "

        lines = [
            "=== ForecastBench evaluation ===",
            f"questions scored: {s['n']}  (market {s['n_market']}, dataset {s['n_dataset']})",
            "",
            f"  Brier Index — Overall: {fmt(s['overall_bi'])}   "
            f"Market: {fmt(s['market_bi'])}   Dataset: {fmt(s['dataset_bi'])}",
            f"  (mean Brier overall: {s['overall_mean_brier']:.4f})"
            if s["overall_mean_brier"] is not None
            else "",
        ]
        return "\n".join(line for line in lines if line != "")


# A predictor maps a question to a probability, or a richer Prediction.
PredictFn = Callable[[EvalQuestion], "float | Prediction"]


def run_eval(
    questions: list[EvalQuestion],
    predict_fn: PredictFn,
    on_result: Callable[[EvalResult], None] | None = None,
) -> EvalReport:
    """Score ``predict_fn`` over ``questions``. Questions run sequentially (the
    BLF predictor already parallelizes its K trials internally); ``on_result`` is
    called after each for progress/streaming to disk. ``predict_fn`` may return a
    bare float or a Prediction (carrying per-trial forecasts for later fitting)."""
    report = EvalReport()
    for q in questions:
        raw = predict_fn(q)
        pred = raw if isinstance(raw, Prediction) else Prediction(p=float(raw))
        p = min(max(pred.p, 0.0), 1.0)
        res = EvalResult(
            id=q.id,
            source=q.source,
            is_market=q.is_market,
            p=p,
            outcome=q.outcome,
            brier=brier_score(p, q.outcome),
            horizon=q.horizon,
            trial_ps=pred.trial_ps,
            prior=pred.prior,
        )
        report.results.append(res)
        if on_result is not None:
            on_result(res)
    return report


def blf_predictor(
    trials: int = 5,
    max_steps: int = 10,
    calibrate: bool = False,
    prior: float = 0.5,
) -> PredictFn:
    """Default predictor that runs the full BLF pipeline per question and returns
    a Prediction carrying the per-trial forecasts (for shrinkage fitting)."""
    from .forecast import forecast  # local import: avoids LLM deps for scoring tests

    def _predict(q: EvalQuestion) -> Prediction:
        fc = forecast(
            q.text,
            cutoff=q.cutoff,
            trials=trials,
            max_steps=max_steps,
            prior=prior,
            source=q.source,
            calibrate=calibrate,
            enable_timeseries=q.timeseries,
        )
        return Prediction(p=fc.p, trial_ps=fc.aggregation.trial_ps, prior=prior)

    return _predict

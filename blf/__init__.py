"""Bayesian Linguistic Forecaster (BLF) — v1 reimplementation.

Reimplements the two novel, training-free ideas from Murphy (2026),
"Agentic Forecasting using Sequential Bayesian Updating of Linguistic Beliefs"
(arXiv:2604.18576):

    1. the linguistic belief-state agent loop (blf.agent / blf.belief)
    2. multi-trial logit-space aggregation with shrinkage (blf.aggregate)

Hierarchical Platt calibration (paper's 3rd idea) needs a backlog of resolved
forecasts and is intentionally left out of v1.
"""

from .aggregate import Aggregation, aggregate
from .belief import BeliefState
from .calibration import PlattCalibrator, ResolvedRecord, ResolvedStore, fit_platt
from .eval import EvalQuestion, EvalReport, brier_index, run_eval
from .forecast import Forecast, forecast
from .shrinkage import ShrinkageParams, fit_shrinkage

__all__ = [
    "forecast",
    "Forecast",
    "aggregate",
    "Aggregation",
    "BeliefState",
    "PlattCalibrator",
    "ResolvedRecord",
    "ResolvedStore",
    "fit_platt",
    "EvalQuestion",
    "EvalReport",
    "brier_index",
    "run_eval",
    "ShrinkageParams",
    "fit_shrinkage",
]

__version__ = "0.1.0"

"""Top-level BLF forecaster: K trials -> logit aggregation -> calibration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .agent import TrialResult, run_trial
from .aggregate import DEFAULT_C, DEFAULT_FLOOR, Aggregation, aggregate
from .calibration import PlattCalibrator
from .shrinkage import ShrinkageParams
from .llm import LLM
from .search import make_search


@dataclass
class Forecast:
    p: float                 # final probability (calibrated if a calibrator applied)
    raw_p: float             # aggregated probability before calibration
    calibrated: bool
    aggregation: Aggregation
    trials: list[TrialResult]
    source: str | None = None


def forecast(
    question: str,
    cutoff: str | None = None,
    trials: int = 5,
    max_steps: int = 10,
    prior: float = 0.5,
    shrink: bool = True,
    shrink_floor: float | None = None,
    shrink_c: float | None = None,
    source: str | None = None,
    calibrate: bool = False,
    calibrator: PlattCalibrator | None = None,
    enable_timeseries: bool = False,
    max_workers: int = 5,
    llm: LLM | None = None,
    search=None,
) -> Forecast:
    """Run K independent BLF rollouts, aggregate them in logit space, and
    optionally apply Platt calibration.

    K defaults to 5, matching the paper. Trials are independent, so we run them
    concurrently. Shrinkage (toward ``prior``) pulls the estimate back when the
    trials disagree. If ``calibrate`` is set (or a ``calibrator`` is passed), the
    aggregated probability is mapped through the fitted calibrator; ``source``
    selects the per-source offset.

    ``shrink_floor``/``shrink_c`` (the f, c hyperparameters) default to the fitted
    values in ~/.blf/shrinkage.json when present, else the code defaults; pass
    explicit values to override.
    """
    if shrink_floor is None or shrink_c is None:
        fitted = ShrinkageParams.load()
        if shrink_floor is None:
            shrink_floor = fitted.f if fitted else DEFAULT_FLOOR
        if shrink_c is None:
            shrink_c = fitted.c if fitted else DEFAULT_C

    llm = llm or LLM()
    search = search or make_search()

    def _one(_i: int) -> TrialResult:
        return run_trial(
            question, llm, search, cutoff=cutoff, max_steps=max_steps,
            enable_timeseries=enable_timeseries,
        )

    if trials == 1:
        results = [_one(0)]
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, trials)) as ex:
            results = list(ex.map(_one, range(trials)))

    agg = aggregate(
        [r.p for r in results], prior=prior, shrink=shrink, f=shrink_floor, c=shrink_c
    )
    raw_p = agg.p

    if calibrator is None and calibrate:
        calibrator = PlattCalibrator.load()  # default ~/.blf/calibrator.json

    if calibrator is not None:
        return Forecast(
            p=calibrator.transform(raw_p, source),
            raw_p=raw_p,
            calibrated=True,
            aggregation=agg,
            trials=results,
            source=source,
        )

    return Forecast(
        p=raw_p, raw_p=raw_p, calibrated=False, aggregation=agg, trials=results, source=source
    )

"""Fit the shrinkage hyperparameters (f, c) by grid search (paper Sec. D.7).

The multi-trial aggregator uses alpha = max(f, 1 - c*std(logits)) (Eq. 11). The
paper selects the two global hyperparameters (f, c) by grid search on
ForecastBench. Because aggregation is a *pure function* of the cached per-trial
forecasts, we run the expensive LLM rollouts only once — recording each
question's `trial_ps`, `prior`, and resolved `outcome` — and then replay
`aggregate(...)` across the (f, c) grid instantly, picking the pair that minimizes
the mean Brier score (equivalently, maximizes the Brier Index).

Fitted params persist to ~/.blf/shrinkage.json and are auto-loaded by
`forecast()` unless overridden. With too few records, the defaults are kept.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .aggregate import DEFAULT_C, DEFAULT_FLOOR, aggregate

MIN_RECORDS = 20
DEFAULT_GRID_F = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
DEFAULT_GRID_C = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _data_dir() -> Path:
    return Path(os.environ.get("BLF_DATA_DIR", str(Path.home() / ".blf"))).expanduser()


@dataclass
class TrialRecord:
    trial_ps: list[float]
    outcome: float
    prior: float = 0.5
    source: str = ""


@dataclass
class ShrinkageParams:
    f: float = DEFAULT_FLOOR
    c: float = DEFAULT_C
    n_fit: int = 0

    def save(self, path: str | os.PathLike | None = None) -> Path:
        p = Path(path) if path else _data_dir() / "shrinkage.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
        return p

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "ShrinkageParams | None":
        p = Path(path) if path else _data_dir() / "shrinkage.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return cls(f=float(d["f"]), c=float(d["c"]), n_fit=int(d.get("n_fit", 0)))


def load_trial_records(path: str | os.PathLike) -> list[TrialRecord]:
    """Read a predictions JSONL (from `blf-eval --out`) into TrialRecords,
    skipping rows that lack per-trial forecasts."""
    out: list[TrialRecord] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        tps = d.get("trial_ps")
        if not tps:
            continue
        out.append(
            TrialRecord(
                trial_ps=[float(x) for x in tps],
                outcome=float(d["outcome"]),
                prior=float(d.get("prior", 0.5)),
                source=str(d.get("source", "")),
            )
        )
    return out


def mean_brier(records: list[TrialRecord], f: float, c: float) -> float:
    total = 0.0
    for r in records:
        p = aggregate(r.trial_ps, prior=r.prior, f=f, c=c).p
        total += (p - r.outcome) ** 2
    return total / len(records)


def brier_index(mb: float) -> float:
    return 100.0 * (1.0 - math.sqrt(mb))


def fit_shrinkage(
    records: list[TrialRecord],
    grid_f: tuple[float, ...] = DEFAULT_GRID_F,
    grid_c: tuple[float, ...] = DEFAULT_GRID_C,
) -> tuple[ShrinkageParams, dict]:
    """Grid-search (f, c) to minimize mean Brier over the cached trial records."""
    usable = [r for r in records if r.trial_ps]
    default_mb = mean_brier(usable, DEFAULT_FLOOR, DEFAULT_C) if usable else float("nan")

    if len(usable) < MIN_RECORDS:
        return ShrinkageParams(DEFAULT_FLOOR, DEFAULT_C, len(usable)), {
            "fitted": False,
            "reason": f"only {len(usable)} usable records (< {MIN_RECORDS}); kept defaults",
            "brier_default": default_mb,
            "brier_fit": default_mb,
        }

    best_mb, best_fc = math.inf, (DEFAULT_FLOOR, DEFAULT_C)
    for f in grid_f:
        for c in grid_c:
            mb = mean_brier(usable, f, c)
            if mb < best_mb - 1e-12:
                best_mb, best_fc = mb, (f, c)

    params = ShrinkageParams(f=best_fc[0], c=best_fc[1], n_fit=len(usable))
    report = {
        "fitted": True,
        "n": len(usable),
        "brier_default": default_mb,
        "brier_fit": best_mb,
        "bi_default": brier_index(default_mb),
        "bi_fit": brier_index(best_mb),
    }
    return params, report

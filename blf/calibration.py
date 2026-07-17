"""Hierarchical Platt calibration (paper Sec. C.11 / "Calibration").

Platt scaling maps a raw forecast to a calibrated probability by fitting a
logistic regression in logit space:

    p_cal = sigmoid( a * logit(p_raw) + b )

Perfect calibration is (a, b) = (1, 0). A global fit, however, can over-shrink
well-calibrated extreme predictions coming from sources with skewed base rates
(the paper's motivation). So BLF uses *hierarchical* Platt scaling: a shared
slope/intercept plus a per-source intercept offset delta_s, L2-regularized
toward zero:

    p_cal = sigmoid( a * logit(p_raw) + b + delta_{source} )
    penalty = (lambda / 2) * sum_s delta_s^2

The regularization strength lambda is chosen by cross-validation (leave-one-out
when data is scarce, else 5-fold) — large lambda collapses the deltas to a plain
global Platt fit; small lambda lets each source drift. Fitting is by Newton /
IRLS (the objective is convex).

This layer only helps once you have a backlog of *resolved* forecasts to fit on
(see ResolvedStore). With too few records it falls back to the identity map.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_EPS = 1e-4
MIN_RECORDS = 10  # below this we do not trust a fit; return identity
LAMBDA_GRID = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


# --------------------------------------------------------------------------- #
# Resolved-forecast store
# --------------------------------------------------------------------------- #
@dataclass
class ResolvedRecord:
    p: float           # raw (pre-calibration) forecast that was made
    source: str        # e.g. "polymarket", "fred", "manual"
    outcome: int       # 1 if the event resolved YES, else 0
    question: str = ""


def _data_dir() -> Path:
    return Path(os.environ.get("BLF_DATA_DIR", str(Path.home() / ".blf"))).expanduser()


class ResolvedStore:
    """Append-only JSONL log of resolved forecasts used to fit the calibrator."""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = Path(path) if path else _data_dir() / "resolved.jsonl"

    def add(self, record: ResolvedRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def load(self) -> list[ResolvedRecord]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(
                ResolvedRecord(
                    p=float(d["p"]),
                    source=str(d.get("source", "")),
                    outcome=int(d["outcome"]),
                    question=str(d.get("question", "")),
                )
            )
        return out


# --------------------------------------------------------------------------- #
# The calibrator
# --------------------------------------------------------------------------- #
@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0
    deltas: dict[str, float] = field(default_factory=dict)
    lam: float = 0.0
    n_fit: int = 0

    @classmethod
    def identity(cls) -> "PlattCalibrator":
        return cls(a=1.0, b=0.0, deltas={}, lam=0.0, n_fit=0)

    @property
    def is_identity(self) -> bool:
        return self.a == 1.0 and self.b == 0.0 and not self.deltas

    def transform(self, p: float, source: str | None = None) -> float:
        """Map a raw probability to its calibrated value. Unknown sources use
        only the global (a, b) term (delta = 0)."""
        z = self.a * _logit(p) + self.b + self.deltas.get(source or "", 0.0)
        return float(_sigmoid(z))

    # -- persistence --
    def save(self, path: str | os.PathLike | None = None) -> Path:
        p = Path(path) if path else _data_dir() / "calibrator.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
        return p

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "PlattCalibrator | None":
        p = Path(path) if path else _data_dir() / "calibrator.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return cls(
            a=float(d["a"]),
            b=float(d["b"]),
            deltas={str(k): float(v) for k, v in d.get("deltas", {}).items()},
            lam=float(d.get("lam", 0.0)),
            n_fit=int(d.get("n_fit", 0)),
        )


# --------------------------------------------------------------------------- #
# Fitting (Newton / IRLS on a convex penalized logistic objective)
# --------------------------------------------------------------------------- #
def _design(records: list[ResolvedRecord], sources: list[str]):
    idx = {s: i for i, s in enumerate(sources)}
    n, k = len(records), len(sources)
    X = np.zeros((n, 2 + k))
    y = np.zeros(n)
    for i, r in enumerate(records):
        X[i, 0] = _logit(r.p)   # slope feature
        X[i, 1] = 1.0           # global intercept
        if r.source in idx:
            X[i, 2 + idx[r.source]] = 1.0
        y[i] = float(r.outcome)
    return X, y


def _fit_newton(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 100, tol: float = 1e-9):
    n, d = X.shape
    theta = np.zeros(d)
    theta[0] = 1.0  # init at identity (a=1, b=0, deltas=0)
    pen = np.zeros(d)
    if d > 2:
        pen[2:] = lam  # L2 only on the per-source offsets
    Pmat = np.diag(pen)
    for _ in range(iters):
        z = np.clip(X @ theta, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1.0 - p)
        grad = X.T @ (p - y) + pen * theta
        H = (X.T * w) @ X + Pmat + 1e-8 * np.eye(d)
        step = np.linalg.solve(H, grad)
        theta = theta - step
        if np.max(np.abs(step)) < tol:
            break
    return theta


def _log_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    z = np.clip(X @ theta, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _cv_lambda(records: list[ResolvedRecord], sources: list[str]) -> float:
    """Pick lambda by CV (LOO if small, else 5-fold). Deterministic folds."""
    n = len(records)
    folds = n if n <= 30 else 5
    assign = [i % folds for i in range(n)]  # no shuffle -> reproducible

    best_lam, best_loss = 0.0, math.inf
    for lam in LAMBDA_GRID:
        losses = []
        for f in range(folds):
            train = [records[i] for i in range(n) if assign[i] != f]
            test = [records[i] for i in range(n) if assign[i] == f]
            if not test or not train:
                continue
            Xtr, ytr = _design(train, sources)
            theta = _fit_newton(Xtr, ytr, lam)
            Xte, yte = _design(test, sources)
            losses.append(_log_loss(theta, Xte, yte))
        mean_loss = float(np.mean(losses)) if losses else math.inf
        if mean_loss < best_loss - 1e-9:
            best_lam, best_loss = lam, mean_loss
    return best_lam


def fit_platt(
    records: list[ResolvedRecord],
    hierarchical: bool = True,
    min_source_count: int = 5,
) -> PlattCalibrator:
    """Fit a (hierarchical) Platt calibrator from resolved forecasts.

    Sources with fewer than ``min_source_count`` resolved records don't get their
    own offset (they fall back to the global term); with no eligible sources this
    reduces to a plain global Platt fit. Below ``MIN_RECORDS`` total, returns the
    identity map.
    """
    if len(records) < MIN_RECORDS:
        return PlattCalibrator.identity()

    sources: list[str] = []
    if hierarchical:
        counts: dict[str, int] = {}
        for r in records:
            counts[r.source] = counts.get(r.source, 0) + 1
        sources = sorted(s for s, c in counts.items() if c >= min_source_count)
        # A single source offset is redundant with the global intercept.
        if len(sources) < 2:
            sources = []

    lam = _cv_lambda(records, sources) if sources else 0.0
    X, y = _design(records, sources)
    theta = _fit_newton(X, y, lam)

    a = float(theta[0])
    b = float(theta[1])
    deltas = {s: float(theta[2 + i]) for i, s in enumerate(sources)}
    return PlattCalibrator(a=a, b=b, deltas=deltas, lam=lam, n_fit=len(records))


def brier(records: list[ResolvedRecord], calibrator: PlattCalibrator | None = None) -> float:
    """Mean Brier score of the (optionally calibrated) forecasts in ``records``."""
    if not records:
        return float("nan")
    total = 0.0
    for r in records:
        p = calibrator.transform(r.p, r.source) if calibrator else r.p
        total += (p - r.outcome) ** 2
    return total / len(records)

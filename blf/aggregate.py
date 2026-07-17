"""Multi-trial aggregation (paper Sec. 3 and Sec. C.9).

LLM forecasting is high-variance across runs, so BLF runs K independent trials
and combines them in *logit* space. With trial logits y_k = logit(p_k), mean
ybar = (1/K) sum_k y_k, and a prior mu = logit(prior), the aggregated estimate is
the shrinkage predictor (Eq. 10):

    p_hat = sigmoid( alpha * ybar + (1 - alpha) * mu )

alpha = 1 recovers plain logit averaging; alpha < 1 shrinks toward the prior,
reducing overconfidence when the K trials disagree. We use the paper's exact
data-dependent shrinkage weight (Eq. 11):

    alpha = max( f, 1 - c * s ),   s = std(y_1, ..., y_K)

where s is the per-question sample standard deviation of the trial logits, and
(f, c) are the two global hyperparameters: f in [0, 1] is a floor on alpha
(f = 1 disables shrinkage entirely; f = 0 allows full shrinkage), and c >= 0
controls how aggressively alpha contracts toward that floor as trials disagree.
The paper selects (f, c) by grid search / LOO-CV on ForecastBench (Sec. D.7);
the defaults here are reasonable starting points, not those fitted optima.

The prior is generalized to any probability ``prior`` (the paper uses the crowd
estimate or empirical base rate; falls back to 0.5 when neither is available).
With prior = 0.5, mu = 0 and the predictor reduces to sigmoid(alpha * ybar).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_EPS = 1e-4

# Global shrinkage hyperparameters (paper's f and c; tune per Sec. D.7).
DEFAULT_FLOOR = 0.1   # f: floor on alpha
DEFAULT_C = 0.5       # c: contraction rate in std units of the trial logits


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Aggregation:
    p: float
    alpha: float
    mean_logit: float
    std_logit: float
    trial_ps: list[float]


def aggregate(
    probs: list[float],
    prior: float = 0.5,
    shrink: bool = True,
    f: float = DEFAULT_FLOOR,
    c: float = DEFAULT_C,
) -> Aggregation:
    """Combine K per-trial probabilities via logit-space shrinkage (Eq. 10-11)."""
    if not probs:
        raise ValueError("need at least one trial probability")

    logits = [_logit(p) for p in probs]
    k = len(logits)
    mean_logit = sum(logits) / k

    if k > 1:
        var = sum((x - mean_logit) ** 2 for x in logits) / (k - 1)
        std_logit = math.sqrt(var)
    else:
        std_logit = 0.0

    # alpha = max(f, 1 - c * s)  (Eq. 11); clamped to [0, 1].
    if shrink and k > 1:
        alpha = max(f, 1.0 - c * std_logit)
        alpha = min(max(alpha, 0.0), 1.0)
    else:
        alpha = 1.0

    prior_logit = _logit(prior)
    combined = alpha * mean_logit + (1.0 - alpha) * prior_logit

    return Aggregation(
        p=_sigmoid(combined),
        alpha=alpha,
        mean_logit=mean_logit,
        std_logit=std_logit,
        trial_ps=list(probs),
    )

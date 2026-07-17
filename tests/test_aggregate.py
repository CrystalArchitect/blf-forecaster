"""Unit tests for logit-space aggregation — no network / API needed."""

import math

from blf.aggregate import aggregate


def test_single_trial_passthrough():
    a = aggregate([0.8])
    assert abs(a.p - 0.8) < 1e-3
    assert a.alpha == 1.0


def test_logit_mean_not_arithmetic_mean():
    # Logit averaging of 0.1 and 0.9 returns ~0.5 (symmetric), but is generally
    # distinct from the arithmetic mean for asymmetric inputs.
    a = aggregate([0.6, 0.9], shrink=False)
    arithmetic = (0.6 + 0.9) / 2
    assert abs(a.p - arithmetic) > 1e-3


def test_agreeing_trials_no_shrink_needed():
    # Zero variance -> alpha = 1 -> no shrinkage.
    a = aggregate([0.7, 0.7, 0.7])
    assert a.alpha == 1.0
    assert abs(a.p - 0.7) < 1e-3


def test_disagreeing_trials_shrink_toward_prior():
    # High spread -> alpha < 1 -> pulled toward the 0.5 prior vs. no-shrink.
    spread = [0.02, 0.98, 0.05, 0.95]
    shrunk = aggregate(spread, prior=0.5, shrink=True)
    plain = aggregate(spread, prior=0.5, shrink=False)
    assert shrunk.alpha < 1.0
    assert abs(shrunk.p - 0.5) < abs(plain.p - 0.5)


def test_prior_generalization_matches_paper_at_half():
    # With prior=0.5, logit(prior)=0, so the formula reduces to the paper's
    # p_hat = sigmoid(alpha * mean_logit).
    ps = [0.3, 0.8]
    a = aggregate(ps, prior=0.5, shrink=True)
    logits = [math.log(p / (1 - p)) for p in ps]
    expected = 1 / (1 + math.exp(-a.alpha * (sum(logits) / 2)))
    assert abs(a.p - expected) < 1e-6


def test_alpha_matches_exact_estimator():
    # Eq. 11: alpha = max(f, 1 - c * std(logits)).
    ps = [0.4, 0.7, 0.6]
    a = aggregate(ps, f=0.1, c=0.5)
    logits = [math.log(p / (1 - p)) for p in ps]
    mean = sum(logits) / len(logits)
    std = math.sqrt(sum((x - mean) ** 2 for x in logits) / (len(logits) - 1))
    assert abs(a.std_logit - std) < 1e-9
    assert abs(a.alpha - max(0.1, 1.0 - 0.5 * std)) < 1e-9


def test_floor_f_bounds_alpha_from_below():
    # Very high disagreement drives 1 - c*s negative; alpha floors at f.
    a = aggregate([0.01, 0.99, 0.02, 0.98], f=0.2, c=0.5)
    assert abs(a.alpha - 0.2) < 1e-9


def test_f_equals_one_disables_shrinkage():
    # f = 1 forces alpha = 1 (pure logit averaging) regardless of spread.
    a = aggregate([0.05, 0.95], f=1.0, c=0.5)
    assert a.alpha == 1.0


def test_larger_c_shrinks_more():
    ps = [0.3, 0.75]
    gentle = aggregate(ps, f=0.0, c=0.2)
    aggressive = aggregate(ps, f=0.0, c=0.8)
    assert aggressive.alpha < gentle.alpha
    assert abs(aggressive.p - 0.5) < abs(gentle.p - 0.5)  # more shrink -> closer to prior

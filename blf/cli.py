"""Command-line entry point: `blf "question" [--cutoff ...] [--trials 5]`."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from .forecast import forecast


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(
        prog="blf",
        description="Bayesian Linguistic Forecaster (v1): belief-state agent + "
        "multi-trial logit aggregation.",
    )
    ap.add_argument("question", help="The binary (YES/NO) forecasting question.")
    ap.add_argument("--cutoff", default=None, help="Knowledge cutoff date, YYYY-MM-DD.")
    ap.add_argument("--trials", "-k", type=int, default=5, help="Number of trials (K).")
    ap.add_argument("--max-steps", type=int, default=10, help="Tool steps per trial.")
    ap.add_argument("--prior", type=float, default=0.5, help="Shrinkage prior.")
    ap.add_argument("--no-shrink", action="store_true", help="Disable shrinkage.")
    ap.add_argument("--source", default=None, help="Source label (for calibration).")
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply the fitted calibrator (~/.blf/calibrator.json). Fit one with `blf-calibrate fit`.",
    )
    ap.add_argument(
        "--timeseries",
        action="store_true",
        help="Enable numeric time-series tools (history/model/combo fetch via Yahoo/FRED) "
        "for questions of the form P(value > threshold at date).",
    )
    ap.add_argument("--verbose", "-v", action="store_true", help="Print per-trial traces.")
    args = ap.parse_args()

    fc = forecast(
        args.question,
        cutoff=args.cutoff,
        trials=args.trials,
        max_steps=args.max_steps,
        prior=args.prior,
        shrink=not args.no_shrink,
        source=args.source,
        calibrate=args.calibrate,
        enable_timeseries=args.timeseries,
    )

    print(f"\n=== BLF forecast ===")
    print(f"Question: {args.question}")
    if fc.calibrated:
        print(f"\nP(YES) = {fc.p:.3f}   (raw {fc.raw_p:.3f} -> calibrated)\n")
    else:
        print(f"\nP(YES) = {fc.p:.3f}\n")
        if args.calibrate:
            print("(no fitted calibrator found; showing raw estimate)\n")

    a = fc.aggregation
    print(
        f"trials={len(fc.trials)}  per-trial p={[round(p, 3) for p in a.trial_ps]}  "
        f"shrink alpha={a.alpha:.3f}  logit-std={a.std_logit:.3f}"
    )

    if args.verbose:
        for i, t in enumerate(fc.trials):
            tag = " (forced)" if t.forced else ""
            print(f"\n--- Trial {i} -> {t.p:.3f}{tag}, {t.steps} steps ---")
            for line in t.trace:
                print("  " + line)
            if t.belief.key_evidence_for:
                print("  evidence for : " + "; ".join(t.belief.key_evidence_for))
            if t.belief.key_evidence_against:
                print("  evidence against: " + "; ".join(t.belief.key_evidence_against))

    return 0


if __name__ == "__main__":
    sys.exit(main())

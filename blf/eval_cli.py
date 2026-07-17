"""`blf-eval` — score BLF over a question set with the Brier Index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .eval import blf_predictor, run_eval
from .forecastbench import load_forecastbench, load_questions


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(
        prog="blf-eval",
        description="Evaluate BLF on a question set and report the Brier Index "
        "(Overall / Market / Dataset).",
    )
    ap.add_argument("questions", help="Question set JSON (internal schema, or FB with --forecastbench).")
    ap.add_argument("--forecastbench", metavar="RESOLUTIONS", default=None,
                    help="Treat `questions` as a ForecastBench question set (path or URL) and score "
                    "against this resolution file (path or URL).")
    ap.add_argument("--sources", default=None,
                    help="Comma-separated source filter, e.g. 'fred,metaculus' (ForecastBench only).")
    ap.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    ap.add_argument("--trials", "-k", type=int, default=5, help="Trials per question (K).")
    ap.add_argument("--max-steps", type=int, default=10, help="Tool steps per trial.")
    ap.add_argument("--calibrate", action="store_true", help="Apply the fitted calibrator.")
    ap.add_argument("--out", default=None, help="Write per-question predictions to this JSONL file.")
    args = ap.parse_args()

    source_filter = set(args.sources.split(",")) if args.sources else None
    if args.forecastbench:
        questions = load_forecastbench(args.questions, args.forecastbench, sources=source_filter)
    else:
        questions = load_questions(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        print("no questions to score.", file=sys.stderr)
        return 1

    out_f = open(args.out, "w") if args.out else None

    def _stream(res):
        tag = "mkt" if res.is_market else "dat"
        print(f"  [{tag}] {res.id[:40]:<40} p={res.p:.3f} o={res.outcome:.0f} brier={res.brier:.3f}")
        if out_f:
            out_f.write(json.dumps({
                "id": res.id, "source": res.source, "is_market": res.is_market,
                "p": res.p, "outcome": res.outcome, "brier": res.brier, "horizon": res.horizon,
                "trial_ps": res.trial_ps, "prior": res.prior,
            }) + "\n")
            out_f.flush()

    print(f"Scoring {len(questions)} question(s) with K={args.trials} trials each...\n")
    predict = blf_predictor(trials=args.trials, max_steps=args.max_steps, calibrate=args.calibrate)
    report = run_eval(questions, predict_fn=predict, on_result=_stream)

    if out_f:
        out_f.close()
        print(f"\npredictions -> {Path(args.out).resolve()}")

    print("\n" + report.format())
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""`blf-shrinkage` — fit the aggregation shrinkage hyperparameters (f, c).

Typical flow:
    # 1. run the eval, dumping per-trial forecasts (needs resolved outcomes)
    blf-eval <questions> --forecastbench <resolutions> --limit 100 --out preds.jsonl
    # 2. grid-search (f, c) offline from the cached trials, save ~/.blf/shrinkage.json
    blf-shrinkage fit preds.jsonl
    # 3. forecasts now auto-load the fitted params
    blf "..."

Set BLF_DATA_DIR to relocate the saved params (default ~/.blf).
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from .shrinkage import ShrinkageParams, fit_shrinkage, load_trial_records


def _cmd_fit(args) -> int:
    records = load_trial_records(args.predictions)
    if not records:
        print(
            "no records with per-trial forecasts found. Generate them with "
            "`blf-eval ... --out preds.jsonl` (needs resolved questions).",
            file=sys.stderr,
        )
        return 1

    params, report = fit_shrinkage(records)
    path = params.save(args.out)

    print(f"records with trial data: {report.get('n', len(records))}")
    if report["fitted"]:
        print(f"fitted f={params.f}  c={params.c}  ->  {path}")
        print(
            f"  mean Brier: default {report['brier_default']:.4f} -> "
            f"fit {report['brier_fit']:.4f}"
        )
        print(
            f"  Brier Index: default {report['bi_default']:.1f} -> "
            f"fit {report['bi_fit']:.1f}"
        )
    else:
        print(f"kept defaults f={params.f} c={params.c} ({report['reason']}) -> {path}")
    return 0


def _cmd_show(args) -> int:
    sp = ShrinkageParams.load()
    if sp is None:
        print("no fitted shrinkage params yet (run `blf-shrinkage fit`). Using code "
              "defaults at forecast time.")
    else:
        print(f"shrinkage: f={sp.f}  c={sp.c}  (fit on {sp.n_fit} records)")
    return 0


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="blf-shrinkage", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="Grid-search (f, c) from a predictions JSONL.")
    f.add_argument("predictions", help="JSONL from `blf-eval --out` (must contain trial_ps).")
    f.add_argument("--out", default=None, help="Output path (default ~/.blf/shrinkage.json).")
    f.set_defaults(func=_cmd_fit)

    s = sub.add_parser("show", help="Show the current fitted params.")
    s.set_defaults(func=_cmd_show)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

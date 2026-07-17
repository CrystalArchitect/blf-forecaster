"""`blf-calibrate` — manage resolved forecasts and fit the Platt calibrator.

Workflow:
    blf-calibrate add --p 0.72 --source polymarket --outcome 1 --question "..."
    blf-calibrate show
    blf-calibrate fit          # fits ~/.blf/calibrator.json from ~/.blf/resolved.jsonl

The calibrator is then applied at forecast time via `blf --calibrate`.
Set BLF_DATA_DIR to relocate the store/calibrator (default ~/.blf).
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from .calibration import (
    PlattCalibrator,
    ResolvedRecord,
    ResolvedStore,
    brier,
    fit_platt,
)


def _cmd_add(args) -> int:
    store = ResolvedStore()
    store.add(
        ResolvedRecord(
            p=args.p, source=args.source, outcome=args.outcome, question=args.question or ""
        )
    )
    print(f"recorded: p={args.p} source={args.source!r} outcome={args.outcome} -> {store.path}")
    return 0


def _cmd_show(args) -> int:
    records = ResolvedStore().load()
    print(f"{len(records)} resolved record(s)")
    by_source: dict[str, list] = {}
    for r in records:
        by_source.setdefault(r.source, []).append(r)
    for s, rs in sorted(by_source.items()):
        yes = sum(r.outcome for r in rs)
        print(f"  {s or '(none)'}: n={len(rs)}  YES-rate={yes / len(rs):.2f}")
    cal = PlattCalibrator.load()
    if cal:
        print(
            f"\ncalibrator: a={cal.a:.3f} b={cal.b:+.3f} lambda={cal.lam} "
            f"(fit on {cal.n_fit})"
        )
        for src, d in sorted(cal.deltas.items()):
            print(f"  delta[{src}] = {d:+.3f}")
    else:
        print("\nno fitted calibrator yet (run `blf-calibrate fit`).")
    return 0


def _cmd_fit(args) -> int:
    records = ResolvedStore().load()
    if not records:
        print("no resolved records; add some with `blf-calibrate add`.", file=sys.stderr)
        return 1
    cal = fit_platt(records, hierarchical=not args.global_only)
    path = cal.save(args.out)

    raw = brier(records, None)
    calibrated = brier(records, cal)
    print(f"fit on {cal.n_fit} records -> {path}")
    print(f"  a={cal.a:.3f}  b={cal.b:+.3f}  lambda={cal.lam}")
    for src, d in sorted(cal.deltas.items()):
        print(f"  delta[{src}] = {d:+.3f}")
    if cal.is_identity:
        print("  (identity — too few records to trust a fit)")
    print(f"\nin-sample Brier: raw={raw:.4f}  calibrated={calibrated:.4f}")
    return 0


def main() -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="blf-calibrate", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Record a resolved forecast.")
    a.add_argument("--p", type=float, required=True, help="Raw forecast that was made.")
    a.add_argument("--source", default="manual", help="Source label.")
    a.add_argument("--outcome", type=int, choices=(0, 1), required=True, help="1=YES, 0=NO.")
    a.add_argument("--question", default=None, help="Optional note.")
    a.set_defaults(func=_cmd_add)

    s = sub.add_parser("show", help="Summarize the store and current calibrator.")
    s.set_defaults(func=_cmd_show)

    f = sub.add_parser("fit", help="Fit the Platt calibrator from the store.")
    f.add_argument("--global-only", action="store_true", help="Disable per-source offsets.")
    f.add_argument("--out", default=None, help="Output path (default ~/.blf/calibrator.json).")
    f.set_defaults(func=_cmd_fit)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

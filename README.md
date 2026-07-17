# BLF — Bayesian Linguistic Forecaster (v1)

A from-scratch, training-free reimplementation of the two novel ideas in
Kevin Murphy, *"Agentic Forecasting using Sequential Bayesian Updating of
Linguistic Beliefs"* ([arXiv:2604.18576](https://arxiv.org/abs/2604.18576)):

1. **Linguistic belief-state agent loop** (Algorithm 1, Sec. C.1–C.3). Instead
   of dumping all retrieved evidence into an ever-growing context, the agent
   keeps a compact, semi-structured belief — `{p, confidence, evidence_for,
   evidence_against, open_questions}` — that the LLM *rewrites* on every step.
   The belief update rides inside the tool call (an `updated_belief` argument),
   so one LLM forward pass produces both the next action and the new belief.
2. **Multi-trial logit aggregation** (Sec. 3, Eq. 10-11). Run K=5 independent
   rollouts and combine them in logit space with the paper's shrinkage predictor
   `p_hat = sigmoid( alpha*ybar + (1-alpha)*mu )`, `alpha = max(f, 1-c*std)`,
   pulling toward the prior when trials disagree.
3. **Hierarchical Platt calibration** (Sec. C.11). Map raw forecasts through
   `p_cal = sigmoid( a * logit(p) + b + delta_source )`, with per-source
   intercept offsets `delta_s` L2-regularized (strength chosen by CV). Needs a
   backlog of *resolved* forecasts to fit on (see below); falls back to identity
   when data is scarce.

The full four-layer date-leakage defense (Sec. B) is out of scope; a best-effort
`cutoff` filter (Brave `freshness` / Perplexity date filter) is included for
backtesting.

## Install

```bash
cd blf-forecaster
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env      # then fill in ANTHROPIC_API_KEY and BRAVE_API_KEY
```

## Use

```bash
blf "Will SpaceX complete an orbital Starship flight before 2027-01-01?" -k 5 -v
```

Or from Python:

```python
from blf import forecast

fc = forecast("Will X happen before date D?", trials=5)
print(fc.p)                       # aggregated P(YES)
print(fc.aggregation.trial_ps)    # per-trial probabilities
print(fc.aggregation.alpha)       # shrinkage weight (1.0 = trials agreed)
```

## Calibration

Calibration learns to correct systematic over/under-confidence from a history of
*resolved* forecasts. Log outcomes as they resolve, then fit:

```bash
blf-calibrate add --p 0.72 --source polymarket --outcome 1 --question "..."
blf-calibrate add --p 0.40 --source fred --outcome 0
blf-calibrate show                 # store summary + current calibrator
blf-calibrate fit                  # writes ~/.blf/calibrator.json, reports Brier gain
```

Then apply it at forecast time (raw estimate is still reported):

```bash
blf "Will X happen before D?" --source polymarket --calibrate
# P(YES) = 0.71   (raw 0.83 -> calibrated)
```

Store/calibrator live under `~/.blf` (override with `BLF_DATA_DIR`). Below ~10
records, or with fewer than 2 well-populated sources, the fit degrades
gracefully (global-only, or identity). Force global-only with
`blf-calibrate fit --global-only`.

## Time-series questions

For numeric questions of the form `P(value > threshold at date)` (the paper's
dataset questions), enable the source-gated time-series tools with `--timeseries`:

```bash
blf "Will the S&P 500 (SPY) close above 800 on 2026-09-30?" \
    --timeseries --cutoff 2026-07-15 -v
```

This exposes three extra tools (paper Table 5), backed by **keyless** endpoints:

| Tool | Does |
|------|------|
| `history_fetch` | recent value history (trailing window, cutoff-enforced) |
| `model_fetch`   | `P(value > threshold at r)` via random-walk-with-drift |
| `combo_fetch`   | both in one call (preferred first action) |

Sources: **`yahoo`** (prices/indices/FX/crypto — e.g. `SPY`, `^GSPC`, `BTC-USD`,
`EURUSD=X`, `^TNX`) and **`fred`** (economic levels — e.g. `UNRATE`, `CPIAUCSL`).
The agent extracts the series id, threshold, and resolution date from the
question. The `cutoff` date is enforced at fetch time (leakage guard). The tools
are off by default so ordinary judgemental questions aren't cluttered with them.

## Evaluation (ForecastBench)

Score BLF on a question set with the paper's **Brier Index**
(`BI = 100·(1 − √mean_Brier)`; higher is better, a 0.5 forecaster scores 50).
Following ForecastBench, the **Overall** score is the unweighted average of the
Market and Dataset category scores.

Run against the live ForecastBench datasets straight from GitHub (paths or URLs):

```bash
BASE=https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets
blf-eval \
  $BASE/question_sets/2026-06-07-llm.json \
  --forecastbench $BASE/resolution_sets/2026-06-07_resolution_set.json \
  --sources metaculus,polymarket,manifold \
  --limit 20 --trials 5 --out preds.jsonl
```

The adapter drives expansion from the **resolution set** (the ground truth): each
resolved, non-combo entry becomes one scored row, with the question templated for
that horizon, `cutoff` set to the forecast-due date (leakage guard), and the
freeze value injected into the prompt (the paper's crowd signal for market
questions, reference value `v_q` for dataset questions). `--sources` filters by
source; `--limit` and `--trials` control cost.

Or score a custom set in the internal schema (see `data/sample_questions.json`):

```bash
blf-eval data/sample_questions.json --trials 5
```

Programmatically, `blf.run_eval(questions, predict_fn)` takes any
`predict_fn(EvalQuestion) -> float`, so scoring is testable without an LLM.

### Fitting the shrinkage hyperparameters

The aggregation `f`/`c` (Eq. 11) can be fit by grid search the way the paper does
(Sec. D.7). Because aggregation is a pure function of the cached per-trial
forecasts, you run the LLM rollouts **once** and replay the grid instantly:

```bash
# 1. eval with --out dumps each question's per-trial forecasts + outcome
blf-eval $QSET --forecastbench $RSET --limit 150 --out preds.jsonl
# 2. grid-search (f, c) offline, save ~/.blf/shrinkage.json, report the BI gain
blf-shrinkage fit preds.jsonl
# 3. forecasts auto-load the fitted params (override with forecast(shrink_floor=, shrink_c=))
blf "..."
blf-shrinkage show
```

The fitter minimizes mean Brier over the grid; with fewer than 20 usable records
it keeps the code defaults. Note the eval that produces `preds.jsonl` should run
**without** `--calibrate`, so the fit sees the raw aggregated forecasts.

## Layout

| File | Role |
|------|------|
| `blf/belief.py`    | `BeliefState` + the `updated_belief` tool-arg schema (Sec. C.1) |
| `blf/agent.py`     | single-trial agent loop, tool defs, system prompt (Algorithm 1) |
| `blf/search.py`    | Brave search + lazy-fetch file store (progressive disclosure, Sec. C.3) |
| `blf/aggregate.py` | logit-space K-trial aggregation with shrinkage (Sec. 3) |
| `blf/timeseries.py` | history / model / combo fetch + random-walk model (Sec. C.4) |
| `blf/calibration.py` | hierarchical Platt calibration + resolved-forecast store (Sec. C.11) |
| `blf/eval.py`      | Brier Index scoring + eval runner |
| `blf/forecastbench.py` | ForecastBench question/resolution adapter |
| `blf/shrinkage.py` | grid-search fit of the (f, c) shrinkage hyperparameters (Sec. D.7) |
| `blf/forecast.py`  | top-level: run K trials concurrently, aggregate, calibrate |
| `blf/llm.py`       | OpenAI-compatible wrappers (main agent + cheap summarizer sub-LLM) |
| `blf/cli.py`       | `blf` forecast command |
| `blf/calibrate_cli.py` | `blf-calibrate` add / show / fit command |
| `blf/eval_cli.py`  | `blf-eval` benchmark command |

## Tests

```bash
pip install pytest && pytest        # aggregation tests run offline (no API needed)
```

## Deviations from the paper

- Shrinkage uses the paper's exact estimator `alpha = max(f, 1 - c*std(logits))`
  (Eq. 11); `f`/`c` default to reasonable values but can be grid-searched on your
  own resolved data with `blf-shrinkage fit` (Sec. D.7). The paper's fitted
  optima are on their private ForecastBench tranches.
- Calibration selects the L2 strength by cross-validation (LOO when data is
  scarce, else 5-fold) rather than strictly LOO, and regularizes only the
  per-source offsets.
- LLM backend is any OpenAI-compatible endpoint (OpenRouter / gateway / Groq);
  search is Perplexity or Brave. The paper uses Gemini + Brave but is
  engine-agnostic. Page text is fetched lazily on `read_files`.
- Time-series `model_fetch` uses a random-walk-with-drift estimate (a transparent
  stand-in for the paper's "simple statistical models", Appendix I). Data comes
  from Yahoo Finance (the paper's yfinance) and FRED; the paper also uses
  DBnomics / Wikipedia snapshots, which are not wired up.
- The ForecastBench harness scores real question/resolution sets, but does not
  reproduce the paper's tranche construction, difficulty-adjusted BI, or the
  four-layer leakage defense (best-effort `cutoff` filter only).

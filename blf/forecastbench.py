"""Loaders that turn question files into the internal `EvalQuestion` schema.

Two entry points:
  - `load_questions(path)`     — a JSON list already in the internal schema
                                 (fields matching EvalQuestion).
  - `load_forecastbench(q, r)` — adapter for the real ForecastBench question-set
                                 + resolution-set files (paths or raw URLs).

ForecastBench schema (github.com/forecastingresearch/forecastbench-datasets):
  question set:  {forecast_due_date, question_set, questions: [
                   {id, source, question, resolution_criteria, background, url,
                    freeze_datetime, freeze_datetime_value, resolution_dates}]}
                 Dataset questions template `{resolution_date}` / `{forecast_due_date}`
                 into `question` and list several `resolution_dates`; market
                 questions have resolution_dates == "N/A".
  resolution set:{forecast_due_date, question_set, resolutions: [
                   {id, source, direction, resolution_date, resolved_to, resolved}]}

We drive expansion from the resolution set (the ground truth): each resolved,
single (non-combo) entry becomes one EvalQuestion, with the matching question
text templated for that horizon. Scoring is Brier = (p - resolved_to)^2.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from .eval import EvalQuestion

# Prediction-market sources (a crowd/market estimate exists); the rest are
# dataset / time-series questions.
MARKET_SOURCES = {"metaculus", "polymarket", "manifold", "infer", "rfi"}
# Dataset sources our numeric time-series tools can actually fetch.
TIMESERIES_SOURCES = {"yfinance", "yahoo", "fred"}


def _read_json(path_or_url: str | Path) -> dict:
    s = str(path_or_url)
    if s.startswith("http://") or s.startswith("https://"):
        resp = httpx.get(s, timeout=60.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.json()
    return json.loads(Path(path_or_url).read_text())


def load_questions(path: str | Path) -> list[EvalQuestion]:
    """Load a JSON array already in the internal schema."""
    data = _read_json(path)
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    out = []
    for d in data:
        src = str(d.get("source", ""))
        out.append(
            EvalQuestion(
                id=str(d["id"]),
                text=str(d["text"]),
                source=src,
                cutoff=str(d["cutoff"]),
                resolution_date=str(d.get("resolution_date", d["cutoff"])),
                outcome=float(d["outcome"]),
                is_market=bool(d.get("is_market", src in MARKET_SOURCES)),
                timeseries=bool(d.get("timeseries", src in TIMESERIES_SOURCES)),
                horizon=d.get("horizon"),
            )
        )
    return out


def _date_part(dt: str) -> str:
    return str(dt).split("T")[0] if dt else ""


def load_forecastbench(
    questions_path: str | Path,
    resolutions_path: str | Path,
    sources: set[str] | None = None,
) -> list[EvalQuestion]:
    """Adapter for the official ForecastBench question-set + resolution files.

    ``sources`` optionally restricts to a subset (e.g. {"fred", "metaculus"}).
    """
    qset = _read_json(questions_path)
    rset = _read_json(resolutions_path)

    due_date = _date_part(qset.get("forecast_due_date", "")) or _date_part(
        rset.get("forecast_due_date", "")
    )
    by_id = {str(q["id"]): q for q in qset.get("questions", [])}

    out: list[EvalQuestion] = []
    for r in rset.get("resolutions", []):
        if not r.get("resolved", False) or r.get("resolved_to") is None:
            continue
        if r.get("direction") is not None:  # skip combo questions
            continue
        qid = r.get("id")
        if not isinstance(qid, str) or qid not in by_id:
            continue
        q = by_id[qid]
        src = str(q.get("source", r.get("source", "")))
        if sources is not None and src not in sources:
            continue

        res_date = _date_part(r["resolution_date"])
        text = (
            str(q.get("question", ""))
            .replace("{resolution_date}", res_date)
            .replace("{forecast_due_date}", due_date)
        )

        # Inject the freeze value: the crowd/market estimate (market questions) or
        # the reference value v_q (dataset questions). This is the paper's crowd
        # signal — the single strongest input for market questions.
        is_market = src in MARKET_SOURCES
        fv = q.get("freeze_datetime_value")
        fdate = _date_part(q.get("freeze_datetime", "")) or due_date
        if fv not in (None, "", "N/A"):
            if is_market:
                text += f"\n\n(As of {fdate}, the crowd/market estimate was {fv}.)"
            else:
                text += f"\n\n(The reference value on {due_date} was {fv}.)"

        criteria = str(q.get("resolution_criteria", "")).strip()
        if criteria and criteria != "N/A":
            text += f"\n\nResolution criteria: {criteria}"

        outcome = float(r["resolved_to"])
        outcome = min(max(outcome, 0.0), 1.0)

        out.append(
            EvalQuestion(
                id=f"{qid}@{res_date}",
                text=text,
                source=src,
                cutoff=due_date,
                resolution_date=res_date,
                outcome=outcome,
                is_market=is_market,
                timeseries=src in TIMESERIES_SOURCES,
                horizon=res_date,
            )
        )
    return out

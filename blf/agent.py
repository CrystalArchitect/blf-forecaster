"""The BLF agent loop (paper Algorithm 1 & 2, Sec. C.2).

A single trial:

    b_0 <- initial belief (p = 0.5)
    for t in 1..T_max:
        (a_t, b_t) <- LLM(history)        # action + updated belief, one call
        if a_t == submit: return b_t.p     # clamped to [0.05, 0.95]
        o_t <- Env(a_t)                    # execute tool
        history <- history + (a_t, o_t)    # deterministic concatenation
    # forced submit at T_max

Tools (subset of Table 4 sufficient for judgemental questions): browse_web,
read_files, submit. Time-series tools are left for a later version.

Wire format is OpenAI chat-completions function calling, so any OpenAI-compatible
backend (OpenRouter / gateway / Groq) works via blf.llm.LLM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .belief import UPDATED_BELIEF_SCHEMA, BeliefState
from .llm import LLM
from .search import FileStore, fetch_page_text
from .timeseries import fetch_series, model_estimate, trim_window

# Submitted probabilities are clamped to bound worst-case Brier loss when the
# agent is confidently wrong (paper Sec. C.2 / "Agent loop and tools").
CLAMP_LO, CLAMP_HI = 0.05, 0.95

SYSTEM_PROMPT = """\
You are BLF, a careful superforecaster estimating the probability that a binary \
question resolves YES.

You operate as an iterative loop. On EVERY turn you must call exactly one tool, \
and every tool call must include an `updated_belief` object — your rewritten \
belief after weighing all evidence gathered so far. Treat `updated_belief` as a \
running scratchpad, not the raw context: keep it concise and current.

Method:
- Start from a base rate. Reason about it explicitly in update_reasoning.
- Use `browse_web` to gather evidence; issue focused queries, one angle at a time.
- Use `read_files` to pull the full text of the most promising search hits when a \
snippet is not enough. Do not read everything — drill into the best leads.
- Update your probability incrementally as evidence arrives (prior + evidence -> \
posterior). Avoid overconfidence; move in proportion to the strength of evidence.
- When further searching is unlikely to change your estimate, call `submit`.

You have at most {max_steps} steps. Calibration matters more than boldness: a \
well-reasoned 0.6 beats a reckless 0.95.
"""


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_TS_SOURCE = {
    "type": "string",
    "enum": ["yahoo", "fred"],
    "description": "Data source: 'yahoo' for prices/indices/FX/crypto, 'fred' for economic levels.",
}
_TS_SERIES_ID = {
    "type": "string",
    "description": "Series id — Yahoo symbol (e.g. 'SPY', 'AAPL', '^GSPC', 'BTC-USD', "
    "'EURUSD=X', '^TNX') or FRED code (e.g. 'UNRATE', 'CPIAUCSL').",
}


def _timeseries_tool_defs() -> list[dict]:
    """The source-gated numeric tools (paper Table 5). Only offer these for
    time-series questions of the form P(Y(r) > v)."""
    return [
        _fn(
            "history_fetch",
            "Fetch the recent value history of a numeric series (up to the cutoff "
            "date). Use for judging trend/level before estimating a threshold prob.",
            {
                "source": _TS_SOURCE,
                "series_id": _TS_SERIES_ID,
                "window_days": {"type": "integer", "description": "Trailing window (default 60)."},
                "updated_belief": UPDATED_BELIEF_SCHEMA,
            },
            ["source", "series_id", "updated_belief"],
        ),
        _fn(
            "model_fetch",
            "Get a statistical estimate of P(series value at resolution_date > "
            "threshold), via a random-walk-with-drift model over recent history.",
            {
                "source": _TS_SOURCE,
                "series_id": _TS_SERIES_ID,
                "threshold": {"type": "number", "description": "The value v to exceed."},
                "resolution_date": {"type": "string", "description": "Resolution date r (YYYY-MM-DD)."},
                "window_days": {"type": "integer", "description": "Trailing window (default 60)."},
                "updated_belief": UPDATED_BELIEF_SCHEMA,
            },
            ["source", "series_id", "threshold", "resolution_date", "updated_belief"],
        ),
        _fn(
            "combo_fetch",
            "Fetch recent history AND the model-based P(value > threshold) in one "
            "call. Preferred first action for FRED/price time-series questions.",
            {
                "source": _TS_SOURCE,
                "series_id": _TS_SERIES_ID,
                "threshold": {"type": "number", "description": "The value v to exceed."},
                "resolution_date": {"type": "string", "description": "Resolution date r (YYYY-MM-DD)."},
                "window_days": {"type": "integer", "description": "Trailing window (default 60)."},
                "updated_belief": UPDATED_BELIEF_SCHEMA,
            },
            ["source", "series_id", "threshold", "resolution_date", "updated_belief"],
        ),
    ]


def _tool_defs(enable_timeseries: bool = False) -> list[dict]:
    tools = [
        _fn(
            "browse_web",
            "Search the web. Returns ~10 titled snippets with ids you can later "
            "pass to read_files for full text.",
            {"query": {"type": "string"}, "updated_belief": UPDATED_BELIEF_SCHEMA},
            ["query", "updated_belief"],
        ),
        _fn(
            "read_files",
            "Fetch and summarize the full page text for one or more search-result "
            "ids returned by a previous browse_web call.",
            {
                "ids": {"type": "array", "items": {"type": "integer"}},
                "updated_belief": UPDATED_BELIEF_SCHEMA,
            },
            ["ids", "updated_belief"],
        ),
        _fn(
            "submit",
            "Submit your final probability that the question resolves YES and end "
            "the loop.",
            {
                "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "updated_belief": UPDATED_BELIEF_SCHEMA,
            },
            ["probability", "updated_belief"],
        ),
    ]
    if enable_timeseries:
        tools.extend(_timeseries_tool_defs())
    return tools


@dataclass
class TrialResult:
    p: float
    steps: int
    forced: bool
    belief: BeliefState
    trace: list[str] = field(default_factory=list)


def _assistant_echo(call) -> dict:
    """Reconstruct the assistant message carrying just the tool call we handle,
    so OpenAI's one-response-per-tool-call invariant holds."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
        ],
    }


def run_trial(
    question: str,
    llm: LLM,
    search,
    cutoff: str | None = None,
    max_steps: int = 10,
    enable_timeseries: bool = False,
) -> TrialResult:
    """Execute one BLF rollout (Algorithm 1) and return its submitted probability."""
    store = FileStore()
    system = SYSTEM_PROMPT.format(max_steps=max_steps)
    if enable_timeseries:
        system += (
            "\n\nThis is a NUMERIC time-series question of the form P(value > "
            "threshold at a date). Prefer combo_fetch first to get the recent "
            "history and a model estimate, then adjust with web evidence."
        )

    user_text = f"QUESTION:\n{question}"
    if cutoff:
        user_text += (
            f"\n\nKNOWLEDGE CUTOFF: {cutoff}. Only use information available on or "
            "before this date."
        )
    messages: list[dict] = [{"role": "user", "content": user_text}]

    belief = BeliefState.initial()
    tools = _tool_defs(enable_timeseries=enable_timeseries)
    trace: list[str] = []

    for step in range(1, max_steps + 1):
        resp = llm.act(system, messages, tools)
        msg = resp.choices[0].message
        calls = msg.tool_calls or []

        if not calls:
            # Model declined to act; nudge and retry within the step budget.
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content": "You must call a tool. Continue."})
            continue

        call = calls[0]
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}

        if isinstance(args.get("updated_belief"), dict):
            belief = BeliefState.from_tool_arg(args["updated_belief"])
        trace.append(f"[{step}] {name}: {belief.summary()}")

        if name == "submit":
            p = float(args.get("probability", belief.p))
            p = min(max(p, CLAMP_LO), CLAMP_HI)
            return TrialResult(p=p, steps=step, forced=False, belief=belief, trace=trace)

        observation = _execute_tool(name, args, question, store, search, llm, cutoff)

        messages.append(_assistant_echo(call))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": observation})

    # Forced submit at max_steps (Algorithm 1, line 13).
    p = min(max(belief.p, CLAMP_LO), CLAMP_HI)
    trace.append(f"[forced submit] {belief.summary()}")
    return TrialResult(p=p, steps=max_steps, forced=True, belief=belief, trace=trace)


def _execute_tool(name, args, question, store, search, llm, cutoff) -> str:
    if name == "browse_web":
        query = str(args.get("query", "")).strip()
        hits = search.search(query, count=10, cutoff=cutoff)
        if not hits:
            return f"No results for query: {query!r}"
        lines = [f"Results for {query!r}:"]
        for h in hits:
            r = store.add(h["title"], h["url"], h["snippet"], query)
            lines.append(f"  id={r.id} | {r.title}\n    {r.url}\n    {r.snippet}")
        lines.append("\nUse read_files with the ids of the most promising results for full text.")
        return "\n".join(lines)

    if name == "read_files":
        ids = args.get("ids", []) or []
        chunks = []
        for rid in ids:
            r = store.get(int(rid))
            if r is None:
                chunks.append(f"id={rid}: (no such result)")
                continue
            page = fetch_page_text(r.url)
            summary = llm.summarize(question, page)
            chunks.append(f"id={rid} | {r.title} ({r.url})\n{summary}")
        return "\n\n".join(chunks) if chunks else "No ids provided."

    if name in ("history_fetch", "model_fetch", "combo_fetch"):
        return _execute_timeseries(name, args, cutoff)

    return f"[unknown tool: {name}]"


def _execute_timeseries(name, args, cutoff) -> str:
    source = str(args.get("source", "")).strip()
    series_id = str(args.get("series_id", "")).strip()
    window = int(args.get("window_days") or 60)

    try:
        raw = fetch_series(source, series_id, cutoff=cutoff, window_days=window)
    except Exception as e:  # noqa: BLE001 — remote CSV is flaky/unknown ids
        return f"[time-series fetch failed for {source}:{series_id}: {e}]"

    series = trim_window(raw, cutoff, window)
    if len(series) == 0:
        return f"[no data for {source}:{series_id} on/before {cutoff or 'now'}]"

    out: list[str] = []
    if name in ("history_fetch", "combo_fetch"):
        last_d, last_v = series.last
        # Show a compact tail so we don't flood context.
        tail = list(zip(series.dates, series.values))[-12:]
        pts = ", ".join(f"{d}={v:g}" for d, v in tail)
        out.append(
            f"{source}:{series_id} — {len(series)} pts in last {window}d; "
            f"latest {last_d}={last_v:g}\n  {pts}"
        )

    if name in ("model_fetch", "combo_fetch"):
        threshold = float(args.get("threshold"))
        resolution_date = str(args.get("resolution_date", "")).strip()
        est = model_estimate(series, threshold, resolution_date)
        if "error" in est:
            out.append(f"model estimate: {est['error']}")
        else:
            out.append(
                "model estimate P(value > {t:g} on {r}) = {p:.3f}  "
                "[{m}; last {lv:g} on {ld}, drift/step {mu:+.4g}, vol/step {sd:.4g}, "
                "horizon {h}d]".format(
                    t=threshold, r=resolution_date, p=est["p"], m=est["model"],
                    lv=est["last_value"], ld=est["last_date"], mu=est["drift_per_step"],
                    sd=est["vol_per_step"], h=est["horizon_days"],
                )
            )
    return "\n".join(out)

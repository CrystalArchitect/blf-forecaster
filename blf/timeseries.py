"""Time-series tools (paper Sec. C.4-C.5, Table 5).

ForecastBench's dataset questions are binary reductions of numeric forecasts:

    p_hat = P( Y_q(r) > v_q | data(<= f) )                                (Eq. 1)

i.e. "will the series exceed threshold v at resolution date r, given data up to
the forecast date f?". BLF gives the agent tools to fetch the series history and
a model-based estimate:

    history-fetch (h) : recent value history up to the cutoff (paper caps ~60d)
    model-fetch   (m) : a statistical estimate of P(Y(r) > v)
    combo-fetch   (c) : both at once (paper uses this for yfinance/FRED)

We back these with two keyless public endpoints so no extra API keys are needed:
Yahoo Finance's chart JSON (prices/indices/FX/crypto — the paper's "yfinance"
source) and FRED's `fredgraph.csv` (economic levels). The date cutoff is enforced
both at fetch time and when trimming history, so the agent cannot see past the
forecast date (leakage guard). (FRED may be unreachable from restricted sandboxes
but works from a normal network; Yahoo is the primary source.)

The `model-fetch` estimate is a random-walk-with-drift: we estimate per-step
drift and volatility from the windowed history (log-returns for positive
price-like series, first differences for levels), project to the horizon, and
read off the Gaussian tail probability. This is a transparent stand-in for the
paper's "simple statistical models" (Appendix I).
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import httpx
import numpy as np

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"

_EPS = 1e-4


@dataclass
class Series:
    dates: list[str]      # ISO 'YYYY-MM-DD', ascending
    values: list[float]

    def __len__(self) -> int:
        return len(self.values)

    @property
    def last(self) -> tuple[str, float] | None:
        return (self.dates[-1], self.values[-1]) if self.values else None


# --------------------------------------------------------------------------- #
# Parsing (pure — unit-testable without network)
# --------------------------------------------------------------------------- #
def parse_fred_csv(text: str) -> Series:
    """FRED fredgraph.csv: 'observation_date,SERIES' with '.' for missing."""
    rows = list(csv.reader(io.StringIO(text)))
    dates, values = [], []
    for row in rows[1:]:  # skip header
        if len(row) < 2:
            continue
        d, v = row[0].strip(), row[1].strip()
        if not d or v in ("", "."):
            continue
        try:
            values.append(float(v))
            dates.append(_iso(d))
        except ValueError:
            continue
    return Series(dates, values)


def parse_yahoo_json(payload: dict) -> Series:
    """Yahoo v8 chart JSON: parallel `timestamp` and `indicators.quote[0].close`
    arrays (close may contain nulls on non-trading days)."""
    try:
        result = payload["chart"]["result"][0]
        ts = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError):
        return Series([], [])
    dates, values = [], []
    for t, c in zip(ts, closes):
        if c is None:
            continue
        dates.append(datetime.fromtimestamp(t, tz=timezone.utc).date().isoformat())
        values.append(float(c))
    return Series(dates, values)


def _iso(s: str) -> str:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s  # leave as-is if unrecognized


def trim_window(series: Series, cutoff: str | None, window_days: int) -> Series:
    """Keep only observations on/before ``cutoff`` and within the trailing
    ``window_days`` calendar days (paper limits history to the recent window)."""
    dates, values = series.dates, series.values
    if cutoff:
        keep = [i for i, d in enumerate(dates) if d <= cutoff]
        dates = [dates[i] for i in keep]
        values = [values[i] for i in keep]
    if not dates:
        return Series([], [])
    end = _to_date(dates[-1])
    lo = end.toordinal() - window_days
    keep = [i for i, d in enumerate(dates) if _to_date(d).toordinal() >= lo]
    return Series([dates[i] for i in keep], [values[i] for i in keep])


def _to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# --------------------------------------------------------------------------- #
# Data providers (network)
# --------------------------------------------------------------------------- #
def _get(url: str, params: dict, timeout: float = 30.0):
    resp = httpx.get(
        url, params=params, timeout=timeout, follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; BLF/0.1)"},
    )
    resp.raise_for_status()
    return resp


def _yahoo_period(cutoff: str | None, window_days: int) -> dict:
    """period1/period2 window that ends at the cutoff (leakage guard) and reaches
    far enough back to estimate drift/volatility."""
    hist_days = max(window_days * 3, 420) + window_days
    if cutoff:
        end = datetime.strptime(cutoff, "%Y-%m-%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    else:
        end = datetime.now(timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=hist_days)
    return {"period1": int(start.timestamp()), "period2": int(end.timestamp()), "interval": "1d"}


def fetch_series(
    source: str, series_id: str, cutoff: str | None = None, window_days: int = 60
) -> Series:
    """Fetch a series from a keyless provider. ``source`` in {yahoo, fred}."""
    source = source.lower()
    if source == "yahoo":
        resp = _get(YAHOO_CHART + series_id, _yahoo_period(cutoff, window_days))
        return parse_yahoo_json(resp.json())
    if source == "fred":
        # Best-effort: FRED can be slow/blocked in restricted networks; retry once.
        last: Exception | None = None
        for _ in range(2):
            try:
                return parse_fred_csv(_get(FRED_CSV, {"id": series_id}, timeout=45.0).text)
            except httpx.HTTPError as e:
                last = e
        raise last  # type: ignore[misc]
    raise ValueError(f"unknown time-series source {source!r} (use 'yahoo' or 'fred')")


# --------------------------------------------------------------------------- #
# Model-based estimate (pure)
# --------------------------------------------------------------------------- #
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def model_estimate(
    series: Series,
    threshold: float,
    resolution_date: str,
    kind: str = "auto",
) -> dict:
    """Random-walk-with-drift estimate of P(Y(resolution_date) > threshold)."""
    if len(series) < 3:
        return {"error": "not enough history to model (need >= 3 points)"}

    vals = np.asarray(series.values, dtype=float)
    last_date, last_val = series.dates[-1], float(vals[-1])

    if kind == "auto":
        kind = "log" if np.all(vals > 0) else "level"

    if kind == "log":
        steps = np.diff(np.log(vals))
    else:
        steps = np.diff(vals)

    mu = float(np.mean(steps))
    sigma = float(np.std(steps, ddof=1)) if len(steps) > 1 else 0.0

    h = (_to_date(resolution_date) - _to_date(last_date)).days
    if h <= 0:
        p = 1.0 if last_val > threshold else 0.0
        return _pack(p, mu, sigma, last_val, last_date, 0, kind, threshold, resolution_date)

    if kind == "log":
        mean = math.log(last_val) + mu * h
        std = sigma * math.sqrt(h)
        target = math.log(threshold) if threshold > 0 else -math.inf
    else:
        mean = last_val + mu * h
        std = sigma * math.sqrt(h)
        target = threshold

    if std <= 0:
        p = 1.0 if mean > target else 0.0
    else:
        p = 1.0 - _norm_cdf((target - mean) / std)

    p = min(max(p, _EPS), 1.0 - _EPS)
    return _pack(p, mu, sigma, last_val, last_date, h, kind, threshold, resolution_date)


def _pack(p, mu, sigma, last_val, last_date, h, kind, threshold, resolution_date) -> dict:
    return {
        "p": float(p),
        "model": f"random-walk-drift ({kind})",
        "drift_per_step": mu,
        "vol_per_step": sigma,
        "last_value": last_val,
        "last_date": last_date,
        "horizon_days": h,
        "threshold": threshold,
        "resolution_date": resolution_date,
    }

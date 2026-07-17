"""Web search + a file store implementing the paper's progressive disclosure.

Paper (Sec. C.3): a web search returns up to ~10 short snippets that are added
to the context, while the *full* page content is saved to files. A separate
``read_files`` tool then loads selected files into a cheap sub-LLM summarizer, so
the agent can drill into promising leads without flooding its context with noise.

Two backends are provided, selected with ``BLF_SEARCH_PROVIDER``:
  - ``brave``      — raw (title, url, snippet) hits, as in the paper.
  - ``perplexity`` — returns a synthesized answer plus cited source URLs; the
                     synthesis is surfaced as result 0 and each citation becomes
                     a readable result. (Default: the user has a Perplexity key.)

Full page text is fetched lazily — only when the agent calls ``read_files`` on a
stored result — which is faithful to "progressive disclosure" and cheaper than
fetching every hit up front.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
PERPLEXITY_ENDPOINT = "https://api.perplexity.ai/chat/completions"


@dataclass
class SearchResult:
    id: int
    title: str
    url: str
    snippet: str
    query: str


class FileStore:
    """Holds search results by integer id (the ``search_{i}_result_{j}`` files)."""

    def __init__(self) -> None:
        self._results: dict[int, SearchResult] = {}
        self._next_id = 0

    def add(self, title: str, url: str, snippet: str, query: str) -> SearchResult:
        r = SearchResult(self._next_id, title, url, snippet, query)
        self._results[self._next_id] = r
        self._next_id += 1
        return r

    def get(self, rid: int) -> SearchResult | None:
        return self._results.get(rid)


class BraveSearch:
    def __init__(self, api_key: str | None = None, timeout: float = 15.0) -> None:
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "BRAVE_API_KEY is not set. Set BLF_SEARCH_PROVIDER=perplexity to "
                "use Perplexity instead, or get a Brave key at "
                "https://brave.com/search/api/."
            )
        self.timeout = timeout

    def search(self, query: str, count: int = 10, cutoff: str | None = None) -> list[dict]:
        params: dict[str, str | int] = {"q": query, "count": count}
        if cutoff:
            # Best-effort leakage guard for backtesting (paper's full 4-layer
            # defense, Sec. B, is out of scope for v1). Brave takes a date range.
            params["freshness"] = f"2000-01-01to{cutoff}"
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        resp = httpx.get(BRAVE_ENDPOINT, params=params, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return [
            {"title": i.get("title", ""), "url": i.get("url", ""), "snippet": i.get("description", "")}
            for i in (data.get("web", {}) or {}).get("results", [])[:count]
        ]


class PerplexitySearch:
    """Uses Perplexity's `sonar` models, which search + synthesize in one call."""

    def __init__(
        self, api_key: str | None = None, model: str | None = None, timeout: float = 40.0
    ) -> None:
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise RuntimeError("PERPLEXITY_API_KEY is not set.")
        self.model = model or os.environ.get("BLF_PERPLEXITY_MODEL", "sonar")
        self.timeout = timeout

    def search(self, query: str, count: int = 10, cutoff: str | None = None) -> list[dict]:
        body: dict = {"model": self.model, "messages": [{"role": "user", "content": query}]}
        if cutoff:
            # Perplexity accepts an inclusive upper bound (MM/DD/YYYY).
            y, m, d = cutoff.split("-")
            body["search_before_date_filter"] = f"{m}/{d}/{y}"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = httpx.post(PERPLEXITY_ENDPOINT, json=body, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        answer = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        results = data.get("search_results") or []
        citations = data.get("citations") or []

        out: list[dict] = []
        # Result 0: the synthesized answer (already distilled evidence).
        first_url = (results[0].get("url") if results else (citations[0] if citations else "")) or ""
        if answer:
            out.append({"title": "Perplexity synthesis", "url": first_url, "snippet": answer})
        # Remaining: the cited sources, so read_files can fetch their full text.
        if results:
            for r in results[:count]:
                out.append({"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("snippet", "")})
        else:
            for url in citations[:count]:
                out.append({"title": url, "url": url, "snippet": "(cited source)"})
        return out


def make_search(provider: str | None = None):
    """Factory dispatching on ``BLF_SEARCH_PROVIDER`` (default: perplexity)."""
    provider = provider or os.environ.get("BLF_SEARCH_PROVIDER", "perplexity")
    if provider == "brave":
        return BraveSearch()
    if provider == "perplexity":
        return PerplexitySearch()
    raise ValueError(f"unknown BLF_SEARCH_PROVIDER {provider!r}; use 'brave' or 'perplexity'")


def fetch_page_text(url: str, timeout: float = 15.0, max_chars: int = 12000) -> str:
    """Best-effort full-text fetch for a stored result (used by read_files)."""
    if not url:
        return "[no URL to fetch]"
    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; BLF/0.1)"},
        )
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001 — network is inherently flaky here
        return f"[could not fetch {url}: {e}]"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(" ").split())[:max_chars]

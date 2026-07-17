"""LLM wrappers over any OpenAI-compatible endpoint.

All of the user's available LLM backends — OpenRouter, the gitlawb gateway, and
Groq — speak the OpenAI chat-completions + function-calling protocol, so a single
client parameterized by (base_url, api_key, model) covers them all. Select one
with ``BLF_LLM_PROVIDER``; override the model per role with ``BLF_MAIN_MODEL`` /
``BLF_SUMMARIZER_MODEL``.

The agent relies on function calling with ``tool_choice="required"`` so every
turn yields exactly one action + its embedded ``updated_belief`` (paper Sec. C.2).
If a backend/model rejects ``required``, we transparently fall back to ``auto``.
"""

from __future__ import annotations

import os

from openai import BadRequestError, OpenAI

# provider -> (base_url, api-key env var, default model)
PROVIDER_PRESETS: dict[str, dict] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "openai/gpt-4o-mini",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "gateway": {  # the gitlawb OpenAI-compatible gateway
        "base_url": os.environ.get("OPENAI_BASE_URL"),
        "key_env": "OPENAI_API_KEY",
        "default_model": os.environ.get("OPENAI_MODEL", "mimo-v2.5-pro"),
    },
    "openai": {  # api.openai.com
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
}

DEFAULT_PROVIDER = os.environ.get("BLF_LLM_PROVIDER", "openrouter")


class LLM:
    def __init__(
        self,
        provider: str | None = None,
        main_model: str | None = None,
        summarizer_model: str | None = None,
        tool_choice: str | None = None,
    ) -> None:
        provider = provider or DEFAULT_PROVIDER
        if provider not in PROVIDER_PRESETS:
            raise ValueError(
                f"unknown BLF_LLM_PROVIDER {provider!r}; "
                f"choose from {sorted(PROVIDER_PRESETS)}"
            )
        preset = PROVIDER_PRESETS[provider]
        base_url = os.environ.get("BLF_LLM_BASE_URL") or preset["base_url"]
        api_key = os.environ.get(preset["key_env"])
        if not api_key:
            raise RuntimeError(
                f"{preset['key_env']} is not set (needed for BLF_LLM_PROVIDER="
                f"{provider!r})."
            )

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.main_model = (
            main_model or os.environ.get("BLF_MAIN_MODEL") or preset["default_model"]
        )
        self.summarizer_model = (
            summarizer_model
            or os.environ.get("BLF_SUMMARIZER_MODEL")
            or self.main_model
        )
        self.tool_choice = tool_choice or os.environ.get("BLF_TOOL_CHOICE", "required")

    def act(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 2048):
        """One agent turn. Forces a tool call, falling back to ``auto`` if the
        model rejects ``required``."""
        full = [{"role": "system", "content": system}, *messages]
        kwargs = dict(model=self.main_model, messages=full, tools=tools, max_tokens=max_tokens)
        try:
            return self.client.chat.completions.create(tool_choice=self.tool_choice, **kwargs)
        except BadRequestError:
            return self.client.chat.completions.create(tool_choice="auto", **kwargs)

    def summarize(self, question: str, page_text: str, max_tokens: int = 512) -> str:
        """Cheap sub-LLM that extracts question-relevant facts from a page."""
        prompt = (
            "You are extracting facts relevant to a forecasting question.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"PAGE CONTENT:\n{page_text}\n\n"
            "Summarize ONLY the facts, dates, and figures relevant to answering "
            "the question. If the page is irrelevant, say so in one line."
        )
        resp = self.client.chat.completions.create(
            model=self.summarizer_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

"""Gemini fallback for Relic Loop text-generation stages.

Groq remains the preferred provider. When Groq is unavailable or rate-limited,
this module transparently routes text/JSON generation through Gemini so a daily
Groq token limit cannot kill the entire production run.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from google import genai
from google.genai import types


def _models() -> list[str]:
    raw = os.environ.get("GEMINI_TEXT_MODELS", "gemini-3.8-flash,gemini-2.5-flash-lite")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _client() -> genai.Client:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is required for the Gemini fallback.")
    return genai.Client(api_key=key)


def _prompt(messages: list[dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = str(m.get("content", ""))
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _generate(messages: list[dict[str, str]], max_completion_tokens: int, temperature: float, json_mode: bool = False) -> str:
    last: Exception | None = None
    client = _client()
    for model in _models():
        for attempt in range(1, 3):
            try:
                config = types.GenerateContentConfig(
                    max_output_tokens=max_completion_tokens,
                    temperature=temperature,
                    response_mime_type="application/json" if json_mode else None,
                )
                response = client.models.generate_content(
                    model=model,
                    contents=_prompt(messages),
                    config=config,
                )
                text = (response.text or "").strip()
                if not text:
                    raise RuntimeError(f"Gemini returned an empty response for {model}.")
                return text
            except Exception as exc:
                last = exc
                print(f"[GEMINI FALLBACK] {model} attempt {attempt}/2 failed: {exc}")
                if attempt < 2:
                    time.sleep(2.0 * attempt)
    raise RuntimeError(f"Gemini fallback failed for all configured models: {last}")


def _install(p) -> None:
    original_groq_call = p.groq_call
    original_groq_json = p.groq_json
    original_groq_browser_search = p.groq_browser_search

    def groq_call_with_fallback(model, messages, *, max_completion_tokens, temperature=0.6, attempts=5):
        try:
            return original_groq_call(
                model, messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                attempts=attempts,
            )
        except Exception as exc:
            print(f"[LLM FAILOVER] Groq text generation failed: {exc}")
            return _generate(messages, max_completion_tokens, temperature, json_mode=False)

    def groq_json_with_fallback(model, messages, *, max_completion_tokens, temperature, attempts=3):
        try:
            return original_groq_json(
                model, messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                attempts=attempts,
            )
        except Exception as exc:
            print(f"[LLM FAILOVER] Groq JSON generation failed: {exc}")
            raw = _generate(messages, max_completion_tokens, temperature, json_mode=True)
            try:
                return p.extract_last_json_object(p.clean_json_text(raw))
            except Exception:
                return json.loads(raw)

    def groq_browser_search_with_fallback(model, prompt, *, max_completion_tokens=2800, attempts=3):
        try:
            return original_groq_browser_search(
                model, prompt,
                max_completion_tokens=max_completion_tokens,
                attempts=attempts,
            )
        except Exception as exc:
            print(f"[LLM FAILOVER] Groq browser research failed: {exc}")
            # Gemini search grounding is used when available, so research does not
            # silently become uncited model-only guessing merely because Groq is out.
            client = _client()
            last: Exception | None = None
            for model_name in _models():
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            max_output_tokens=max_completion_tokens,
                            temperature=0.3,
                            tools=[types.Tool(google_search=types.GoogleSearch())],
                        ),
                    )
                    text = (response.text or "").strip()
                    if text:
                        return text
                except Exception as search_exc:
                    last = search_exc
                    print(f"[GEMINI SEARCH FALLBACK] {model_name} failed: {search_exc}")
            print("[GEMINI SEARCH FALLBACK] Search grounding unavailable; using plain Gemini research.")
            return _generate(
                [{"role": "user", "content": prompt}],
                max_completion_tokens,
                0.3,
                json_mode=False,
            )

    p.groq_call = groq_call_with_fallback
    p.groq_json = groq_json_with_fallback
    p.groq_browser_search = groq_browser_search_with_fallback
    print("[GEMINI HARDENING] Groq -> Gemini text/JSON/search failover installed.")


_install(__import__("pipeline"))

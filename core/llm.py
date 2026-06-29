"""
LLM clients. Three providers, two roles:
  - Groq Llama 3.3 70B Versatile  -> routing / classification / cheap calls
  - Google Gemini 2.5 Flash/Pro   -> synthesis with long context
  - Siemens OpenAI-compatible API -> fallback when Gemini or Groq quota exhausted

Uses the new google-genai SDK (the old google-generativeai package is deprecated).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError(
        "Could not import `google.genai`. The new Google Gen AI SDK package is "
        "`google-genai` (note the hyphen and the missing 'erative'). The OLD "
        "package `google-generativeai` is deprecated and does NOT provide "
        "`google.genai`.\n\n"
        "Fix:  pip install -U google-genai\n"
        "Then (optional, to silence the deprecation warning):  pip uninstall google-generativeai\n\n"
        f"Underlying error: {e}"
    ) from e

from groq import Groq
from openai import OpenAI

from . import config

_gemini = genai.Client(api_key=config.GOOGLE_API_KEY)
_groq = Groq(api_key=config.GROQ_API_KEY)

# Siemens fallback client (None if key not configured)
_siemens: OpenAI | None = None
if config.SIEMENS_API_KEY:
    _siemens = OpenAI(
        api_key=config.SIEMENS_API_KEY,
        base_url=config.SIEMENS_BASE_URL,
    )


def _normalize_model(name: str | None, default: str) -> str:
    """The new google-genai SDK takes bare model ids (no 'models/' prefix)."""
    n = (name or default).strip()
    return n[len("models/") :] if n.startswith("models/") else n


def _is_quota_exhausted(err_str: str) -> bool:
    return "429" in err_str or "RESOURCE_EXHAUSTED" in err_str


def _backoff_seconds(err_str: str, attempt: int) -> float:
    """Return how long to sleep given an error string and attempt number.

    For Gemini 429 rate limits, parse the 'retry in Xs' value from the message
    so we actually wait the suggested delay instead of retrying immediately.
    Caps at 120s so the UI doesn't freeze indefinitely.
    """
    if _is_quota_exhausted(err_str):
        m = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", err_str, re.IGNORECASE)
        suggested = float(m.group(1)) if m else 60.0
        return min(suggested + 2, 120.0)
    if "503" in err_str or "UNAVAILABLE" in err_str:
        return 5.0 * (attempt + 1)
    return float(attempt + 1)


def _siemens_generate(
    prompt: str,
    *,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
) -> str:
    """Call the Siemens OpenAI-compatible endpoint."""
    if _siemens is None:
        raise RuntimeError("Siemens API not configured. Set SIEMENS_API_KEY in .env")
    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    r = _siemens.chat.completions.create(
        model=config.SIEMENS_MODEL or "gpt-oss-120b-onprem",
        messages=messages,
        temperature=temperature,
        max_tokens=min(max_output_tokens, 4096),
    )
    return (r.choices[0].message.content or "").strip()


def _build_gen_config(
    *,
    system_instruction: str | None,
    temperature: float,
    max_output_tokens: int,
    response_mime_type: str | None,
    response_schema: dict | None,
    thinking_budget: int | None,
) -> types.GenerateContentConfig:
    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if system_instruction:
        kwargs["system_instruction"] = system_instruction
    if response_mime_type:
        kwargs["response_mime_type"] = response_mime_type
    if response_schema:
        kwargs["response_schema"] = response_schema
    if thinking_budget is not None:
        # thinking_budget=0 disables Gemini 2.5's default "thinking" mode.
        # Critical because thinking tokens count against max_output_tokens;
        # leaving it on default can yield empty replies on tight budgets.
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
    return types.GenerateContentConfig(**kwargs)


def groq_chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    response_format_json: bool = False,
) -> str:
    """Single-shot Groq chat completion with Siemens fallback."""
    kwargs: dict[str, Any] = {
        "model": model or config.GROQ_ROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format_json:
        kwargs["response_format"] = {"type": "json_object"}
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = _groq.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))

    # Groq failed after retries — fall back to Siemens if available
    if _siemens is not None:
        fallback_msgs = list(messages)
        if response_format_json:
            # Inject JSON instruction since Siemens may not support response_format
            fallback_msgs = []
            for msg in messages:
                if msg["role"] == "system":
                    fallback_msgs.append({
                        "role": "system",
                        "content": msg["content"] + "\n\nIMPORTANT: Return ONLY valid JSON with no markdown fences or extra text.",
                    })
                else:
                    fallback_msgs.append(msg)
            if not any(m["role"] == "system" for m in messages):
                fallback_msgs.insert(0, {
                    "role": "system",
                    "content": "Return ONLY valid JSON with no markdown fences or extra text.",
                })
        r = _siemens.chat.completions.create(
            model=config.SIEMENS_MODEL or "gpt-oss-120b-onprem",
            messages=fallback_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (r.choices[0].message.content or "").strip()

    raise RuntimeError(f"Groq call failed after retries: {last_err!r}")


def gemini_generate(
    prompt: str,
    *,
    model: str | None = None,
    system_instruction: str | None = None,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
    response_mime_type: str | None = None,
    response_schema: dict | None = None,
    thinking_budget: int | None = 0,
) -> str:
    """Single-turn Gemini generation with Siemens fallback on quota exhaustion.

    `thinking_budget`:
        0   (default) -> thinking disabled. Reproducible token use, no empty
                         replies caused by thinking eating the budget.
        N>0           -> allow N thinking tokens before the model emits output.
        None          -> use Gemini's automatic thinking budget.
    """
    cfg = _build_gen_config(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
        thinking_budget=thinking_budget,
    )
    model_name = _normalize_model(model, config.GEMINI_SYNTHESIS_MODEL)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = _gemini.models.generate_content(
                model=model_name,
                contents=prompt,
                config=cfg,
            )
            return (resp.text or "").strip()
        except Exception as e:
            last_err = e
            err_str = str(e)
            wait = _backoff_seconds(err_str, attempt)
            if attempt < 2:
                time.sleep(wait)

    # Gemini exhausted — fall back to Siemens if quota error and key is configured
    if _is_quota_exhausted(str(last_err)) and _siemens is not None:
        return _siemens_generate(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    raise RuntimeError(f"Gemini call failed after retries: {last_err!r}")


def gemini_json(prompt: str, schema: dict, *, model: str | None = None, **kwargs) -> dict:
    """Force-JSON helper using Gemini's response_schema. Falls back to Siemens on quota."""
    gemini_kwargs = {k: v for k, v in kwargs.items()
                    if k not in ("response_mime_type", "response_schema")}
    last_err: Exception | None = None
    try:
        text = gemini_generate(
            prompt,
            model=model,
            response_mime_type="application/json",
            response_schema=schema,
            **gemini_kwargs,
        )
        return json.loads(text)
    except RuntimeError as e:
        last_err = e
        if _siemens is None or not _is_quota_exhausted(str(e)):
            raise
    except json.JSONDecodeError as e:
        snippet = str(e)[:200]
        raise ValueError(f"Gemini returned non-JSON: {snippet!r}") from e

    # Siemens fallback: instruct model to return plain JSON, parse ourselves
    json_prompt = (
        f"{prompt}\n\n"
        "IMPORTANT: Return ONLY valid JSON with no markdown fences or extra text."
    )
    raw = _siemens_generate(
        json_prompt,
        system_instruction=kwargs.get("system_instruction"),
        temperature=kwargs.get("temperature", 0.1),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
    )
    # Strip any markdown fences the model added anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Siemens fallback returned non-JSON: {raw[:200]!r}") from e

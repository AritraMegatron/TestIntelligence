from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from json import JSONDecodeError
from typing import Literal

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # Keeps non-LLM imports and unit tests from crashing.
    OpenAI = None


load_dotenv(override=True)

LLMProvider = Literal["openai", "ollama"]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3").strip()
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))

_active_provider: LLMProvider = "openai"
_openai_client = None


def configure_llm_provider(provider: str) -> LLMProvider:
    """Select the provider used by all LLM-backed application features."""
    normalized = (provider or "").strip().lower()

    match normalized:
        case "openai":
            selected: LLMProvider = "openai"
        case "ollama":
            selected = "ollama"
        case _:
            raise ValueError(
                f"Unsupported LLM_PROVIDER '{provider}'. Use 'openai' or 'ollama'."
            )

    global _active_provider
    _active_provider = selected
    return selected


def get_llm_provider() -> LLMProvider:
    return _active_provider


def get_llm_model() -> str:
    match _active_provider:
        case "openai":
            return OPENAI_MODEL
        case "ollama":
            return OLLAMA_MODEL


def _get_openai_client():
    global _openai_client

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your .env file or select "
            "LLM_PROVIDER=ollama."
        )

    if OpenAI is None:
        raise RuntimeError(
            "The openai package is missing. Install it with: pip install openai"
        )

    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)

    return _openai_client


def _strip_markdown_fence(text: str) -> str:
    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    if cleaned.lower().startswith("json\n"):
        cleaned = cleaned[5:].strip()

    return cleaned


def _extract_first_json_object(text: str) -> str:
    """Extract the first balanced JSON object from model output."""
    cleaned = _strip_markdown_fence(text)

    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    start = cleaned.find("{")
    if start < 0:
        raise JSONDecodeError("No JSON object found", cleaned, 0)

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(cleaned)):
        char = cleaned[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start:index + 1]

    raise JSONDecodeError("Unbalanced JSON object", cleaned, start)


def _parse_json_lenient(text: str) -> dict:
    json_text = _extract_first_json_object(text)
    parsed = json.loads(json_text)

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object, but got a JSON array or scalar.")

    return parsed


def _call_openai_text(system_prompt: str, user_prompt: str) -> str:
    response = _get_openai_client().responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    text = (getattr(response, "output_text", None) or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty response.")
    return text


def _call_ollama_text(system_prompt: str, user_prompt: str) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=LLM_TIMEOUT_SECONDS) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code} from {url}. "
            f"Model: {OLLAMA_MODEL}. Response: {error_body or exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {url}. Make sure Ollama is running "
            f"and that OLLAMA_BASE_URL is correct. Error: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama did not respond within {LLM_TIMEOUT_SECONDS:g} seconds."
        ) from exc

    try:
        result = json.loads(raw_body)
    except JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON: {raw_body[:500]}") from exc

    text = str(result.get("message", {}).get("content", "")).strip()
    if not text:
        raise RuntimeError(
            f"Ollama returned an empty response for model '{OLLAMA_MODEL}'."
        )
    return text


def _call_llm_text(system_prompt: str, user_prompt: str) -> str:
    match _active_provider:
        case "openai":
            return _call_openai_text(system_prompt, user_prompt)
        case "ollama":
            return _call_ollama_text(system_prompt, user_prompt)


def call_llm_for_json(system_prompt: str, user_prompt: str) -> dict:
    """
    Call the selected LLM provider and return one parsed JSON object.

    The parser accepts raw JSON, markdown-fenced JSON, or a JSON object surrounded
    by prose. If parsing fails, the same provider gets one repair attempt.
    """
    text_output = _call_llm_text(system_prompt, user_prompt)

    try:
        return _parse_json_lenient(text_output)
    except Exception:
        repair_prompt = f"""
The previous model response was supposed to be exactly one JSON object, but it was not parseable.

Return only the corrected JSON object.
Do not add markdown.
Do not add explanations.

Original user prompt:
{user_prompt}

Unparseable model response:
{text_output}
"""
        repaired_output = _call_llm_text(system_prompt, repair_prompt)

        try:
            return _parse_json_lenient(repaired_output)
        except Exception as second_exc:
            raise ValueError(
                f"{_active_provider.title()} did not return valid JSON after one repair attempt.\n\n"
                f"First raw output:\n{text_output}\n\n"
                f"Repair raw output:\n{repaired_output}"
            ) from second_exc

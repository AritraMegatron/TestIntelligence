import os
import json
from json import JSONDecodeError
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # keeps non-LLM unit tests/imports from crashing
    OpenAI = None

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")

client = None


def _get_client():
    global client

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

    if OpenAI is None:
        raise RuntimeError("openai package is missing. Install it with: pip install openai")

    if client is None:
        client = OpenAI(api_key=OPENAI_API_KEY)

    return client


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
    """
    Extract the first balanced JSON object from model text.
    Handles common cases like: "Here is the JSON: {...}" or markdown fences.
    """
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
    response = _get_client().responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )
    return response.output_text


def call_llm_for_json(system_prompt: str, user_prompt: str) -> dict:
    """
    Calls OpenAI and returns a parsed JSON object.

    Hardened behavior for MVP chat:
    - accepts raw JSON
    - accepts JSON wrapped in markdown fences
    - extracts the first balanced JSON object from surrounding prose
    - retries once with a JSON-repair prompt if parsing fails
    """
    text_output = _call_openai_text(system_prompt, user_prompt)

    try:
        return _parse_json_lenient(text_output)
    except Exception as first_exc:
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
        repaired_output = _call_openai_text(system_prompt, repair_prompt)

        try:
            return _parse_json_lenient(repaired_output)
        except Exception as second_exc:
            raise ValueError(
                "LLM did not return valid JSON after one repair attempt.\n\n"
                f"First raw output:\n{text_output}\n\n"
                f"Repair raw output:\n{repaired_output}"
            ) from second_exc
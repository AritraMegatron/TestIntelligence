"""
Deterministic guardrails for the RF chat generator.

These helpers keep the NiceGUI chat flow resilient when the LLM returns
slightly messy JSON, incomplete ready objects, or asks for defaults.
They intentionally do not contain any UI code.
"""

from __future__ import annotations

import re
from typing import Any

from app.models.rf_models import RFGenerationResponse, RFTestCase


DEFAULT_ASSUMPTION_TEXT = "You can reply 'use defaults' to generate a demo-ready test with safe MVP defaults."


def user_requested_defaults(message: str) -> bool:
    """
    Detect when the user explicitly wants the app to fill missing required
    RF bench parameters with demo/default values.

    Important MVP behavior:
    - "Use defaults" as a standalone reply should fill missing parameters.
    - "Use defaults for missing parameters" should fill missing parameters.
    - But a one-shot request like:
        "Create an EVM test at 2450 MHz using defaults and also measure input power"
      should NOT automatically bypass clarification, because it may simply mean
      use normal test assumptions while still asking for required bench details.
    """
    msg = (message or "").strip().lower()

    # Strong standalone intent after the app asked clarifying questions.
    standalone_phrases = {
        "use default",
        "use defaults",
        "default",
        "defaults",
        "demo",
        "demo values",
        "make a demo",
        "demo test",
    }

    if msg in standalone_phrases:
        return True

    for phrase in standalone_phrases:
        if msg.endswith(f". {phrase}") or msg.endswith(f", {phrase}"):
            return True

    # Strong explicit intent to fill missing parameters.
    explicit_fill_phrases = [
        "use defaults for missing",
        "use default for missing",
        "use defaults for the missing",
        "use default for the missing",
        "fill missing with defaults",
        "fill missing with default",
        "use demo default values",
        "fill missing parameters with defaults",
        "fill missing bench parameters with defaults",
        "use demo values for missing",
        "use demo values for the missing",
        "use defaults for everything else",
        "use default for everything else",
        "everything else default",
        "keep everything else default",
        "for other parameters use defaults",
        "for other parameters use default",
        "other parameters use defaults",
        "other parameters use default",
    ]

    return any(phrase in msg for phrase in explicit_fill_phrases)


def detect_test_type_from_text(text: str) -> str | None:
    """
    Detect only clear test-type intent.
    Important: do not infer GAIN from DUT mode names like TX HIGH GAIN.
    """
    msg = (text or "").lower()

    if re.search(r"\b(evm|error vector magnitude)\b", msg):
        return "EVM"

    if re.search(r"\b(acpr|aclr|adjacent channel power|adjacent channel leakage)\b", msg):
        return "ACPR"

    if re.search(r"\b(current test|current consumption|supply current|icc|idd|iddq)\b", msg):
        return "CURRENT"

    sparam_keywords = [
        "s-parameter",
        "s parameter",
        "s11",
        "s21",
        "s12",
        "s22",
        "return loss",
        "insertion loss",
        "insertion gain",
        "reverse isolation",
        "output match",
        "input match",
        "vswr",
        "vna",
    ]
    if any(keyword in msg for keyword in sparam_keywords):
        return "S_PARAMETER"

    gain_patterns = [
        r"\bgain test\b",
        r"\btest gain\b",
        r"\bmeasure gain\b",
        r"\bmeasurement of gain\b",
        r"\bsmall signal gain\b",
        r"\bpower gain test\b",
    ]
    if any(re.search(pattern, msg) for pattern in gain_patterns):
        return "GAIN"

    return None


def is_small_talk(message: str) -> bool:
    msg = (message or "").strip().lower()
    return msg in {"hi", "hello", "hey", "yo", "sup", "thanks", "thank you"}


def coerce_generation_json(raw: Any, test_id: str) -> dict:
    """
    Make raw LLM JSON safer before Pydantic validation.
    This does not invent RF parameters; it only fixes container shape issues.
    """
    if not isinstance(raw, dict):
        return clarification_payload(
            ["I could not read a valid RF test request from that message. Which test type do you want: GAIN, EVM, ACPR, CURRENT, or S_PARAMETER?"],
            [DEFAULT_ASSUMPTION_TEXT],
        )

    payload = dict(raw)
    payload.setdefault("status", "needs_clarification")
    payload.setdefault("clarifying_questions", [])
    payload.setdefault("default_assumptions", [])

    if payload["status"] not in {"ready", "needs_clarification"}:
        payload["status"] = "needs_clarification"

    if not isinstance(payload.get("clarifying_questions"), list):
        payload["clarifying_questions"] = [str(payload.get("clarifying_questions"))]

    if not isinstance(payload.get("default_assumptions"), list):
        payload["default_assumptions"] = [str(payload.get("default_assumptions"))]

    if payload["status"] == "needs_clarification":
        payload["test_case"] = None
        return payload

    test_case = payload.get("test_case")
    if not isinstance(test_case, dict):
        return clarification_payload(
            ["I understood the request, but the generated RF test object was incomplete. Please provide the required RF details or reply 'use defaults'."],
            payload.get("default_assumptions") or [DEFAULT_ASSUMPTION_TEXT],
        )

    test_case.setdefault("schema_version", "inventide.rf.testcase.v1")
    test_case.setdefault("domain", "RF")
    test_case.setdefault("test_id", test_id)
    test_case.setdefault("objective", "Generated RF test from chat request")
    test_case.setdefault("stimulus", {})
    test_case.setdefault("dut_bias", {})
    test_case.setdefault("environment", {})
    test_case.setdefault("execution_target", {"runner": "Generic_RF_TestRunner", "config_format": "csv"})

    if "test_type" in test_case and "measurement" not in test_case:
        test_case["measurement"] = {"metric": test_case["test_type"], "limit": None}

    if "measurement" in test_case and isinstance(test_case["measurement"], dict):
        if not test_case["measurement"].get("metric") and test_case.get("test_type"):
            test_case["measurement"]["metric"] = test_case["test_type"]

    if test_case.get("test_type") == "S_PARAMETER":
        test_case["bench_type"] = "S_PARAMETER_BENCH"
    else:
        test_case.setdefault("bench_type", "RF_BENCH")

    sweep = test_case.get("sweep")
    if isinstance(sweep, dict) and isinstance(sweep.get("expansion_order"), list):
        sweep["expansion_order"] = [
            "power" if axis in {"input_power_dbm", "servo_output_power_dbm"} else axis
            for axis in sweep["expansion_order"]
        ]

    payload["test_case"] = test_case
    return payload


def clarification_payload(questions: list[str], assumptions: list[str] | None = None) -> dict:
    return {
        "status": "needs_clarification",
        "clarifying_questions": questions or ["Please provide the remaining RF test details."],
        "default_assumptions": assumptions or [DEFAULT_ASSUMPTION_TEXT],
        "test_case": None,
    }


def response_from_clarification(questions: list[str], assumptions: list[str] | None = None) -> RFGenerationResponse:
    return RFGenerationResponse.model_validate(clarification_payload(questions, assumptions))


def questions_from_validation_errors(errors: list[str]) -> list[str]:
    """
    Convert missing-field validator errors into user-friendly chat questions.
    Non-missing engineering errors are intentionally ignored here so they remain
    visible in the validation panel instead of being rephrased as missing info.
    """
    questions: list[str] = []

    mapping = [
        ("frequency_mhz", "What RF frequency or frequency sweep should I use?"),
        ("dut_mode_name", "What DUT mode name should I use, for example TX HIGH GAIN?"),
        ("modulation", "What modulation should I use, for example CW, 802.11ax_HE20, LTE, or 5G NR?"),
        ("input_power_dbm", "What input/source power in dBm should I use?"),
        ("servo_output_power_dbm", "What target output power in dBm should the bench servo to?"),
        ("vcc_v", "What VCC voltage should I use?"),
        ("vdd_v", "What VDD voltage should I use?"),
        ("temperature_c", "What temperature in C should I use?"),
        ("bandwidth_mhz", "What bandwidth in MHz should I use?"),
        ("duty_cycle_pct", "What duty cycle percentage should I use for dynamic EVM?"),
        ("evm_settings.evm_type", "Should this be STATIC EVM or DYNAMIC EVM?"),
        ("evm_type", "Should this be STATIC EVM or DYNAMIC EVM?"),
        ("acpr band pair", "What ACPR lower/upper offset pair should I use, for example -10 MHz and +10 MHz?"),
        ("acpr_band_pairs", "What ACPR lower/upper offset pair should I use, for example -10 MHz and +10 MHz?"),
        ("s_parameter_settings", "What VNA S-parameter settings should I use: measurements, start/stop frequency, ports, and VNA power?"),
    ]

    missing_markers = [
        "is required",
        "requires",
        "must be non-empty",
        "missing",
    ]

    for error in errors:
        lower_error = error.lower()
        if not any(marker in lower_error for marker in missing_markers):
            continue

        for needle, question in mapping:
            if needle.lower() in lower_error and question not in questions:
                questions.append(question)

    return questions


def default_assumptions_for_missing() -> list[str]:
    return [
        "temperature_c can default to [25 C]",
        "demo mode can fill safe MVP values for frequency, DUT mode, modulation, power, VCC, and VDD",
    ]
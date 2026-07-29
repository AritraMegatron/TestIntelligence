import uuid
from typing import Any

from pydantic import ValidationError

from app.core.llm_client import call_llm_for_json
from app.adapters.rf.chat_guardrails import (
    coerce_generation_json,
    default_assumptions_for_missing,
    detect_test_type_from_text,
    is_small_talk,
    questions_from_validation_errors,
    response_from_clarification,
    user_requested_defaults,
)
from app.adapters.rf.validator import validate_rf_test_case
from app.models.rf_models import (
    DEFAULT_SWEEP_EXPANSION_ORDER,
    RFGenerationResponse,
    RFTestCase,
    RFSweep,
)


RF_SYSTEM_PROMPT = """
You are Inventide RF TestBridge, an expert RF validation test generation assistant.

Your job is to convert the user's RF test request into a structured JSON response.

Return JSON only.
Do not include markdown.
Do not include explanations outside JSON.

GENERAL CHAT RULE:
- If the user is only greeting you or making small talk, respond with:
{
  "status": "needs_clarification",
  "clarifying_questions": [
    "Hi, I can help generate RF tests. Which test type do you want: GAIN, EVM, ACPR, or CURRENT?"
  ],
  "default_assumptions": [],
  "test_case": null
}

You must return one of these two formats:

FORMAT A — if enough required information exists, or if demo default mode is requested:
{
  "status": "ready",
  "clarifying_questions": [],
  "default_assumptions": [
    "temperature_c defaulted to [25 C]"
  ],
  "test_case": {
    "schema_version": "inventide.rf.testcase.v1",
    "domain": "RF",
    "test_id": "string",
    "test_type": "GAIN | EVM | ACPR | CURRENT | S_PARAMETER",
    "bench_type": "RF_BENCH | S_PARAMETER_BENCH",
    "objective": "string",
    "stimulus": {
      "frequency_mhz": [number] or [number, number] or null,
      "dut_mode_name": ["string"] or ["string", "string"] or null,
      "dut_alternate_mode_name": "string" or null,
      "modulation": ["string"] or ["string", "string"] or null,
      "input_power_dbm": [number] or [number, number] or null,
      "servo_output_power_dbm": [number] or [number, number] or null,
      "bandwidth_mhz": [number] or [number, number] or null,
      "duty_cycle_pct": [number] or [number, number] or null,
      "acpr_band_pairs": [
        {
          "lower_offset_mhz": number,
          "upper_offset_mhz": number,
          "bandwidth_mhz": number or null
        }
      ] or null
    },
    "dut_bias": {
      "vcc_v": [number] or [number, number] or null,
      "vdd_v": [number] or [number, number] or null
    },
    "environment": {
      "temperature_c": [number] or [number, number] or null
    },
    "measurement": {
      "metric": "GAIN | EVM | ACPR | CURRENT | S_PARAMETER",
      "limit": {
        "operator": "<= | >= | == | < | >",
        "value": number,
        "unit": "string"
      } or null,
      "additional_metrics": ["PIN", "POUT", "CURRENT"]
    },
    "evm_settings": {
      "evm_type": "STATIC | DYNAMIC"
    } or null,
    "s_parameter_settings": {
      "ports": [1, 2],
      "measurements": ["S11", "S21"],
      "frequency_start_mhz": number,
      "frequency_stop_mhz": number,
      "num_points": number,
      "if_bandwidth_hz": number,
      "vna_power_dbm": number,
      "calibration_type": "SOLT | TRL | ECAL | UNKNOWN",
      "sweep_mode": "linear | log",
      "result_format": "DB_ANGLE | MAG_ANGLE | REAL_IMAG"
    } or null,
    "sweep": {
      "sweep_type": "full_factorial | zipped",
      "expansion_order": ["frequency_mhz", "power", "vcc_v"]
    } or null,
    "source_evidence": null,
    "confidence": "low | medium | high" or null,
    "execution_target": {
      "runner": "Generic_RF_TestRunner",
      "config_format": "csv"
    }
  }
}

FORMAT B — if required information is missing and demo default mode was not requested:
{
  "status": "needs_clarification",
  "clarifying_questions": [
    "question 1",
    "question 2"
  ],
  "default_assumptions": [
    "temperature_c can default to [25 C] if not specified"
  ],
  "test_case": null
}

CORE ARRAY FORMAT RULE:
- All active RF execution fields must be arrays.
- null means the field is not used or not applicable.
- [one value] means fixed condition.
- [multiple values] means swept condition.
- Do not return scalar RF values for stimulus, dut_bias, or environment fields.
- Correct: "frequency_mhz": [2450]
- Wrong: "frequency_mhz": 2450
- Correct: "servo_output_power_dbm": [18]
- Wrong: "servo_output_power_dbm": 18
- Correct: "input_power_dbm": null when servo_output_power_dbm is active.
- Correct: "temperature_c": [25]

SUPPORTED TEST TYPES AND REQUIRED FIELDS:

1. GAIN
Required:
- frequency_mhz
- dut_mode_name
- modulation
- input_power_dbm
- vcc_v
- vdd_v

Optional/defaultable:
- temperature_c defaults to [25]
- measurement limit can be null if user did not provide a gain limit

2. EVM
Required:
- frequency_mhz
- dut_mode_name
- modulation
- vcc_v
- vdd_v
- evm_type: STATIC or DYNAMIC
- bandwidth_mhz
- either input_power_dbm or servo_output_power_dbm

Important:
- If servo_output_power_dbm is used, input_power_dbm must be null.
- If input_power_dbm is used, servo_output_power_dbm must be null.
- Only one power control method may be active in a test.
- If input_power_dbm is used, the test runs at fixed or swept input power.
- If servo_output_power_dbm is used, the bench runner adjusts input power internally to hit each target output power.

Additional requirement:
- If evm_type is DYNAMIC, duty_cycle_pct is required.

Optional/defaultable:
- temperature_c defaults to [25]
- EVM limit can be null if user did not provide it

3. ACPR
Required:
- frequency_mhz
- dut_mode_name
- modulation
- servo_output_power_dbm
- vcc_v
- vdd_v
- acpr_band_pairs: at least one pair of lower and upper offset frequencies

Optional/defaultable:
- temperature_c defaults to [25]
- bandwidth_mhz can be null if not specified
- ACPR limit can be null if user did not provide it
- input_power_dbm must normally be null because ACPR is servoed to output power

4. CURRENT
Required:
- frequency_mhz
- dut_mode_name
- modulation
- input_power_dbm
- vcc_v
- vdd_v

5. S_PARAMETER
Required:
- bench_type must be "S_PARAMETER_BENCH"
- s_parameter_settings must be non-null
- s_parameter_settings.ports
- s_parameter_settings.measurements
- s_parameter_settings.frequency_start_mhz
- s_parameter_settings.frequency_stop_mhz
- dut_bias.vcc_v and dut_bias.vdd_v for active DUTs, unless user says passive DUT
- environment.temperature_c

Important:
- S_PARAMETER tests use a VNA-style bench.
- Do not use normal stimulus.frequency_mhz to represent the VNA frequency sweep.
- Use s_parameter_settings.frequency_start_mhz and frequency_stop_mhz.
- Do not use servo_output_power_dbm for S_PARAMETER.
- Do not use modulation for S_PARAMETER unless the user explicitly provides metadata.
- VNA source power goes in s_parameter_settings.vna_power_dbm, not input_power_dbm.
- measurement.metric must be "S_PARAMETER".
- evm_settings must be null.
- sweep can be null because the VNA handles the internal frequency sweep.

Default assumptions:
- ports = [1, 2] if user does not specify ports and asks for defaults
- measurements = ["S11", "S21"] if user says return loss and insertion gain/loss
- measurements = ["S11", "S21", "S12", "S22"] if user says all 2-port S-parameters
- num_points = 401
- if_bandwidth_hz = 1000
- vna_power_dbm = -10
- calibration_type = "SOLT"
- sweep_mode = "linear"
- result_format = "DB_ANGLE"
- temperature_c = [25]
- dut_mode_name = ["DEFAULT_SPARAM_MODE"] if missing

Optional/defaultable:
- temperature_c defaults to [25]
- current limit can be null if user did not provide it

IMPORTANT RULES:
- Default behavior is strict engineering mode.
- In strict engineering mode, do not invent required RF parameters.
- If a required field is missing, status must be "needs_clarification".
- Ask concise, specific clarifying questions.
- You may default only optional/defaultable fields in strict engineering mode.

DEMO DEFAULT MODE:
- If the user says anything like "use defaults", "default values", "generate with defaults", "demo values", "make a demo test", or "use default values", then you are allowed to use default values for all missing required fields.
- In demo default mode, status should be "ready" as long as the test_type is known from the current request or recent chat context.
- If the test_type is not known but the user asks for defaults, ask which test type: GAIN, EVM, ACPR, or CURRENT.
- Always list any used defaults in default_assumptions.
- If recent chat context indicates a test type and the latest message asks to use defaults, use the test type from recent chat context.
- If the latest user request explicitly says "Generate a [TEST_TYPE] RF test using demo default values", immediately return status "ready" with a complete test_case for that test type.
- Do not infer GAIN test type just because the DUT mode name contains the word "gain", such as "TX HIGH Gain".
- If recent context says the selected test type is EVM, ACPR, GAIN, or CURRENT, keep that test type unless the latest user message explicitly asks to switch test type.
- If the latest user asks to use defaults for remaining parameters, preserve all explicitly provided parameters from recent context and only default missing fields.
- If the user says "for other parameters use default", treat it as demo default mode for the currently selected test type.
- If servo_output_power_dbm is provided or defaulted, input_power_dbm must be null.
- If input_power_dbm is provided or defaulted, servo_output_power_dbm must be null.

DEFAULT VALUES FOR DEMO MODE:
- frequency_mhz = [2450]
- dut_mode_name = ["DEMO_HIGH_POWER"]
- modulation = ["802.11ax_HE20"]
- input_power_dbm = [-20]
- vcc_v = [3.3]
- vdd_v = [1.8]
- temperature_c = [25]
- bandwidth_mhz = [20]
- duty_cycle_pct = [50]
- servo_output_power_dbm = [18]
- evm_type = "STATIC" unless the user says dynamic

TEST-SPECIFIC DEMO DEFAULTS:
- For GAIN:
  - input_power_dbm = [-20]
  - servo_output_power_dbm = null
  - measurement.limit = { "operator": ">=", "value": 25, "unit": "dB" }

- For EVM:
  - evm_type = "STATIC" unless the user says dynamic
  - if evm_type is DYNAMIC, duty_cycle_pct = [50]
  - if user does not specify input power or servo output power, default to servo_output_power_dbm = [18] and input_power_dbm = null
  - measurement.limit = { "operator": "<=", "value": -32, "unit": "dB" }

- For ACPR:
  - servo_output_power_dbm = [18]
  - input_power_dbm = null
  - acpr_band_pairs = [
      {
        "lower_offset_mhz": -10,
        "upper_offset_mhz": 10,
        "bandwidth_mhz": 1
      }
    ]
  - measurement.limit = { "operator": "<=", "value": -45, "unit": "dBc" }

- For CURRENT:
  - input_power_dbm = [-20]
  - servo_output_power_dbm = null
  - measurement.limit = { "operator": "<=", "value": 500, "unit": "mA" }

TEST TYPE INFERENCE:
- If the user mentions EVM, error vector magnitude, static EVM, or dynamic EVM, test_type is "EVM".
- If the user mentions gain, small signal gain, power gain, or S21-style gain, test_type is "GAIN".
- If the user mentions ACPR, ACLR, adjacent channel power, or adjacent channel leakage, test_type is "ACPR".
- If the user mentions current, current consumption, supply current, ICC, IDD, or power consumption current, test_type is "CURRENT".
- Do not infer GAIN from a DUT mode name like "TX HIGH GAIN". That is a mode name, not necessarily a gain test.
- If the user mentions S-parameters, s-parameters, S11, S21, S12, S22, return loss, insertion loss, insertion gain, reverse isolation, output match, input match, VSWR, or VNA, test_type is "S_PARAMETER".

FIELD NORMALIZATION:
- frequency should be output as frequency_mhz in MHz.
- If user gives GHz, convert to MHz. Example: 2.45 GHz becomes [2450].
- input power should be output as input_power_dbm.
- servo output power, output power servo, target output power, or Pout servo should be output as servo_output_power_dbm.
- VCC and VDD should be output inside dut_bias.
- Bandwidth should be output as bandwidth_mhz.
- Duty cycle should be output as duty_cycle_pct.
- DUT mode should be output as dut_mode_name.
- Use null for fields that do not apply to a test type.
- Use arrays for fields that are active.

POWER CONTROL RULES:
- input_power_dbm and servo_output_power_dbm must always be arrays or null.
- Only one of input_power_dbm or servo_output_power_dbm may be non-null in a test.
- Use input_power_dbm when the user asks for source power, input power, Pin, generator power, fixed input power, or input power sweep.
- Use servo_output_power_dbm when the user asks for output power, output servo, servo output power, Pout, target output power, output power sweep, or power servo.
- If servo_output_power_dbm is active, input_power_dbm must be null.
- If input_power_dbm is active, servo_output_power_dbm must be null.
- Never put "input_power_dbm" or "servo_output_power_dbm" in sweep.expansion_order.
- Use "power" in sweep.expansion_order for either input_power_dbm or servo_output_power_dbm sweeps.
- If power has one value, it is a fixed condition.
- If power has multiple values, it is a swept condition.

SWEEP RULES:
- Use the list-based field format for both fixed and swept values.
- If a field has one value, it is fixed.
- If a field has multiple values, it is swept.
- sweep object stores execution order only.
- sweep object does not store values.
- If no field has multiple values, sweep can be null.
- If any field has multiple values, sweep must be non-null.
- Default sweep_type is "full_factorial".
- Use "zipped" only if the user explicitly says paired conditions, paired sweep, zip the values, one-to-one mapping, matching index, or pair by index.
- expansion_order is outer-to-inner loop order.
- If user does not specify expansion order, use this default order but include only axes that are actually swept:
  ["frequency_mhz", "vcc_v", "vdd_v", "temperature_c", "power", "duty_cycle_pct", "dut_mode_name", "modulation"]
- Only change expansion_order if the user explicitly asks for a specific nesting order.
- Example: "for each frequency, sweep voltage, and for each voltage sweep duty cycle" means:
  expansion_order = ["frequency_mhz", "vcc_v", "duty_cycle_pct"]
- Example: "make duty cycle outermost, then voltage, then frequency" means:
  expansion_order = ["duty_cycle_pct", "vcc_v", "frequency_mhz"]
- If input_power_dbm has multiple values, sweep.expansion_order must include "power".
- If servo_output_power_dbm has multiple values, sweep.expansion_order must include "power".
- If frequency_mhz has multiple values, sweep.expansion_order must include "frequency_mhz".
- If vcc_v has multiple values, sweep.expansion_order must include "vcc_v".
- If vdd_v has multiple values, sweep.expansion_order must include "vdd_v".
- If temperature_c has multiple values, sweep.expansion_order must include "temperature_c".
- If bandwidth_mhz has multiple values, sweep.expansion_order must include "bandwidth_mhz".
- If duty_cycle_pct has multiple values, sweep.expansion_order must include "duty_cycle_pct".
- If dut_mode_name has multiple values, sweep.expansion_order must include "dut_mode_name".
- If modulation has multiple values, sweep.expansion_order must include "modulation".

RANGE EXPANSION RULES:
- If the user gives a numeric range and a step size, expand it into a full list.
- Example: "input power from 0 to 10 dBm in 2 dB steps" becomes [0, 2, 4, 6, 8, 10].
- Example: "servo output power from 25 to 30 dBm in 1 dB steps" becomes [25, 26, 27, 28, 29, 30].
- Example: "frequency from 2400 to 2500 MHz in 50 MHz steps" becomes [2400, 2450, 2500].
- If user gives power range without step size and asks to use defaults, use 1 dB step.
- If user gives frequency range without step size and asks to use defaults, use 5 MHz step only for narrow RF demo ranges, otherwise ask for clarification.
- If user gives voltage range without step size and asks to use defaults, ask for clarification unless the values are explicitly enumerated.
- If user gives duty cycle range without step size and asks to use defaults, ask for clarification unless the values are explicitly enumerated.
- If user gives a range without step size and does not ask for defaults, ask for the step size.

MEASUREMENT METRIC RULES:
- For GAIN, measurement.metric must be "GAIN".
- For EVM, measurement.metric must be "EVM".
- For ACPR, measurement.metric must be "ACPR".
- For CURRENT, measurement.metric must be "CURRENT".

ADDITIONAL MEASUREMENT RULES:
- measurement.metric is the primary metric of the test.
- If the user asks to measure, capture, log, or record input power, add "PIN" to measurement.additional_metrics.
- If the user asks to measure, capture, log, or record output power, add "POUT" to measurement.additional_metrics.
- If the user asks to measure, capture, log, or record current, supply current, ICC, or IDD, add "CURRENT" to measurement.additional_metrics.
- Do not add PIN, POUT, or CURRENT unless the user explicitly asks for them.
- Do not change the main test_type because of additional metrics.
- Example: An EVM test that also measures input and output power should still have test_type = "EVM" and measurement.metric = "EVM", with measurement.additional_metrics = ["PIN", "POUT"].

EVM RULES:
- evm_settings must be non-null for EVM tests.
- evm_settings.evm_type must be either "STATIC" or "DYNAMIC".
- If user says dynamic EVM, evm_type must be "DYNAMIC".
- If user says static EVM, evm_type must be "STATIC".
- If user says EVM but does not specify static or dynamic and does not ask for defaults, ask which type.
- If user asks for EVM with default values, use STATIC.
- For EVM, either input_power_dbm or servo_output_power_dbm must be active.
- For dynamic EVM, duty_cycle_pct is required in strict mode and defaulted to [50] in demo default mode.
- If duty_cycle_pct is active, evm_type must be "DYNAMIC".
- If duty_cycle_pct has multiple values, evm_type must be "DYNAMIC".

DYNAMIC EVM DUT MODE SWITCHING RULES:
- For EVM tests, dut_mode_name is the active DUT mode/state.
- dut_mode_name may be a list because the same test can sweep across multiple active DUT modes.
- dut_alternate_mode_name is the inactive/alternate DUT mode/state used during dynamic EVM switching.
- dut_alternate_mode_name must be a single string, not a list.
- For STATIC EVM, there is no DUT mode switching, so dut_alternate_mode_name must be null and duty_cycle_pct must be null.
- For DYNAMIC EVM, duty_cycle_pct is required.
- For DYNAMIC EVM, dut_alternate_mode_name is required.
- If DYNAMIC EVM is requested and the user explicitly allows defaults, set dut_alternate_mode_name to "OFF" if the user did not specify it.
- If DYNAMIC EVM is requested and defaults are not allowed, ask for dut_alternate_mode_name when it is missing.
- Example: "dynamic EVM with TX_HIGH_POWER at 50% duty cycle" means dut_mode_name = ["TX_HIGH_POWER"], dut_alternate_mode_name should be asked for unless defaults are allowed.
- Example: "dynamic EVM switching between TX_HIGH_POWER and OFF" means dut_mode_name = ["TX_HIGH_POWER"], dut_alternate_mode_name = "OFF".

EVM SERVO AND DUTY-CYCLE RULES:
- If the user provides servo_output_power_dbm or says "servo to output power", then input_power_dbm must be null.
- Servoing to output power means the bench/test software adjusts input power internally to reach the requested output power.
- If servo_output_power_dbm is present, do not default input_power_dbm.
- If the user provides duty_cycle_pct or says "duty cycle", then evm_type must be "DYNAMIC".
- STATIC EVM must not include duty_cycle_pct unless the user explicitly requests a static test with duty cycle metadata.
- If evm_type is DYNAMIC and duty_cycle_pct is missing, ask for duty_cycle_pct in strict mode or default it to [50] in demo default mode.

ACPR RULES:
- acpr_band_pairs must contain at least one pair.
- Each pair must include lower_offset_mhz and upper_offset_mhz.
- If user says "offset pair -10 MHz and +10 MHz", output lower_offset_mhz = -10 and upper_offset_mhz = 10.
- If user asks for ACPR with default values, use one pair: -10 MHz and +10 MHz, bandwidth 1 MHz.
- For ACPR, input_power_dbm should be null unless the user explicitly provides it.
- ACPR is normally servoed to a target output power using servo_output_power_dbm.

MODULATION RULES:
- For GAIN and CURRENT, modulation can be "CW" only if the user explicitly says CW or describes an unmodulated tone.
- If user does not specify modulation in strict engineering mode, ask for it.
- If user asks for defaults, use ["802.11ax_HE20"].
- For EVM and ACPR, modulation cannot be CW unless the user explicitly requests a special CW-style measurement.

MODULATION / BANDWIDTH RULES:
- bandwidth_mhz is not an independent sweep axis.
- Do not include "bandwidth_mhz" in sweep.expansion_order.
- modulation may be a sweep axis.
- If modulation implies bandwidth, set bandwidth_mhz to the corresponding list in the same order as modulation.
- Example: modulation ["802.11ax_HE20", "802.11ax_HE160"] should have bandwidth_mhz [20, 160].
- The bench config generator will pair modulation[i] with bandwidth_mhz[i].
- Do not create full-factorial combinations between modulation and bandwidth.

S-PARAMETER LANGUAGE MAPPING:
- "input return loss", "return loss", "input match", or "S11" usually maps to measurements ["S11"].
- "insertion loss", "insertion gain", "forward gain", "forward transmission", or "S21" usually maps to ["S21"].
- "reverse isolation", "reverse transmission", or "S12" usually maps to ["S12"].
- "output return loss", "output match", or "S22" usually maps to ["S22"].
- "all 2-port S-parameters" maps to ["S11", "S21", "S12", "S22"] and ports [1, 2].
- If user says ports 1 and 2, use ports [1, 2].
- If user says 4-port S-parameters, use ports [1, 2, 3, 4] and ask for specific measurements unless they say all.
- Sij is valid only if both i and j are present in ports.

SAFETY / VALIDITY RULES:
- Do not use input_power_dbm greater than 10 unless user explicitly provides it.
- Do not use servo_output_power_dbm greater than 30 unless user explicitly provides it.
- Do not use vcc_v or vdd_v greater than 5 unless user explicitly provides it.
- If the user provides unsafe values, still return a ready test case if all required fields are present; the deterministic validator will flag safety issues later.

OUTPUT RULES:
- Use the provided test_id exactly when creating a ready test case.
- status must be either "ready" or "needs_clarification".
- clarifying_questions must be an array.
- default_assumptions must be an array.
- test_case must be null when status is "needs_clarification".
- test_case must be non-null when status is "ready".
- Return exactly one JSON object.
"""


def _is_active(values) -> bool:
    return values is not None and len(values) > 0


def _is_sweep(values) -> bool:
    return values is not None and len(values) > 1


def _dedupe_preserve_order(values):
    if not _is_active(values):
        return None

    cleaned = []

    for value in values:
        if value not in cleaned:
            cleaned.append(value)

    return cleaned or None


def _normalize_list_field(values):
    """
    Pydantic should already convert scalars to lists through rf_models.py.
    This is just a second deterministic cleanup layer.
    """
    return _dedupe_preserve_order(values)


def _active_power_field(test_case: RFTestCase):
    stim = test_case.stimulus

    if _is_active(stim.input_power_dbm):
        return "input_power_dbm"

    if _is_active(stim.servo_output_power_dbm):
        return "servo_output_power_dbm"

    return None


def _active_power_values(test_case: RFTestCase):
    stim = test_case.stimulus
    power_field = _active_power_field(test_case)

    if power_field == "input_power_dbm":
        return stim.input_power_dbm

    if power_field == "servo_output_power_dbm":
        return stim.servo_output_power_dbm

    return None


def _axis_values(test_case: RFTestCase, axis_name: str):
    stim = test_case.stimulus
    bias = test_case.dut_bias
    env = test_case.environment

    if axis_name == "frequency_mhz":
        return stim.frequency_mhz

    if axis_name == "power":
        return _active_power_values(test_case)

    if axis_name == "vcc_v":
        return bias.vcc_v

    if axis_name == "vdd_v":
        return bias.vdd_v

    if axis_name == "temperature_c":
        return env.temperature_c

    if axis_name == "duty_cycle_pct":
        return stim.duty_cycle_pct

    if axis_name == "dut_mode_name":
        return stim.dut_mode_name

    if axis_name == "modulation":
        return stim.modulation

    return None


def _multi_value_axes(test_case: RFTestCase) -> list:
    axes = []

    for axis in DEFAULT_SWEEP_EXPANSION_ORDER:
        values = _axis_values(test_case, axis)

        if _is_sweep(values):
            axes.append(axis)

    return axes


def _apply_default_sweep_order(test_case: RFTestCase) -> None:
    """
    If there are swept fields and no explicit expansion order, apply the
    default order.

    If there is an explicit expansion order, preserve it but append missing
    swept axes using the default order.
    """
    multi_axes = _multi_value_axes(test_case)

    if not multi_axes:
        test_case.sweep = None
        return

    if test_case.sweep is None:
        test_case.sweep = RFSweep(
            sweep_type="full_factorial",
            expansion_order=[
                axis for axis in DEFAULT_SWEEP_EXPANSION_ORDER
                if axis in multi_axes
            ],
        )
        return

    if not test_case.sweep.sweep_type:
        test_case.sweep.sweep_type = "full_factorial"

    existing_order = test_case.sweep.expansion_order or []

    # Remove invalid or inactive axes.
    cleaned_existing_order = [
        axis
        for axis in existing_order
        if axis in DEFAULT_SWEEP_EXPANSION_ORDER and _is_active(_axis_values(test_case, axis))
    ]

    # Important: never allow physical power names in order. The schema should
    # block these already, but this helps if raw LLM output is weird before validation.
    cleaned_existing_order = [
        "power" if axis in ["input_power_dbm", "servo_output_power_dbm"] else axis
        for axis in cleaned_existing_order
    ]

    cleaned_existing_order = _dedupe_preserve_order(cleaned_existing_order) or []

    missing_axes = [
        axis
        for axis in DEFAULT_SWEEP_EXPANSION_ORDER
        if axis in multi_axes and axis not in cleaned_existing_order
    ]

    test_case.sweep.expansion_order = cleaned_existing_order + missing_axes

    if not test_case.sweep.expansion_order:
        test_case.sweep = None

def _infer_bandwidth_from_modulation_name(modulation: str) -> float | None:
    """
    Infer WLAN bandwidth from common modulation names.

    Examples:
        802.11ax_HE20       -> 20
        802.11ax_HE40       -> 40
        802.11ax_HE80       -> 80
        802.11ax_HE160      -> 160
        AC160-MCS9-300US    -> 160
        VHT80-MCS7          -> 80
    """
    text = (modulation or "").upper()

    if any(token in text for token in ["HE160", "VHT160", "EHT160", "AC160", "BW160", "160MHZ"]):
        return 160.0

    if any(token in text for token in ["HE80", "VHT80", "EHT80", "AC80", "BW80", "80MHZ"]):
        return 80.0

    if any(token in text for token in ["HE40", "VHT40", "EHT40", "AC40", "BW40", "40MHZ"]):
        return 40.0

    if any(token in text for token in ["HE20", "VHT20", "EHT20", "AC20", "BW20", "20MHZ"]):
        return 20.0

    return None


def _remove_bandwidth_from_sweep_order(test_case: RFTestCase) -> None:
    """
    bandwidth_mhz is allowed in stimulus, but it is not an independent sweep axis.
    It is either fixed metadata or paired to modulation.
    """
    if test_case.sweep is None:
        return

    if not test_case.sweep.expansion_order:
        return

    test_case.sweep.expansion_order = [
        axis for axis in test_case.sweep.expansion_order
        if axis != "bandwidth_mhz"
    ]


def _normalize_modulation_bandwidth_pairing(test_case: RFTestCase) -> list[str]:
    """
    Keep bandwidth_mhz paired to modulation.

    Rules:
    - bandwidth_mhz must not be in sweep.expansion_order.
    - If modulation values imply bandwidths, use those bandwidths.
    - If there are multiple modulations, bandwidth_mhz may have the same length,
      where bandwidth_mhz[i] belongs to modulation[i].
    - Do not full-factorial expand bandwidth independently.
    """
    assumptions: list[str] = []

    stim = test_case.stimulus

    _remove_bandwidth_from_sweep_order(test_case)

    if not _is_active(stim.modulation):
        return assumptions

    modulations = list(stim.modulation or [])
    inferred_bandwidths: list[float | None] = [
        _infer_bandwidth_from_modulation_name(m) for m in modulations
    ]

    has_any_inferred = any(bw is not None for bw in inferred_bandwidths)
    has_all_inferred = all(bw is not None for bw in inferred_bandwidths)

    if not has_any_inferred:
        return assumptions

    # If all modulation names imply BW, make bandwidth list match modulation list.
    if has_all_inferred:
        new_bandwidths = [float(bw) for bw in inferred_bandwidths if bw is not None]

        if stim.bandwidth_mhz != new_bandwidths:
            stim.bandwidth_mhz = new_bandwidths
            assumptions.append(
                f"bandwidth_mhz paired to modulation as {new_bandwidths}; bandwidth_mhz is not an independent sweep axis"
            )

        return assumptions

    # Partial inference is risky: keep existing bandwidth but do not sweep bandwidth.
    assumptions.append(
        "bandwidth_mhz was not expanded because it is metadata paired to modulation, not a sweep axis"
    )

    return assumptions

def _normalize_evm_dynamic_switching(test_case: RFTestCase, defaults_allowed: bool = False) -> list[str]:
    """
    Enforce dynamic/static EVM switching rules.

    STATIC EVM:
        - no duty cycle
        - no alternate DUT mode

    DYNAMIC EVM:
        - duty cycle required
        - alternate DUT mode required
        - if defaults are allowed, alternate mode defaults to "OFF"
    """
    assumptions: list[str] = []

    if test_case.test_type != "EVM":
        return assumptions

    stim = test_case.stimulus

    if test_case.evm_settings is None:
        return assumptions

    evm_type = test_case.evm_settings.evm_type

    if evm_type == "STATIC":
        if _is_active(stim.duty_cycle_pct):
            stim.duty_cycle_pct = None
            assumptions.append("duty_cycle_pct set to null because STATIC EVM does not use duty-cycle switching")

        if getattr(stim, "dut_alternate_mode_name", None):
            stim.dut_alternate_mode_name = None
            assumptions.append("dut_alternate_mode_name set to null because STATIC EVM does not use DUT mode switching")

        return assumptions

    if evm_type == "DYNAMIC":
        # If dynamic EVM has duty cycle but alternate mode is missing,
        # use OFF only when defaults are explicitly allowed.
        if not getattr(stim, "dut_alternate_mode_name", None) and defaults_allowed:
            stim.dut_alternate_mode_name = "OFF"
            assumptions.append('dut_alternate_mode_name defaulted to "OFF" for dynamic EVM')

    return assumptions

def _normalize_power_fields(test_case: RFTestCase) -> None:
    """
    RF rule:
    Only one power control method can be active:
    - input_power_dbm
    - servo_output_power_dbm

    If servo power exists, input power is cleared because servo output power
    means the bench controls input power internally.
    """
    stim = test_case.stimulus

    stim.input_power_dbm = _normalize_list_field(stim.input_power_dbm)
    stim.servo_output_power_dbm = _normalize_list_field(stim.servo_output_power_dbm)

    if _is_active(stim.servo_output_power_dbm):
        stim.input_power_dbm = None


def _normalize_all_list_fields(test_case: RFTestCase) -> None:
    stim = test_case.stimulus
    bias = test_case.dut_bias
    env = test_case.environment

    stim.frequency_mhz = _normalize_list_field(stim.frequency_mhz)
    stim.dut_mode_name = _normalize_list_field(stim.dut_mode_name)
    stim.modulation = _normalize_list_field(stim.modulation)
    stim.bandwidth_mhz = _normalize_list_field(stim.bandwidth_mhz)
    stim.duty_cycle_pct = _normalize_list_field(stim.duty_cycle_pct)

    bias.vcc_v = _normalize_list_field(bias.vcc_v)
    bias.vdd_v = _normalize_list_field(bias.vdd_v)

    env.temperature_c = _normalize_list_field(env.temperature_c)

    if not _is_active(env.temperature_c):
        env.temperature_c = [25.0]

def _detect_additional_measurement_metrics(text: str) -> list[str]:
    """
    Detect extra bench values the user wants to log in addition to the main metric.

    Example:
        Main metric: EVM
        User says: "also measure input power and output power"
        Result: ["PIN", "POUT"]

    This does not change the primary measurement metric.
    """
    msg = (text or "").lower()
    metrics: list[str] = []

    input_power_phrases = [
        "measure input power",
        "measure pin",
        "capture input power",
        "capture pin",
        "log input power",
        "log pin",
        "record input power",
        "record pin",
        "input power too",
        "pin too",
        "pin and pout",
        "input and output power",
        "input/output power",
        "input output power",
    ]

    output_power_phrases = [
        "measure output power",
        "measure pout",
        "capture output power",
        "capture pout",
        "log output power",
        "log pout",
        "record output power",
        "record pout",
        "output power too",
        "pout too",
        "pin and pout",
        "input and output power",
        "input/output power",
        "input output power",
    ]

    current_phrases = [
        "measure current",
        "capture current",
        "log current",
        "record current",
        "current too",
        "supply current",
        "measure supply current",
        "capture supply current",
        "log supply current",
        "icc",
        "idd",
        "current consumption",
    ]

    if any(phrase in msg for phrase in input_power_phrases):
        metrics.append("PIN")

    if any(phrase in msg for phrase in output_power_phrases):
        metrics.append("POUT")

    if any(phrase in msg for phrase in current_phrases):
        metrics.append("CURRENT")

    return metrics


def _normalize_additional_measurement_metrics(test_case: RFTestCase, user_message: str = "") -> None:
    """
    Adds optional additional measurement metrics requested by the user.

    Keeps:
        measurement.metric = main test metric, e.g. EVM

    Adds:
        measurement.additional_metrics = ["PIN", "POUT", "CURRENT"]

    Only adds these if the user explicitly asks.
    """
    if test_case.measurement is None:
        return

    existing = list(test_case.measurement.additional_metrics or [])
    requested = _detect_additional_measurement_metrics(user_message)

    for metric in requested:
        if metric not in existing:
            existing.append(metric)

    test_case.measurement.additional_metrics = existing

def normalize_rf_test_case(
    test_case: RFTestCase,
    user_message: str = "",
    defaults_allowed: bool = False,
) -> RFTestCase:
    """
    Deterministic cleanup after LLM generation.

    Pydantic validation guarantees the required nested RF objects exist before
    this function runs. This function then makes the output bench-consistent.
    """
    _normalize_all_list_fields(test_case)
    _normalize_power_fields(test_case)
    _normalize_additional_measurement_metrics(test_case, user_message)
    _normalize_evm_dynamic_switching(test_case, defaults_allowed=defaults_allowed)
    _normalize_modulation_bandwidth_pairing(test_case)

    # Duty cycle implies dynamic EVM.
    if test_case.test_type == "EVM":
        if _is_active(test_case.stimulus.duty_cycle_pct):
            if test_case.evm_settings is not None:
                test_case.evm_settings.evm_type = "DYNAMIC"

    # ACPR is normally servoed to output power.
    if test_case.test_type == "ACPR":
        if _is_active(test_case.stimulus.servo_output_power_dbm):
            test_case.stimulus.input_power_dbm = None

    if test_case.test_type == "S_PARAMETER":
        test_case.bench_type = "S_PARAMETER_BENCH"
        test_case.evm_settings = None
        test_case.sweep = None

        # S-parameter uses VNA power, not RF bench power fields.
        test_case.stimulus.input_power_dbm = None
        test_case.stimulus.servo_output_power_dbm = None
        test_case.stimulus.frequency_mhz = None
        test_case.stimulus.modulation = None
        test_case.stimulus.bandwidth_mhz = None
        test_case.stimulus.duty_cycle_pct = None
        test_case.stimulus.acpr_band_pairs = None

        if not _is_active(test_case.stimulus.dut_mode_name):
            test_case.stimulus.dut_mode_name = ["DEFAULT_SPARAM_MODE"]

        if not _is_active(test_case.environment.temperature_c):
            test_case.environment.temperature_c = [25.0]

        return test_case

    _apply_default_sweep_order(test_case)

    return test_case



def _set_default_limit(test_case: RFTestCase) -> None:
    if test_case.measurement.limit is not None:
        return

    limit_defaults = {
        "GAIN": {"operator": ">=", "value": 25, "unit": "dB"},
        "EVM": {"operator": "<=", "value": -32, "unit": "dB"},
        "ACPR": {"operator": "<=", "value": -45, "unit": "dBc"},
        "CURRENT": {"operator": "<=", "value": 500, "unit": "mA"},
    }

    default_limit = limit_defaults.get(test_case.test_type)
    if default_limit:
        from app.models.rf_models import RFLimit
        test_case.measurement.limit = RFLimit(**default_limit)


def apply_demo_defaults(test_case: RFTestCase, user_message: str = "") -> tuple[RFTestCase, list[str]]:
    """
    Fill missing required MVP fields while preserving anything the user/LLM already set.
    Used only when the user asked for defaults/demo values.

    MVP measurement behavior:
    - measurement.metric stays as the primary metric, for example EVM.
    - measurement.additional_metrics can include optional extra logged metrics:
        PIN, POUT, CURRENT
    - Those are added by normalize_rf_test_case(test_case, user_message)
      if the user explicitly asks for them.
    """
    assumptions: list[str] = []
    stim = test_case.stimulus
    bias = test_case.dut_bias
    env = test_case.environment

    def set_if_missing(obj, attr: str, value, label: str):
        if not _is_active(getattr(obj, attr, None)):
            setattr(obj, attr, value)
            assumptions.append(label)

    # Make sure old/generated measurement objects have the new field.
    if getattr(test_case.measurement, "additional_metrics", None) is None:
        test_case.measurement.additional_metrics = []

    if test_case.test_type == "S_PARAMETER":
        test_case.bench_type = "S_PARAMETER_BENCH"
        test_case.evm_settings = None
        test_case.sweep = None
        stim.frequency_mhz = None
        stim.input_power_dbm = None
        stim.servo_output_power_dbm = None
        stim.modulation = None
        stim.bandwidth_mhz = None
        stim.duty_cycle_pct = None
        stim.acpr_band_pairs = None

        # S-parameter measurements are stored in s_parameter_settings.measurements.
        # Do not use RF bench additional metrics like PIN/POUT/CURRENT here.
        test_case.measurement.additional_metrics = []

        if not _is_active(stim.dut_mode_name):
            stim.dut_mode_name = ["DEFAULT_SPARAM_MODE"]
            assumptions.append("dut_mode_name defaulted to ['DEFAULT_SPARAM_MODE']")

        if test_case.s_parameter_settings is None:
            from app.models.rf_models import RFSParameterSettings

            test_case.s_parameter_settings = RFSParameterSettings(
                ports=[1, 2],
                measurements=["S11", "S21"],
                frequency_start_mhz=2400,
                frequency_stop_mhz=2500,
                num_points=401,
                if_bandwidth_hz=1000,
                vna_power_dbm=-10,
                calibration_type="SOLT",
                sweep_mode="linear",
                result_format="DB_ANGLE",
            )
            assumptions.append("S-parameter VNA settings defaulted to 2400-2500 MHz, 401 points, -10 dBm, SOLT")

        if not _is_active(env.temperature_c):
            env.temperature_c = [25.0]
            assumptions.append("temperature_c defaulted to [25 C]")

        if test_case.measurement.metric != "S_PARAMETER":
            test_case.measurement.metric = "S_PARAMETER"

        return normalize_rf_test_case(test_case, user_message, defaults_allowed=True,), assumptions

    set_if_missing(stim, "frequency_mhz", [2450.0], "frequency_mhz defaulted to [2450 MHz]")
    set_if_missing(stim, "dut_mode_name", ["DEMO_HIGH_POWER"], "dut_mode_name defaulted to ['DEMO_HIGH_POWER']")
    set_if_missing(stim, "modulation", ["802.11ax_HE20"], "modulation defaulted to ['802.11ax_HE20']")
    set_if_missing(bias, "vcc_v", [3.3], "vcc_v defaulted to [3.3 V]")
    set_if_missing(bias, "vdd_v", [1.8], "vdd_v defaulted to [1.8 V]")

    if not _is_active(env.temperature_c):
        env.temperature_c = [25.0]
        assumptions.append("temperature_c defaulted to [25 C]")

    if test_case.test_type in {"GAIN", "CURRENT"}:
        if not _is_active(stim.input_power_dbm) and not _is_active(stim.servo_output_power_dbm):
            stim.input_power_dbm = [-20.0]
            assumptions.append("input_power_dbm defaulted to [-20 dBm]")

        # GAIN/CURRENT default mode uses fixed input power, not output-power servo.
        stim.servo_output_power_dbm = None

    if test_case.test_type == "EVM":
        if not _is_active(stim.bandwidth_mhz):
            stim.bandwidth_mhz = [20.0]
            assumptions.append("bandwidth_mhz defaulted to [20 MHz]")

        if test_case.evm_settings is None:
            from app.models.rf_models import RFEVMSettings

            test_case.evm_settings = RFEVMSettings(evm_type="STATIC")
            assumptions.append("evm_type defaulted to STATIC")

        if test_case.evm_settings.evm_type == "DYNAMIC" and not _is_active(stim.duty_cycle_pct):
            stim.duty_cycle_pct = [50.0]
            assumptions.append("duty_cycle_pct defaulted to [50%] for dynamic EVM")

        if test_case.evm_settings.evm_type == "DYNAMIC" and not getattr(stim, "dut_alternate_mode_name", None):
            stim.dut_alternate_mode_name = "OFF"
            assumptions.append('dut_alternate_mode_name defaulted to "OFF" for dynamic EVM')

        if not _is_active(stim.input_power_dbm) and not _is_active(stim.servo_output_power_dbm):
            stim.servo_output_power_dbm = [18.0]
            assumptions.append("servo_output_power_dbm defaulted to [18 dBm]")

    if test_case.test_type == "ACPR":
        stim.input_power_dbm = None

        if not _is_active(stim.servo_output_power_dbm):
            stim.servo_output_power_dbm = [18.0]
            assumptions.append("servo_output_power_dbm defaulted to [18 dBm]")

        if not stim.acpr_band_pairs:
            from app.models.rf_models import ACPRBandPair

            stim.acpr_band_pairs = [
                ACPRBandPair(lower_offset_mhz=-10.0, upper_offset_mhz=10.0, bandwidth_mhz=1.0)
            ]
            assumptions.append("ACPR band pair defaulted to -10/+10 MHz with 1 MHz bandwidth")

    _set_default_limit(test_case)

    # This is where user-requested extra metrics are added.
    # Example:
    #   "also measure input and output power"
    # becomes:
    #   measurement.additional_metrics = ["PIN", "POUT"]
    test_case = normalize_rf_test_case(test_case, user_message, defaults_allowed=True,)

    return test_case, assumptions


def build_demo_test_case(test_type: str, test_id: str) -> RFTestCase:
    from app.models.rf_models import RFEnvironment, RFMeasurement, RFBias, RFStimulus

    test_type = test_type.upper().strip()

    if test_type not in {"GAIN", "EVM", "ACPR", "CURRENT", "S_PARAMETER"}:
        test_type = "GAIN"

    bench_type = "S_PARAMETER_BENCH" if test_type == "S_PARAMETER" else "RF_BENCH"

    base = RFTestCase(
        test_id=test_id,
        test_type=test_type,
        bench_type=bench_type,
        objective=f"Demo {test_type} RF test generated from chat defaults",
        stimulus=RFStimulus(),
        dut_bias=RFBias(),
        environment=RFEnvironment(),
        measurement=RFMeasurement(metric=test_type),
    )

    completed, _assumptions = apply_demo_defaults(base)
    return completed


def generate_rf_test_from_chat(
    user_message: str,
    chat_context: str = "",
) -> RFGenerationResponse:
    test_id = f"RF_TEST_{str(uuid.uuid4())[:8].upper()}"

    user_prompt = f"""
Recent chat context:
{chat_context}

Latest user request:
{user_message}

Generate one RF test generation response.

Use this test_id if a test case is ready:
{test_id}

Remember:
- All active RF fields must be arrays.
- null means not used.
- [one value] means fixed.
- [multiple values] means swept.
- sweep.expansion_order uses "power" for input_power_dbm or servo_output_power_dbm sweeps.
- Never use input_power_dbm or servo_output_power_dbm inside sweep.expansion_order.
"""

    requested_defaults = user_requested_defaults(user_message)
    local_test_type = (
        detect_test_type_from_text(user_message)
        or detect_test_type_from_text(chat_context)
    )

    if is_small_talk(user_message):
        return response_from_clarification(
            ["Hi, I can help generate RF tests. Which test type do you want: GAIN, EVM, ACPR, CURRENT, or S_PARAMETER?"],
            [],
        )

    # If the user explicitly asks for defaults and we know the type, we can
    # always recover with a deterministic demo test even if the model fails.
    try:
        raw_json = call_llm_for_json(
            system_prompt=RF_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        safe_json = coerce_generation_json(raw_json, test_id)
        response = RFGenerationResponse.model_validate(safe_json)
    except (ValueError, ValidationError, RuntimeError):
        if requested_defaults and local_test_type:
            test_case = build_demo_test_case(local_test_type, test_id)
            return RFGenerationResponse(
                status="ready",
                clarifying_questions=[],
                default_assumptions=[
                    "LLM output was not usable, so TestBridge generated a deterministic demo-default test instead.",
                ],
                test_case=test_case,
            )

        if requested_defaults:
            return response_from_clarification(
                ["I see you want to use defaults, but I don't know what test type you want to generate. Please reply with 'EVM', 'GAIN', 'ACPR', 'CURRENT', or 'S_PARAMETER'."],
                [],
            )

        return response_from_clarification(
            [
                "I could not convert that message into a valid RF test object yet. Please provide the test type and key bench parameters, or reply 'use defaults' after selecting a test type."
            ],
            default_assumptions_for_missing(),
        )

    if response.status == "needs_clarification":
        return response

    if response.test_case is None:
        return response_from_clarification(
            ["The generated response said it was ready, but no test case was returned. Please provide the missing RF parameters or reply 'use defaults'."],
            default_assumptions_for_missing(),
        )

    response.test_case = RFTestCase.model_validate(response.test_case.model_dump())

    # Guard against accidental test-type inference from DUT mode strings like TX HIGH GAIN.
    if local_test_type and response.test_case.test_type != local_test_type:
        response.test_case.test_type = local_test_type
        response.test_case.measurement.metric = local_test_type
        response.test_case.bench_type = "S_PARAMETER_BENCH" if local_test_type == "S_PARAMETER" else "RF_BENCH"

    response.test_case = normalize_rf_test_case(response.test_case, user_message, defaults_allowed=requested_defaults,)

    if requested_defaults:
        response.test_case, deterministic_assumptions = apply_demo_defaults(response.test_case, user_message)
        response.default_assumptions.extend(deterministic_assumptions)

    validation = validate_rf_test_case(response.test_case)
    missing_questions = questions_from_validation_errors(validation.get("errors", []))

    if (
            response.test_case
            and response.test_case.test_type == "EVM"
            and response.test_case.evm_settings is not None
            and response.test_case.evm_settings.evm_type == "DYNAMIC"
    ):
        if not _is_active(response.test_case.stimulus.duty_cycle_pct):
            q = "What duty cycle percentage should I use for dynamic EVM?"
            if q not in missing_questions:
                missing_questions.append(q)

        if not getattr(response.test_case.stimulus, "dut_alternate_mode_name", None):
            q = "What alternate/inactive DUT mode should I use for dynamic EVM switching? You can say OFF."
            if q not in missing_questions:
                missing_questions.append(q)

    # Missing required fields should become a chat clarification, not a broken/invalid JSON panel.
    if missing_questions and not requested_defaults:
        return response_from_clarification(
            missing_questions,
            default_assumptions_for_missing(),
        )

    # Defaults were requested but the LLM still produced an incomplete test.
    # Recover by building a complete deterministic demo test for the detected/returned type.
    if missing_questions and requested_defaults:
        fallback_type = local_test_type or response.test_case.test_type
        test_case = build_demo_test_case(fallback_type, test_id)
        return RFGenerationResponse(
            status="ready",
            clarifying_questions=[],
            default_assumptions=response.default_assumptions + [
                "Missing required fields were filled using deterministic demo defaults.",
            ],
            test_case=test_case,
        )

    return response
from typing import List
import uuid

from app.core.llm_client import call_llm_for_json
from app.models.rf_models import (
    DEFAULT_SWEEP_EXPANSION_ORDER,
    RFSpecGenerationResponse,
    RFTestCase,
    RFSweep,
)


SPEC_SUPPORTED_TEST_TYPES = {"GAIN", "EVM", "ACPR", "CURRENT", "P1DB"}

# bandwidth_mhz is intentionally not a sweep axis.
# It is paired metadata with modulation.
SPEC_ALLOWED_SWEEP_AXES = {
    "frequency_mhz",
    "power",
    "vcc_v",
    "vdd_v",
    "temperature_c",
    "duty_cycle_pct",
    "dut_mode_name",
    "modulation",
}


RF_SPEC_SYSTEM_PROMPT = """
You are Inventide RF TestBridge, an expert RF validation engineer.

Your job is to read RF product datasheet/specification text and generate suggested RF validation tests.

Return JSON only.
Do not include markdown.
Do not include explanations outside JSON.

IMPORTANT:
- You are reading datasheet/spec text, not a normal chat request.
- Be conservative.
- Do not invent tests from vague marketing claims.
- Generate only RF tests that are supported by evidence in the provided text.
- If exact values are missing but a reasonable datasheet-derived default is possible, include it in default_assumptions.
- If the datasheet clearly implies a common validation test but lacks some bench details, you may generate a suggested test with confidence "medium" or "low", and list the assumptions.
- Prefer fewer high-quality suggested tests over many weak tests.

MVP SUPPORTED TEST TYPES:
1. GAIN
2. EVM
3. ACPR
4. CURRENT
5. P1DB

Do not generate S_PARAMETER tests in the SPEC upload MVP.

You must return this format:

{
  "status": "ready" or "no_tests_found",
  "summary": "short summary of what RF tests were found or why no tests were found",
  "generated_tests": [
    {
      "schema_version": "inventide.rf.testcase.v1",
      "domain": "RF",
      "test_id": "string",
      "test_type": "GAIN | EVM | ACPR | CURRENT | P1DB",
      "bench_type": "RF_BENCH",
      "objective": "string",
      "stimulus": {
        "frequency_mhz": [number] or null,
        "dut_mode_name": ["string"] or null,
        "dut_alternate_mode_name": "string" or null,
        "modulation": ["string"] or null,
        "input_power_dbm": [number] or null,
        "servo_output_power_dbm": [number] or null,
        "bandwidth_mhz": [number] or null,
        "duty_cycle_pct": [number] or null,
        "acpr_band_pairs": [
          {
            "lower_offset_mhz": number,
            "upper_offset_mhz": number,
            "bandwidth_mhz": number or null
          }
        ] or null
      },
      "dut_bias": {
        "vcc_v": [number] or null,
        "vdd_v": [number] or null
      },
      "environment": {
        "temperature_c": [number] or null
      },
      "measurement": {
        "metric": "GAIN | EVM | ACPR | CURRENT | P1DB",
        "limit": {
          "operator": "<= | >= | == | < | >",
          "value": number,
          "unit": "string"
        } or null,
        "additional_metrics": ["PIN", "POUT", "CURRENT", "GAIN"],
        "measurement_method": "POWER_SWEEP | GAIN_COMPRESSION" or null,
        "compression_threshold_db": number or null,
        "settling_time_ms": number or null,
        "averages": number or null,
        "pin_start_dbm": number or null,
        "pin_stop_dbm": number or null,
        "pin_step_dbm": number or null
      },
      "evm_settings": {
        "evm_type": "STATIC | DYNAMIC"
      } or null,
      "sweep": {
        "sweep_type": "full_factorial | zipped",
        "expansion_order": [
          "frequency_mhz",
          "power",
          "vcc_v",
          "vdd_v",
          "temperature_c",
          "duty_cycle_pct",
          "dut_mode_name",
          "modulation"
        ]
      } or null,
      "source_evidence": [
        "short quote or evidence phrase from datasheet"
      ],
      "confidence": "low | medium | high",
      "execution_target": {
        "runner": "LabVIEW_RF_TestRunner",
        "config_format": "csv"
      }
    }
  ],
  "rejected_candidates": [
    {
      "candidate": "string",
      "reason": "string"
    }
  ],
  "default_assumptions": [
    "string"
  ]
}

CORE ARRAY FORMAT RULE:
- All active RF execution fields must be arrays except dut_alternate_mode_name.
- dut_alternate_mode_name is a single string or null.
- null means the field is not used or not applicable.
- [one value] means fixed condition.
- [multiple values] means swept condition.
- Do not return scalar RF values for frequency, modulation, power, bias, environment, duty cycle, or bandwidth.
- Correct: "frequency_mhz": [2450]
- Wrong: "frequency_mhz": 2450
- Correct: "servo_output_power_dbm": [18]
- Wrong: "servo_output_power_dbm": 18
- Correct: "temperature_c": [25]
- Correct: "dut_alternate_mode_name": "OFF"
- Wrong: "dut_alternate_mode_name": ["OFF", "SLEEP"]

DATASHEET EXTRACTION RULES:
- Use the datasheet text as the primary source of truth.
- Look for RF parameters such as frequency, band, channel, gain, EVM, ACPR/ACLR, output power, current, voltage rails, modulation, bandwidth, duty cycle, temperature, active mode, and alternate/inactive mode.
- Use source_evidence to cite the exact short phrase that supports the test.
- If a value is not present in the datasheet and must be assumed, list it in default_assumptions.
- Do not include unsupported guesses in source_evidence.
- If the datasheet says "Gain = 23 dB", create a GAIN test candidate if frequency and supply conditions can be found or reasonably defaulted.
- If the datasheet says "EVM", "802.11", "LTE", "NR", "modulated", or "error vector magnitude", create EVM tests when enough bench context is available.
- If the datasheet says "ACPR", "ACLR", "adjacent channel", or "spectrum mask", create ACPR tests when offset/bandwidth context is available or use conservative default ACPR offsets with low/medium confidence.
- If the datasheet gives supply current, ICC, IDD, quiescent current, or current consumption, create CURRENT tests if operating mode and supply rails are available or defaultable.
- If the datasheet says "P1dB", "1 dB compression", "compression point", "output power at 1dB compression", or "OP1dB", create P1DB tests when power sweep context is available or use conservative default power sweep range with low/medium confidence.

SUPPORTED TEST DETAILS:

GAIN:
Required or defaultable:
- frequency_mhz
- dut_mode_name
- modulation
- input_power_dbm
- vcc_v
- vdd_v
- temperature_c

Measurement:
- metric = "GAIN"
- If datasheet gives gain target/spec, use it as measurement.limit.
- Example: "Gain = 23 dB" becomes limit { "operator": ">=", "value": 23, "unit": "dB" }.
- If datasheet says PIN/POUT are measured or logged, add "PIN" and/or "POUT" to measurement.additional_metrics.

Default assumptions if missing:
- input_power_dbm = [-20]
- temperature_c = [25]
- modulation = ["CW"] only if datasheet implies CW/small-signal gain.
- dut_mode_name = ["DEFAULT_TX_MODE"] if no mode is given.
- vcc_v = [3.3] and vdd_v = [1.8] only if supply rails are not available and a demo test is still useful.

EVM:
Required or defaultable:
- frequency_mhz
- dut_mode_name
- modulation
- bandwidth_mhz
- vcc_v
- vdd_v
- either input_power_dbm or servo_output_power_dbm
- evm_settings

Measurement:
- metric = "EVM"
- If datasheet gives EVM limit such as -32 dB, use <= -32 dB.
- If EVM limit is given as percent, preserve percent.
- If datasheet says PIN/POUT/current are measured or logged during EVM, add them to measurement.additional_metrics.

Default assumptions if missing:
- evm_type = "STATIC" unless duty cycle, pulsed mode, burst mode, packet mode, dynamic mode, or switching mode is present.
- If duty cycle is present, evm_type = "DYNAMIC".
- If dynamic EVM is used and no alternate/inactive mode is specified, set dut_alternate_mode_name = "OFF" and list the assumption.
- If static EVM is used, dut_alternate_mode_name must be null and duty_cycle_pct must be null.
- If power target is absent but output power, Pout, rated output power, or PSAT is present, use servo_output_power_dbm.
- If no EVM limit exists, measurement.limit may be null.
- temperature_c = [25].

ACPR:
Required or defaultable:
- frequency_mhz
- dut_mode_name
- modulation
- servo_output_power_dbm
- vcc_v
- vdd_v
- acpr_band_pairs

Measurement:
- metric = "ACPR"
- If datasheet gives ACPR/ACLR limit, use it as <= value dBc.
- If no ACPR limit exists, measurement.limit may be null.
- If datasheet says PIN/POUT/current are measured or logged during ACPR, add them to measurement.additional_metrics.

Default assumptions if missing:
- input_power_dbm = null.
- servo_output_power_dbm may come from Pout/PSAT/rated output power.
- If adjacent offsets are missing but ACPR is clearly requested/specified, default one pair to -10 MHz and +10 MHz with 1 MHz bandwidth and mark confidence low or medium.
- temperature_c = [25].

CURRENT:
Required or defaultable:
- frequency_mhz
- dut_mode_name
- modulation
- input_power_dbm
- vcc_v
- vdd_v

Measurement:
- metric = "CURRENT"
- If datasheet gives current max, use <= value mA.
- If current is in A, convert to mA.
- If no current limit exists, do not generate CURRENT unless the datasheet clearly discusses current consumption.
- Do not put "CURRENT" into additional_metrics for a CURRENT test because CURRENT is already the primary metric.

Default assumptions if missing:
- input_power_dbm = [-20]
- temperature_c = [25].

P1DB:
Required or defaultable:
- frequency_mhz
- dut_mode_name
- modulation
- vcc_v
- vdd_v
- pin_start_dbm
- pin_stop_dbm
- pin_step_dbm

Measurement:
- metric = "P1DB"
- If datasheet gives P1dB limit such as OP1dB >= 18 dBm, use it as limit.
- P1dB uses power sweep via pin_start_dbm, pin_stop_dbm, pin_step_dbm instead of input_power_dbm or servo_output_power_dbm.
- measurement_method defaults to "POWER_SWEEP".
- compression_threshold_db defaults to 1.0.
- settling_time_ms defaults to 50.
- averages defaults to 20.
- If datasheet says PIN/POUT/current/gain are measured or logged during P1dB, add them to measurement.additional_metrics.

Default assumptions if missing:
- pin_start_dbm = -30 (at least 15 dB below expected IP1dB)
- pin_stop_dbm = +10 (at least 3 dB above expected IP1dB)
- pin_step_dbm = 0.5
- measurement_method = "POWER_SWEEP"
- compression_threshold_db = 1.0
- settling_time_ms = 50
- averages = 20
- temperature_c = [25].
- input_power_dbm = null (P1dB uses power sweep)
- servo_output_power_dbm = null (P1dB uses power sweep)

ADDITIONAL MEASUREMENT RULES:
- measurement.metric is the primary metric of the test.
- measurement.additional_metrics is only for extra logged values.
- If the datasheet/test text says input power is measured, logged, captured, or recorded, add "PIN".
- If the datasheet/test text says output power is measured, logged, captured, or recorded, add "POUT".
- If the datasheet/test text says current, supply current, ICC, or IDD is measured, logged, captured, or recorded as an extra value, add "CURRENT".
- If the datasheet/test text says gain is measured, logged, captured, or recorded as an extra value, add "GAIN".
- Do not change test_type because of additional_metrics.
- Do not add PIN/POUT/CURRENT/GAIN unless supported by the text or clearly required by the generated test context.

DYNAMIC EVM DUT MODE SWITCHING RULES:
- For EVM tests, dut_mode_name is the active DUT mode/state.
- dut_mode_name may be a list because the same test can sweep across multiple active DUT modes.
- dut_alternate_mode_name is the inactive/alternate DUT mode/state used during dynamic EVM switching.
- dut_alternate_mode_name must be a single string, not a list.
- For STATIC EVM, there is no DUT mode switching, so dut_alternate_mode_name must be null and duty_cycle_pct must be null.
- For DYNAMIC EVM, duty_cycle_pct is required.
- For DYNAMIC EVM, dut_alternate_mode_name is required.
- If DYNAMIC EVM is generated and the datasheet does not specify an alternate mode, set dut_alternate_mode_name to "OFF" and list that in default_assumptions.
- Example: "dynamic EVM with TX_HIGH_POWER at 50% duty cycle" means dut_mode_name = ["TX_HIGH_POWER"], dut_alternate_mode_name = "OFF" if not otherwise specified.
- Example: "dynamic EVM switching between TX_HIGH_POWER and RX-LNA" means dut_mode_name = ["TX_HIGH_POWER"], dut_alternate_mode_name = "RX-LNA".

MODULATION / BANDWIDTH RULES:
- bandwidth_mhz is not an independent sweep axis.
- Do not include "bandwidth_mhz" in sweep.expansion_order.
- modulation may be a sweep axis.
- If modulation implies bandwidth, set bandwidth_mhz to the corresponding list in the same order as modulation.
- Example: modulation ["802.11ax_HE20", "802.11ax_HE160"] should have bandwidth_mhz [20, 160].
- Example: modulation ["802.11ax_HE20", "AC160-MCS9-300US"] should have bandwidth_mhz [20, 160].
- The bench config generator will pair modulation[i] with bandwidth_mhz[i].
- Do not create full-factorial combinations between modulation and bandwidth.

POWER CONTROL RULES:
- input_power_dbm and servo_output_power_dbm must always be arrays or null.
- Only one of input_power_dbm or servo_output_power_dbm may be non-null in a test.
- Use input_power_dbm when the test is source-power driven, gain-driven, current-driven, Pin-driven, or small-signal.
- Use servo_output_power_dbm when the test is output-power driven, Pout-driven, PSAT-driven, output servo driven, or target output power driven.
- If servo_output_power_dbm is active, input_power_dbm must be null.
- If input_power_dbm is active, servo_output_power_dbm must be null.
- Never put "input_power_dbm" or "servo_output_power_dbm" in sweep.expansion_order.
- Use "power" in sweep.expansion_order for either input_power_dbm or servo_output_power_dbm sweeps.

SWEEP RULES:
- Use multiple values in the field itself when the datasheet gives multiple operating points, bands, voltages, powers, duty cycles, modes, or modulations that should be tested.
- bandwidth_mhz is not an independent sweep field.
- If modulation has multiple values and bandwidth_mhz has multiple values, bandwidth_mhz must be paired to modulation by index.
- If a field has one value, it is fixed.
- If a field has multiple values, it is swept.
- sweep object stores execution order only.
- sweep object does not store values.
- If no allowed sweep field has multiple values, sweep can be null.
- If any allowed sweep field has multiple values, sweep must be non-null.
- Default sweep_type is "full_factorial".
- Use "zipped" only if the datasheet clearly presents paired operating conditions that should be tested one-to-one.
- expansion_order is outer-to-inner loop order.
- Valid expansion_order axes are:
  ["frequency_mhz", "vcc_v", "vdd_v", "temperature_c", "power", "duty_cycle_pct", "dut_mode_name", "modulation"]
- If input_power_dbm has multiple values, sweep.expansion_order must include "power".
- If servo_output_power_dbm has multiple values, sweep.expansion_order must include "power".
- If frequency_mhz has multiple values, sweep.expansion_order must include "frequency_mhz".
- If vcc_v has multiple values, sweep.expansion_order must include "vcc_v".
- If vdd_v has multiple values, sweep.expansion_order must include "vdd_v".
- If temperature_c has multiple values, sweep.expansion_order must include "temperature_c".
- If duty_cycle_pct has multiple values, sweep.expansion_order must include "duty_cycle_pct".
- If dut_mode_name has multiple values, sweep.expansion_order must include "dut_mode_name".
- If modulation has multiple values, sweep.expansion_order must include "modulation".
- Never include "bandwidth_mhz" in sweep.expansion_order.

RANGE AND TABLE RULES:
- If the datasheet gives a range like 2400–2500 MHz, do not automatically expand every MHz.
- For datasheet ranges, prefer representative points: low, mid, high.
- Example: 2400–2500 MHz becomes [2400, 2450, 2500].
- If voltage range is given, use low and nominal/high if clear.
- Example: 3.0–3.6 V becomes [3.0, 3.3, 3.6] if nominal 3.3 is reasonable.
- If output power range is given, use representative points unless explicit step size is stated.
- Example: Pout 20–28 dBm becomes [20, 24, 28] unless step size is given.
- If a table gives exact operating points, use those exact values.
- If a table gives paired rows, use zipped only when values are clearly paired by row.

FREQUENCY RULES:
- Convert GHz to MHz.
- Example: 2.4 GHz becomes 2400.
- Example: 2.4–2.5 GHz becomes [2400, 2450, 2500].
- If a datasheet gives a frequency band but no exact center frequency, use low/mid/high representative points and list assumption.

MODULATION RULES:
- If the datasheet explicitly says CW, use ["CW"].
- If the datasheet says Wi-Fi, WLAN, 802.11, HE20, HT20, VHT, AC160, OFDM, use the most specific modulation/bandwidth found.
- If it says LTE or NR, use the most specific LTE/NR waveform found.
- Do not use CW for EVM unless the text explicitly describes a special CW-style measurement.
- If modulation is missing but the product context clearly suggests a wireless standard, use that standard with medium/low confidence and list the assumption.

DUT MODE RULES:
- Preserve mode names exactly when present.
- Examples: "TX HIGH GAIN", "High Power Mode", "Bypass Mode", "PA Enable", "2-stage mode".
- Do not infer test_type GAIN just because DUT mode contains "GAIN".
- If no mode exists, use ["DEFAULT_TX_MODE"] and list assumption.

EVM RULES:
- evm_settings must be non-null for EVM tests.
- If duty_cycle_pct exists, evm_type must be "DYNAMIC".
- If duty_cycle_pct has multiple values, evm_type must be "DYNAMIC".
- If evm_type is "DYNAMIC", duty_cycle_pct must exist.
- If evm_type is "DYNAMIC", dut_alternate_mode_name must exist.
- If no duty cycle exists, use STATIC unless datasheet says burst, pulsed, dynamic, packet, switching, or duty-cycle mode.

ACPR RULES:
- acpr_band_pairs must contain at least one pair for ACPR.
- Lower adjacent offset is usually negative.
- Upper adjacent offset is usually positive.
- If offset pair is defaulted, list it in default_assumptions and mark confidence low or medium.

CONFIDENCE RULES:
- high: datasheet explicitly provides most required values and the test is directly supported.
- medium: datasheet supports the test but some bench setup values are assumed.
- low: datasheet only partially supports the test and several demo assumptions were required.
- Do not use high confidence if frequency, voltage, or power had to be defaulted.

REJECTION RULES:
- If a datasheet claim is not actionable as a supported bench test, add it to rejected_candidates.
- Example: "Best-in-class efficiency" should be rejected because EFFICIENCY is not in the MVP test list.
- Example: "Plug-and-play" should be rejected.
- Example: "Compact form factor" should be rejected.
- Example: "High data rate modulation speeds" may support EVM only if modulation/bandwidth context exists; otherwise reject or mark low confidence.
- Example: "PSAT = 20 dBm" alone is not enough for EVM or ACPR unless other context is present.

OUTPUT RULES:
- Return exactly one JSON object.
- If no tests can be generated, return:
  {
    "status": "no_tests_found",
    "summary": "No supported RF validation tests could be generated from the provided text.",
    "generated_tests": [],
    "rejected_candidates": [...],
    "default_assumptions": []
  }
- generated_tests must be an array.
- rejected_candidates must be an array.
- default_assumptions must be an array.
- Each generated test must have a unique test_id.
- Use the supplied test_id_prefix when creating test IDs.
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
    return _dedupe_preserve_order(values)


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


def _normalize_modulation_bandwidth_pairing(test_case: RFTestCase) -> list[str]:
    """
    bandwidth_mhz is paired metadata with modulation, not an independent sweep axis.
    """
    assumptions: list[str] = []

    stim = test_case.stimulus

    if not _is_active(stim.modulation):
        return assumptions

    modulations = list(stim.modulation or [])
    inferred_bandwidths = [
        _infer_bandwidth_from_modulation_name(modulation)
        for modulation in modulations
    ]

    has_any_inferred = any(bw is not None for bw in inferred_bandwidths)
    has_all_inferred = all(bw is not None for bw in inferred_bandwidths)

    if has_all_inferred:
        new_bandwidths = [float(bw) for bw in inferred_bandwidths if bw is not None]

        if stim.bandwidth_mhz != new_bandwidths:
            stim.bandwidth_mhz = new_bandwidths
            assumptions.append(
                f"bandwidth_mhz paired to modulation as {new_bandwidths}; bandwidth_mhz is not an independent sweep axis"
            )

        return assumptions

    if has_any_inferred:
        assumptions.append(
            "Some modulation bandwidths were inferred, but not all. bandwidth_mhz was kept as provided and not used as a sweep axis."
        )

    return assumptions


def _normalize_additional_metrics(test_case: RFTestCase) -> None:
    """
    Clean measurement.additional_metrics if the model supports it.
    """
    if not hasattr(test_case.measurement, "additional_metrics"):
        return

    existing = getattr(test_case.measurement, "additional_metrics", []) or []
    cleaned = []

    for metric in existing:
        metric = str(metric).upper().strip()

        if metric in {"PIN", "POUT", "CURRENT"} and metric not in cleaned:
            if metric == test_case.measurement.metric:
                continue
            cleaned.append(metric)

    test_case.measurement.additional_metrics = cleaned


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

    # IMPORTANT:
    # bandwidth_mhz is intentionally NOT a sweep axis.
    # It is paired metadata for modulation.
    if axis_name == "bandwidth_mhz":
        return None

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
        if axis == "bandwidth_mhz":
            continue

        values = _axis_values(test_case, axis)

        if _is_sweep(values):
            axes.append(axis)

    return axes


def _clean_sweep_axis(axis: str) -> str | None:
    if axis in {"input_power_dbm", "servo_output_power_dbm"}:
        return "power"

    if axis == "bandwidth_mhz":
        return None

    if axis in SPEC_ALLOWED_SWEEP_AXES:
        return axis

    return None


def _apply_default_sweep_order(test_case: RFTestCase) -> None:
    multi_axes = _multi_value_axes(test_case)

    if not multi_axes:
        test_case.sweep = None
        return

    if test_case.sweep is None:
        test_case.sweep = RFSweep(
            sweep_type="full_factorial",
            expansion_order=[
                axis for axis in DEFAULT_SWEEP_EXPANSION_ORDER
                if axis in multi_axes and axis != "bandwidth_mhz"
            ],
        )
        return

    if not test_case.sweep.sweep_type:
        test_case.sweep.sweep_type = "full_factorial"

    existing_order = test_case.sweep.expansion_order or []

    cleaned_existing_order = []

    for axis in existing_order:
        cleaned_axis = _clean_sweep_axis(axis)
        if cleaned_axis and _is_active(_axis_values(test_case, cleaned_axis)):
            cleaned_existing_order.append(cleaned_axis)

    cleaned_existing_order = _dedupe_preserve_order(cleaned_existing_order) or []

    missing_axes = [
        axis
        for axis in DEFAULT_SWEEP_EXPANSION_ORDER
        if axis in multi_axes
        and axis != "bandwidth_mhz"
        and axis not in cleaned_existing_order
    ]

    test_case.sweep.expansion_order = cleaned_existing_order + missing_axes

    if not test_case.sweep.expansion_order:
        test_case.sweep = None


def _normalize_power_fields(test_case: RFTestCase) -> None:
    stim = test_case.stimulus

    stim.input_power_dbm = _normalize_list_field(stim.input_power_dbm)
    stim.servo_output_power_dbm = _normalize_list_field(stim.servo_output_power_dbm)

    # If servo-output power exists, it wins because input power is controlled by bench servo.
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

    # dut_alternate_mode_name is a single string, not a list.
    if hasattr(stim, "dut_alternate_mode_name"):
        alternate = getattr(stim, "dut_alternate_mode_name", None)

        if isinstance(alternate, list):
            if len(alternate) == 0:
                stim.dut_alternate_mode_name = None
            else:
                stim.dut_alternate_mode_name = str(alternate[0])

        elif alternate is not None:
            alternate = str(alternate).strip()
            stim.dut_alternate_mode_name = alternate if alternate else None

    bias.vcc_v = _normalize_list_field(bias.vcc_v)
    bias.vdd_v = _normalize_list_field(bias.vdd_v)

    env.temperature_c = _normalize_list_field(env.temperature_c)

    if not _is_active(env.temperature_c):
        env.temperature_c = [25.0]


def _normalize_evm_dynamic_switching(test_case: RFTestCase) -> list[str]:
    assumptions: list[str] = []

    if test_case.test_type != "EVM":
        return assumptions

    stim = test_case.stimulus

    if test_case.evm_settings is None:
        return assumptions

    if test_case.evm_settings.evm_type == "STATIC":
        if _is_active(stim.duty_cycle_pct):
            stim.duty_cycle_pct = None
            assumptions.append("duty_cycle_pct set to null because STATIC EVM does not use duty-cycle switching")

        if getattr(stim, "dut_alternate_mode_name", None):
            stim.dut_alternate_mode_name = None
            assumptions.append("dut_alternate_mode_name set to null because STATIC EVM does not use DUT mode switching")

        return assumptions

    if test_case.evm_settings.evm_type == "DYNAMIC":
        if not _is_active(stim.duty_cycle_pct):
            stim.duty_cycle_pct = [50.0]
            assumptions.append("duty_cycle_pct defaulted to [50%] for dynamic EVM")

        if not getattr(stim, "dut_alternate_mode_name", None):
            stim.dut_alternate_mode_name = "OFF"
            assumptions.append('dut_alternate_mode_name defaulted to "OFF" for dynamic EVM')

    return assumptions


def normalize_rf_spec_test_case(test_case: RFTestCase) -> tuple[RFTestCase, list[str]]:
    """
    Deterministic cleanup after datasheet LLM generation.
    """
    assumptions: list[str] = []

    _normalize_all_list_fields(test_case)
    _normalize_power_fields(test_case)
    _normalize_additional_metrics(test_case)

    if test_case.test_type == "EVM":
        if _is_active(test_case.stimulus.duty_cycle_pct):
            if test_case.evm_settings is not None:
                test_case.evm_settings.evm_type = "DYNAMIC"

    if test_case.test_type == "ACPR":
        if _is_active(test_case.stimulus.servo_output_power_dbm):
            test_case.stimulus.input_power_dbm = None

    assumptions.extend(_normalize_evm_dynamic_switching(test_case))
    assumptions.extend(_normalize_modulation_bandwidth_pairing(test_case))

    _apply_default_sweep_order(test_case)

    return test_case, assumptions


def _coerce_raw_spec_json(raw_json: dict, test_id_prefix: str) -> dict:
    """
    Clean LLM output before Pydantic validation.

    This is important because:
    - bandwidth_mhz is no longer a legal sweep axis.
    - input_power_dbm / servo_output_power_dbm must be converted to power.
    - SPEC upload MVP should only generate GAIN/EVM/ACPR/CURRENT.
    """
    if not isinstance(raw_json, dict):
        return {
            "status": "no_tests_found",
            "summary": "LLM did not return a valid JSON object.",
            "generated_tests": [],
            "rejected_candidates": [],
            "default_assumptions": [],
        }

    raw_json.setdefault("status", "no_tests_found")
    raw_json.setdefault("summary", "")
    raw_json.setdefault("generated_tests", [])
    raw_json.setdefault("rejected_candidates", [])
    raw_json.setdefault("default_assumptions", [])

    if not isinstance(raw_json["generated_tests"], list):
        raw_json["generated_tests"] = []

    cleaned_tests = []
    rejected = list(raw_json.get("rejected_candidates") or [])

    for index, test in enumerate(raw_json["generated_tests"], start=1):
        if not isinstance(test, dict):
            rejected.append({
                "candidate": f"generated_tests[{index}]",
                "reason": "Rejected because the generated test was not a JSON object.",
            })
            continue

        test_type = str(test.get("test_type", "")).upper().strip()

        if test_type not in SPEC_SUPPORTED_TEST_TYPES:
            rejected.append({
                "candidate": test.get("objective", f"generated_tests[{index}]"),
                "reason": f"Rejected because SPEC MVP only supports GAIN, EVM, ACPR, CURRENT, P1DB; got {test_type or 'unknown'}.",
            })
            continue

        test["test_type"] = test_type
        test["bench_type"] = "RF_BENCH"
        test.setdefault("schema_version", "inventide.rf.testcase.v1")
        test.setdefault("domain", "RF")
        test.setdefault("test_id", f"{test_id_prefix}_{index:02d}")
        test.setdefault("objective", f"Suggested {test_type} test from spec sheet")
        test.setdefault("stimulus", {})
        test.setdefault("dut_bias", {})
        test.setdefault("environment", {"temperature_c": [25.0]})
        test.setdefault("measurement", {"metric": test_type, "limit": None, "additional_metrics": []})
        test.setdefault("source_evidence", [])
        test.setdefault("confidence", "low")
        test.setdefault("execution_target", {
            "runner": "LabVIEW_RF_TestRunner",
            "config_format": "csv",
        })

        measurement = test.get("measurement")
        if isinstance(measurement, dict):
            measurement["metric"] = test_type
            measurement.setdefault("limit", None)
            measurement.setdefault("additional_metrics", [])

        stimulus = test.get("stimulus")
        if isinstance(stimulus, dict):
            if test_type != "EVM":
                stimulus["dut_alternate_mode_name"] = None

        sweep = test.get("sweep")
        if isinstance(sweep, dict):
            order = sweep.get("expansion_order") or []
            if not isinstance(order, list):
                order = []

            cleaned_order = []

            for axis in order:
                cleaned_axis = _clean_sweep_axis(str(axis))
                if cleaned_axis and cleaned_axis not in cleaned_order:
                    cleaned_order.append(cleaned_axis)

            sweep["expansion_order"] = cleaned_order
            sweep.setdefault("sweep_type", "full_factorial")

            if not sweep["expansion_order"]:
                test["sweep"] = None

        cleaned_tests.append(test)

    raw_json["generated_tests"] = cleaned_tests
    raw_json["rejected_candidates"] = rejected

    if cleaned_tests:
        raw_json["status"] = "ready"
    else:
        raw_json["status"] = "no_tests_found"

    return raw_json


def generate_rf_tests_from_spec_text(
    spec_text: str,
    source_names: List[str] | None = None,
) -> RFSpecGenerationResponse:
    """
    Generate suggested RF tests from extracted datasheet/spec-sheet text.

    spec_text:
        Combined extracted text from uploaded PDFs.

    source_names:
        Optional list of source PDF filenames for LLM context.
    """

    test_id_prefix = f"RF_SPEC_{str(uuid.uuid4())[:8].upper()}"

    sources_text = ""
    if source_names:
        sources_text = "\n".join([f"- {name}" for name in source_names])

    user_prompt = f"""
Source files:
{sources_text if sources_text else "- not provided"}

Use this test_id_prefix when generating test IDs:
{test_id_prefix}

Datasheet/specification text:
{spec_text}

Generate supported RF validation tests from this datasheet/spec text.

Remember:
- SPEC upload MVP supports only GAIN, EVM, ACPR, CURRENT.
- RF fields must be arrays or null, except dut_alternate_mode_name which is one string or null.
- [one value] means fixed condition.
- [multiple values] means swept condition.
- null means not used.
- sweep.expansion_order uses "power" for input_power_dbm or servo_output_power_dbm sweeps.
- Never use input_power_dbm or servo_output_power_dbm inside sweep.expansion_order.
- Never use bandwidth_mhz inside sweep.expansion_order.
- bandwidth_mhz is paired metadata with modulation.
- Only one power control method may be active in a test.
- For dynamic EVM, include duty_cycle_pct and dut_alternate_mode_name.
- For static EVM, dut_alternate_mode_name must be null and duty_cycle_pct must be null.
- Be conservative and include source_evidence.
"""

    raw_json = call_llm_for_json(
        system_prompt=RF_SPEC_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    raw_json = _coerce_raw_spec_json(raw_json, test_id_prefix)

    response = RFSpecGenerationResponse.model_validate(raw_json)

    normalized_tests = []

    for test_case in response.generated_tests:
        validated_test_case = RFTestCase.model_validate(test_case.model_dump())
        validated_test_case, deterministic_assumptions = normalize_rf_spec_test_case(validated_test_case)

        for assumption in deterministic_assumptions:
            if assumption not in response.default_assumptions:
                response.default_assumptions.append(assumption)

        normalized_tests.append(validated_test_case)

    response.generated_tests = normalized_tests

    if response.generated_tests and response.status != "ready":
        response.status = "ready"

    if not response.generated_tests and response.status != "no_tests_found":
        response.status = "no_tests_found"

    return response
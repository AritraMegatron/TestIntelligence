import csv
from itertools import product
from pathlib import Path
from typing import List

from app.models.rf_models import (
    DEFAULT_SWEEP_EXPANSION_ORDER,
    RFTestCase,
)
from app.settings import GENERATED_DIR


def _is_active(values) -> bool:
    return values is not None and len(values) > 0


def _is_sweep(values) -> bool:
    return values is not None and len(values) > 1


def _first(values):
    if not _is_active(values):
        return ""
    return values[0]


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


def _join_list(values, separator: str = "|") -> str:
    if values is None:
        return ""

    if not isinstance(values, list):
        return str(values)

    return separator.join(str(v) for v in values)


def _bandwidth_for_modulation(test_case: RFTestCase, modulation_value):
    """
    bandwidth_mhz is not an independent sweep axis.

    If modulation and bandwidth_mhz have the same length, pair by index:
        modulation[i] -> bandwidth_mhz[i]

    If bandwidth_mhz has one value, use it for all modulations.

    If no bandwidth is available, return blank.
    """
    stim = test_case.stimulus

    bandwidths = stim.bandwidth_mhz or []
    modulations = stim.modulation or []

    if not bandwidths:
        return ""

    if modulation_value is None or modulation_value == "":
        return bandwidths[0]

    if len(bandwidths) == 1:
        return bandwidths[0]

    if len(modulations) == len(bandwidths):
        try:
            index = modulations.index(modulation_value)
            return bandwidths[index]
        except ValueError:
            return bandwidths[0]

    # Fallback: do not expand bandwidth independently.
    # Use first bandwidth if pairing is ambiguous.
    return bandwidths[0]


def _sparameter_config_row(test_case: RFTestCase) -> dict:
    settings = test_case.s_parameter_settings
    bias = test_case.dut_bias
    env = test_case.environment
    limit = test_case.measurement.limit

    if settings is None:
        raise ValueError("S_PARAMETER test requires s_parameter_settings.")

    return {
        "test_id": test_case.test_id,
        "test_type": test_case.test_type,
        "bench_type": test_case.bench_type,
        "dut_mode_name": _first(test_case.stimulus.dut_mode_name),
        "dut_alternate_mode_name": getattr(test_case.stimulus, "dut_alternate_mode_name", "") or "",
        "vna_ports": _join_list(settings.ports),
        "s_parameters": _join_list(settings.measurements),
        "frequency_start_mhz": settings.frequency_start_mhz,
        "frequency_stop_mhz": settings.frequency_stop_mhz,
        "num_points": settings.num_points,
        "if_bandwidth_hz": settings.if_bandwidth_hz,
        "vna_power_dbm": settings.vna_power_dbm,
        "calibration_type": settings.calibration_type,
        "sweep_mode": settings.sweep_mode,
        "result_format": settings.result_format,
        "vcc_v": _first(bias.vcc_v),
        "vdd_v": _first(bias.vdd_v),
        "temperature_c": _first(env.temperature_c),
        "measurement_metric": test_case.measurement.metric,
        "additional_metrics": _join_list(
            getattr(test_case.measurement, "additional_metrics", []),
            separator="|",
        ),
        "limit_operator": limit.operator if limit else "",
        "limit_value": limit.value if limit else "",
        "limit_unit": limit.unit if limit else "",
        "sweep_enabled": "FALSE",
        "sweep_type": "",
        "sweep_order": "",
        "sweep_index": "",
        "sweep_total": "",
        "runner": "LabVIEW_SParameter_TestRunner",
    }


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
        raise ValueError(
            "bandwidth_mhz is not a supported sweep axis. "
            "Use modulation as the sweep axis and pair bandwidth_mhz by index."
        )

    if axis_name == "duty_cycle_pct":
        return stim.duty_cycle_pct

    if axis_name == "dut_mode_name":
        return stim.dut_mode_name

    if axis_name == "modulation":
        return stim.modulation

    raise ValueError(f"Unsupported sweep axis: {axis_name}")


def _default_expansion_order_for_test(test_case: RFTestCase) -> list:
    """
    Build default order using only active multi-value axes.

    Important:
        bandwidth_mhz is not included because it is paired to modulation,
        not independently expanded.
    """
    order = []

    for axis in DEFAULT_SWEEP_EXPANSION_ORDER:
        if axis == "bandwidth_mhz":
            continue

        values = _axis_values(test_case, axis)
        if _is_sweep(values):
            order.append(axis)

    return order


def _effective_expansion_order(test_case: RFTestCase) -> list:
    """
    Use user/LLM-provided expansion_order if available.
    Add missing multi-value axes using the default order.

    Important:
        bandwidth_mhz is filtered out even if the LLM/user accidentally puts it
        in sweep.expansion_order.
    """
    default_order = _default_expansion_order_for_test(test_case)

    if not test_case.sweep or not test_case.sweep.expansion_order:
        return default_order

    provided_order = []

    for axis in test_case.sweep.expansion_order:
        if axis == "bandwidth_mhz":
            continue

        if _is_active(_axis_values(test_case, axis)):
            provided_order.append(axis)

    missing_axes = [
        axis
        for axis in default_order
        if axis not in provided_order
    ]

    return provided_order + missing_axes


def _acpr_pairs_to_string(test_case: RFTestCase) -> str:
    pairs = test_case.stimulus.acpr_band_pairs

    if not pairs:
        return ""

    return ";".join(
        [
            f"{p.lower_offset_mhz},{p.upper_offset_mhz},{p.bandwidth_mhz if p.bandwidth_mhz is not None else ''}"
            for p in pairs
        ]
    )


def _base_config_row(test_case: RFTestCase) -> dict:
    limit = test_case.measurement.limit
    stim = test_case.stimulus
    bias = test_case.dut_bias
    env = test_case.environment

    power_field = _active_power_field(test_case)

    input_power = ""
    servo_power = ""

    if power_field == "input_power_dbm":
        input_power = _first(stim.input_power_dbm)

    if power_field == "servo_output_power_dbm":
        servo_power = _first(stim.servo_output_power_dbm)

    selected_modulation = _first(stim.modulation)

    return {
        "test_id": test_case.test_id,
        "test_type": test_case.test_type,
        "dut_mode_name": _first(stim.dut_mode_name),
        "dut_alternate_mode_name": getattr(stim, "dut_alternate_mode_name", "") or "",
        "frequency_mhz": _first(stim.frequency_mhz),
        "modulation": selected_modulation,
        "input_power_dbm": input_power,
        "servo_output_power_dbm": servo_power,

        # bandwidth_mhz follows selected modulation.
        # It is not independently expanded.
        "bandwidth_mhz": _bandwidth_for_modulation(test_case, selected_modulation),

        "duty_cycle_pct": _first(stim.duty_cycle_pct),
        "evm_type": test_case.evm_settings.evm_type if test_case.evm_settings else "",
        "acpr_band_pairs": _acpr_pairs_to_string(test_case),
        "vcc_v": _first(bias.vcc_v),
        "vdd_v": _first(bias.vdd_v),
        "temperature_c": _first(env.temperature_c),
        "measurement_metric": test_case.measurement.metric,
        "additional_metrics": _join_list(
            getattr(test_case.measurement, "additional_metrics", []),
            separator="|",
        ),
        "limit_operator": limit.operator if limit else "",
        "limit_value": limit.value if limit else "",
        "limit_unit": limit.unit if limit else "",
        "sweep_enabled": "FALSE",
        "sweep_type": "",
        "sweep_order": "",
        "sweep_index": "",
        "sweep_total": "",
        "averages": 10,
        "capture_time_ms": 20,
        "runner": test_case.execution_target["runner"],
    }


def _apply_axis_value_to_row(row: dict, test_case: RFTestCase, axis_name: str, value):
    """
    Maps abstract sweep axes to physical CSV columns.
    """
    if axis_name == "frequency_mhz":
        row["frequency_mhz"] = value

    elif axis_name == "power":
        power_field = _active_power_field(test_case)

        if power_field == "input_power_dbm":
            row["input_power_dbm"] = value
            row["servo_output_power_dbm"] = ""

        elif power_field == "servo_output_power_dbm":
            row["servo_output_power_dbm"] = value
            row["input_power_dbm"] = ""

        else:
            raise ValueError("Sweep axis 'power' requested, but no active power field exists.")

    elif axis_name == "vcc_v":
        row["vcc_v"] = value

    elif axis_name == "vdd_v":
        row["vdd_v"] = value

    elif axis_name == "temperature_c":
        row["temperature_c"] = value

    elif axis_name == "duty_cycle_pct":
        row["duty_cycle_pct"] = value

    elif axis_name == "dut_mode_name":
        row["dut_mode_name"] = value

    elif axis_name == "modulation":
        row["modulation"] = value

        # This is the key pairing behavior:
        # modulation[i] uses bandwidth_mhz[i] when both lists match.
        row["bandwidth_mhz"] = _bandwidth_for_modulation(test_case, value)

    elif axis_name == "bandwidth_mhz":
        raise ValueError(
            "bandwidth_mhz cannot be used as a sweep axis. "
            "It is paired to modulation."
        )

    else:
        raise ValueError(f"Unsupported sweep axis: {axis_name}")


def _expand_full_factorial(test_case: RFTestCase) -> List[dict]:
    base_row = _base_config_row(test_case)

    expansion_order = _effective_expansion_order(test_case)

    if not expansion_order:
        return [base_row]

    value_lists = []

    for axis in expansion_order:
        values = _axis_values(test_case, axis)

        if not _is_active(values):
            raise ValueError(f"Sweep axis {axis} has no active values.")

        value_lists.append(values)

    combinations = list(product(*value_lists))
    sweep_order_text = ">".join(expansion_order)

    rows = []

    for index, combination in enumerate(combinations, start=1):
        row = dict(base_row)
        row["test_id"] = f"{test_case.test_id}_P{index:03d}"
        row["sweep_enabled"] = "TRUE"
        row["sweep_type"] = "full_factorial"
        row["sweep_order"] = sweep_order_text
        row["sweep_index"] = index
        row["sweep_total"] = len(combinations)

        for axis, value in zip(expansion_order, combination):
            _apply_axis_value_to_row(row, test_case, axis, value)

        rows.append(row)

    return rows


def _expand_zipped(test_case: RFTestCase) -> List[dict]:
    base_row = _base_config_row(test_case)

    expansion_order = _effective_expansion_order(test_case)

    if not expansion_order:
        return [base_row]

    axis_value_map = {}

    for axis in expansion_order:
        values = _axis_values(test_case, axis)

        if not _is_active(values):
            raise ValueError(f"Sweep axis {axis} has no active values.")

        axis_value_map[axis] = values

    total = max(len(values) for values in axis_value_map.values())
    sweep_order_text = ">".join(expansion_order)

    rows = []

    for row_index in range(total):
        row = dict(base_row)
        row["test_id"] = f"{test_case.test_id}_P{row_index + 1:03d}"
        row["sweep_enabled"] = "TRUE"
        row["sweep_type"] = "zipped"
        row["sweep_order"] = sweep_order_text
        row["sweep_index"] = row_index + 1
        row["sweep_total"] = total

        for axis, values in axis_value_map.items():
            if len(values) == 1:
                value = values[0]
            else:
                value = values[row_index]

            _apply_axis_value_to_row(row, test_case, axis, value)

        rows.append(row)

    return rows


def _test_case_to_config_rows(test_case: RFTestCase) -> List[dict]:
    """
    Converts one RFTestCase into one or more bench config rows.

    No sweep:
        one JSON test -> one CSV row

    Sweep:
        list-based RF JSON -> expanded executable CSV rows

    Important:
        "power" in expansion_order maps to either:
            - input_power_dbm
            - servo_output_power_dbm

        "bandwidth_mhz" is not a sweep axis.
        It is paired metadata with modulation.
    """

    if test_case.test_type == "S_PARAMETER":
        return [_sparameter_config_row(test_case)]

    expansion_order = _effective_expansion_order(test_case)

    if not expansion_order:
        return [_base_config_row(test_case)]

    sweep_type = "full_factorial"

    if test_case.sweep:
        sweep_type = test_case.sweep.sweep_type

    if sweep_type == "zipped":
        return _expand_zipped(test_case)

    return _expand_full_factorial(test_case)


def generate_labview_config_csv(test_case: RFTestCase) -> Path:
    """
    Backward-compatible single-test config generator.
    """
    return generate_labview_config_csv_multi(
        [test_case],
        f"{test_case.test_id}_labview_config.csv",
    )


def generate_labview_config_csv_multi(
    test_cases: List[RFTestCase],
    file_name: str = "rf_bench_config.csv",
) -> Path:
    """
    Generates a CSV for one or many RF tests.

    JSON behavior:
        RF fields are arrays:
            [one value] = fixed condition
            [multiple values] = swept condition

    CSV behavior:
        Full factorial or zipped expansion produces one executable row per bench point.
    """
    if not test_cases:
        raise ValueError("No test cases provided for config generation.")

    file_path = GENERATED_DIR / file_name

    rows = []

    for test_case in test_cases:
        rows.extend(_test_case_to_config_rows(test_case))

    if not rows:
        raise ValueError("No CSV rows generated.")

    # Use union of all columns because RF bench rows and S-parameter rows
    # do not have identical fields.
    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with file_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return file_path
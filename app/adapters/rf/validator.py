from app.models.rf_models import (
    DEFAULT_SWEEP_EXPANSION_ORDER,
    RFTestCase,
)


DEMO_MIN_FREQ_MHZ = 100
DEMO_MAX_FREQ_MHZ = 7000
DEMO_MAX_VOLTAGE = 5.0
DEMO_MAX_POWER_DBM = 35.0


def _is_active(values) -> bool:
    return values is not None and len(values) > 0


def _is_sweep(values) -> bool:
    return values is not None and len(values) > 1


def _first(values):
    if not _is_active(values):
        return None
    return values[0]


def _active_power_field(stim):
    """
    Returns:
        "input_power_dbm"
        "servo_output_power_dbm"
        None
    """
    has_input = _is_active(stim.input_power_dbm)
    has_servo = _is_active(stim.servo_output_power_dbm)

    if has_input:
        return "input_power_dbm"

    if has_servo:
        return "servo_output_power_dbm"

    return None


def _active_power_values(stim):
    power_field = _active_power_field(stim)

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
        return _active_power_values(stim)

    if axis_name == "vcc_v":
        return bias.vcc_v

    if axis_name == "vdd_v":
        return bias.vdd_v

    if axis_name == "temperature_c":
        return env.temperature_c

    if axis_name == "bandwidth_mhz":
        return stim.bandwidth_mhz

    if axis_name == "duty_cycle_pct":
        return stim.duty_cycle_pct

    if axis_name == "dut_mode_name":
        return stim.dut_mode_name

    if axis_name == "modulation":
        return stim.modulation

    return None


def _multi_value_axes(test_case: RFTestCase) -> list:
    """
    Returns axes that have more than one active value and therefore require
    expansion_order.
    """
    axes = []

    for axis in DEFAULT_SWEEP_EXPANSION_ORDER:
        values = _axis_values(test_case, axis)
        if _is_sweep(values):
            axes.append(axis)

    return axes


def _validate_numeric_list(
    field_name: str,
    values,
    errors: list,
    min_value=None,
    max_value=None,
):
    if not _is_active(values):
        return

    for value in values:
        if not isinstance(value, (int, float)):
            errors.append(f"{field_name} contains non-numeric value: {value}")
            continue

        if min_value is not None and value < min_value:
            errors.append(f"{field_name} contains {value}, below minimum {min_value}.")

        if max_value is not None and value > max_value:
            errors.append(f"{field_name} contains {value}, above maximum {max_value}.")


def _validate_string_list(field_name: str, values, errors: list):
    if not _is_active(values):
        return

    for value in values:
        if not isinstance(value, str):
            errors.append(f"{field_name} contains non-string value: {value}")
            continue

        if not value.strip():
            errors.append(f"{field_name} contains an empty string.")


def _validate_duplicate_values(field_name: str, values, warnings: list):
    if not _is_active(values):
        return

    try:
        if len(values) != len(set(values)):
            warnings.append(f"{field_name} contains duplicate values.")
    except TypeError:
        # Some complex values may not be hashable. Ignore duplicate check.
        pass


def _validate_required_list(field_name: str, values, errors: list):
    if not _is_active(values):
        errors.append(f"{field_name} is required and must be a non-empty list.")
def _is_valid_sparameter_name(name: str) -> bool:
    """
    Valid examples:
    S11, S21, S12, S22, S31, S43, etc.
    """
    if not isinstance(name, str):
        return False

    name = name.upper().strip()

    if len(name) < 3:
        return False

    if not name.startswith("S"):
        return False

    row_col = name[1:]

    if not row_col.isdigit():
        return False

    return True


def _sparameter_ports_from_name(name: str) -> tuple[int, int] | None:
    """
    S21 means response at port 2 due to stimulus at port 1.
    Returns (2, 1).
    """
    if not _is_valid_sparameter_name(name):
        return None

    name = name.upper().strip()
    digits = name[1:]

    if len(digits) != 2:
        return None

    return int(digits[0]), int(digits[1])

def _validate_sweep_order(test_case: RFTestCase, errors: list, warnings: list):
    multi_axes = _multi_value_axes(test_case)

    # No sweep needed.
    if not multi_axes:
        if test_case.sweep and test_case.sweep.expansion_order:
            warnings.append(
                "sweep.expansion_order is provided, but all active fields have only one value."
            )
        return

    # Sweep needed.
    if test_case.sweep is None:
        errors.append(
            f"Sweep fields detected {multi_axes}, but sweep object is missing."
        )
        return

    order = test_case.sweep.expansion_order or []

    if not order:
        errors.append(
            f"Sweep fields detected {multi_axes}, but sweep.expansion_order is empty."
        )
        return

    # User rule: expansion_order should use "power", not physical power field names.
    invalid_power_axes = [
        axis
        for axis in order
        if axis in ["input_power_dbm", "servo_output_power_dbm"]
    ]

    if invalid_power_axes:
        errors.append(
            "sweep.expansion_order must use 'power', not input_power_dbm or servo_output_power_dbm."
        )

    for axis in multi_axes:
        if axis not in order:
            errors.append(
                f"{axis} has multiple values, so sweep.expansion_order must include '{axis}'."
            )

    for axis in order:
        values = _axis_values(test_case, axis)

        if not _is_active(values):
            errors.append(
                f"sweep.expansion_order includes '{axis}', but that axis has no active values."
            )
        elif len(values) == 1:
            warnings.append(
                f"sweep.expansion_order includes '{axis}', but it has only one value."
            )

    if test_case.sweep.sweep_type == "zipped":
        zipped_lengths = []

        for axis in order:
            values = _axis_values(test_case, axis)
            if _is_sweep(values):
                zipped_lengths.append(len(values))

        if zipped_lengths and len(set(zipped_lengths)) != 1:
            errors.append(
                "zipped sweep requires all swept axes to have the same number of values."
            )

def _validate_sparameter_test(test_case: RFTestCase, errors: list, warnings: list):
    settings = test_case.s_parameter_settings
    bias = test_case.dut_bias
    env = test_case.environment

    if test_case.bench_type != "S_PARAMETER_BENCH":
        errors.append("S_PARAMETER test requires bench_type = S_PARAMETER_BENCH.")

    if settings is None:
        errors.append("S_PARAMETER test requires s_parameter_settings.")
        return

    if not settings.ports:
        errors.append("S_PARAMETER test requires at least one VNA port.")

    if len(settings.ports) < 2:
        warnings.append(
            "S_PARAMETER test has fewer than two ports. Most RF S-parameter tests are 2-port or higher."
        )

    for port in settings.ports:
        if not isinstance(port, int):
            errors.append(f"S-parameter port must be an integer. Got {port}.")
        elif port <= 0:
            errors.append(f"S-parameter port must be positive. Got {port}.")

    if not settings.measurements:
        errors.append("S_PARAMETER test requires at least one S-parameter measurement.")

    normalized_measurements = []

    for measurement in settings.measurements:
        if not isinstance(measurement, str):
            errors.append(f"S-parameter measurement must be a string. Got {measurement}.")
            continue

        measurement_name = measurement.upper().strip()
        normalized_measurements.append(measurement_name)

        ports = _sparameter_ports_from_name(measurement_name)

        if ports is None:
            errors.append(
                f"Invalid S-parameter measurement '{measurement}'. Expected format like S11, S21, S12, S22."
            )
            continue

        output_port, input_port = ports

        if output_port not in settings.ports:
            errors.append(
                f"{measurement_name} uses output port {output_port}, but port {output_port} is not listed in ports."
            )

        if input_port not in settings.ports:
            errors.append(
                f"{measurement_name} uses input port {input_port}, but port {input_port} is not listed in ports."
            )

    settings.measurements = normalized_measurements

    if settings.frequency_start_mhz <= 0:
        errors.append("frequency_start_mhz must be greater than 0.")

    if settings.frequency_stop_mhz <= 0:
        errors.append("frequency_stop_mhz must be greater than 0.")

    if settings.frequency_stop_mhz <= settings.frequency_start_mhz:
        errors.append("frequency_stop_mhz must be greater than frequency_start_mhz.")

    if settings.num_points < 2:
        errors.append("num_points must be at least 2 for an S-parameter sweep.")

    if settings.if_bandwidth_hz <= 0:
        errors.append("if_bandwidth_hz must be greater than 0.")

    if settings.vna_power_dbm is not None and settings.vna_power_dbm > 10:
        warnings.append(
            f"VNA power {settings.vna_power_dbm} dBm is relatively high for S-parameter measurement."
        )

    if not _is_active(bias.vcc_v):
        warnings.append("S_PARAMETER test has no VCC bias. This is okay only for passive DUTs.")

    if not _is_active(bias.vdd_v):
        warnings.append("S_PARAMETER test has no VDD bias. This is okay only for passive DUTs.")

    if not _is_active(env.temperature_c):
        warnings.append("S_PARAMETER test has no temperature setting. Default lab ambient may be used.")

    # For S-parameter tests, normal RF stimulus power fields should not be used.
    if _is_active(test_case.stimulus.input_power_dbm):
        warnings.append(
            "S_PARAMETER test uses VNA power from s_parameter_settings.vna_power_dbm. "
            "stimulus.input_power_dbm will usually be ignored."
        )

    if _is_active(test_case.stimulus.servo_output_power_dbm):
        errors.append("S_PARAMETER test should not use servo_output_power_dbm.")

    if test_case.sweep and test_case.sweep.expansion_order:
        warnings.append(
            "S_PARAMETER frequency sweep is handled inside s_parameter_settings. "
            "Use sweep.expansion_order only for external sweeps such as VCC or temperature."
        )

def validate_rf_test_case(test_case: RFTestCase) -> dict:
    errors = []
    warnings = []

    stim = test_case.stimulus
    bias = test_case.dut_bias
    env = test_case.environment

    # ------------------------------------------------------------
    # S param checks
    # ------------------------------------------------------------

    if test_case.test_type == "S_PARAMETER":
        _validate_sparameter_test(test_case, errors, warnings)

        if test_case.measurement.metric != test_case.test_type:
            errors.append(
                f"measurement.metric must match test_type. "
                f"Got measurement.metric={test_case.measurement.metric}, "
                f"test_type={test_case.test_type}."
            )

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # ------------------------------------------------------------
    # Basic required field checks
    # ------------------------------------------------------------
    _validate_required_list("frequency_mhz", stim.frequency_mhz, errors)
    _validate_required_list("dut_mode_name", stim.dut_mode_name, errors)
    _validate_required_list("modulation", stim.modulation, errors)
    _validate_required_list("vcc_v", bias.vcc_v, errors)
    _validate_required_list("vdd_v", bias.vdd_v, errors)
    _validate_required_list("temperature_c", env.temperature_c, errors)

    # ------------------------------------------------------------
    # Type and range checks
    # ------------------------------------------------------------
    _validate_numeric_list(
        "frequency_mhz",
        stim.frequency_mhz,
        errors,
        min_value=DEMO_MIN_FREQ_MHZ,
        max_value=DEMO_MAX_FREQ_MHZ,
    )

    _validate_numeric_list(
        "input_power_dbm",
        stim.input_power_dbm,
        errors,
        max_value=DEMO_MAX_POWER_DBM,
    )

    _validate_numeric_list(
        "servo_output_power_dbm",
        stim.servo_output_power_dbm,
        errors,
        max_value=DEMO_MAX_POWER_DBM,
    )

    _validate_numeric_list(
        "bandwidth_mhz",
        stim.bandwidth_mhz,
        errors,
        min_value=0,
    )

    _validate_numeric_list(
        "duty_cycle_pct",
        stim.duty_cycle_pct,
        errors,
        min_value=0,
        max_value=100,
    )

    _validate_numeric_list(
        "vcc_v",
        bias.vcc_v,
        errors,
        min_value=0,
        max_value=DEMO_MAX_VOLTAGE,
    )

    _validate_numeric_list(
        "vdd_v",
        bias.vdd_v,
        errors,
        min_value=0,
        max_value=DEMO_MAX_VOLTAGE,
    )

    _validate_numeric_list(
        "temperature_c",
        env.temperature_c,
        errors,
    )

    _validate_string_list("dut_mode_name", stim.dut_mode_name, errors)
    _validate_string_list("modulation", stim.modulation, errors)

    # ------------------------------------------------------------
    # Duplicate warnings
    # ------------------------------------------------------------
    _validate_duplicate_values("frequency_mhz", stim.frequency_mhz, warnings)
    _validate_duplicate_values("input_power_dbm", stim.input_power_dbm, warnings)
    _validate_duplicate_values("servo_output_power_dbm", stim.servo_output_power_dbm, warnings)
    _validate_duplicate_values("bandwidth_mhz", stim.bandwidth_mhz, warnings)
    _validate_duplicate_values("duty_cycle_pct", stim.duty_cycle_pct, warnings)
    _validate_duplicate_values("vcc_v", bias.vcc_v, warnings)
    _validate_duplicate_values("vdd_v", bias.vdd_v, warnings)
    _validate_duplicate_values("temperature_c", env.temperature_c, warnings)
    _validate_duplicate_values("dut_mode_name", stim.dut_mode_name, warnings)
    _validate_duplicate_values("modulation", stim.modulation, warnings)

    # ------------------------------------------------------------
    # Power control rules
    # ------------------------------------------------------------
    has_input_power = _is_active(stim.input_power_dbm)
    has_servo_power = _is_active(stim.servo_output_power_dbm)

    if has_input_power and has_servo_power:
        errors.append(
            "Only one power control method is allowed per test: "
            "input_power_dbm or servo_output_power_dbm."
        )

    if _is_sweep(stim.input_power_dbm) or _is_sweep(stim.servo_output_power_dbm):
        if test_case.sweep is None or "power" not in test_case.sweep.expansion_order:
            errors.append(
                "Power has multiple values, so sweep.expansion_order must include 'power'."
            )

    # ------------------------------------------------------------
    # Sweep order validation
    # ------------------------------------------------------------
    _validate_sweep_order(test_case, errors, warnings)

    # ------------------------------------------------------------
    # Measurement consistency
    # ------------------------------------------------------------
    if test_case.measurement.metric != test_case.test_type:
        errors.append(
            f"measurement.metric must match test_type. "
            f"Got measurement.metric={test_case.measurement.metric}, "
            f"test_type={test_case.test_type}."
        )

    if test_case.measurement.limit is None:
        warnings.append(
            "No pass/fail limit was provided. Test can run, but automatic pass/fail may be unavailable."
        )

    # ------------------------------------------------------------
    # Test-specific validation
    # ------------------------------------------------------------
    if test_case.test_type == "GAIN":
        if not has_input_power:
            errors.append("GAIN test requires input_power_dbm.")

        if has_servo_power:
            warnings.append(
                "GAIN tests are usually driven by input power. "
                "servo_output_power_dbm may not be used by the runner."
            )

    elif test_case.test_type == "EVM":
        if test_case.evm_settings is None:
            errors.append("EVM test requires evm_settings.evm_type.")

        if not _is_active(stim.bandwidth_mhz):
            errors.append("EVM test requires bandwidth_mhz.")

        if not has_input_power and not has_servo_power:
            errors.append("EVM test requires input_power_dbm or servo_output_power_dbm.")

        has_duty_cycle = _is_active(stim.duty_cycle_pct)

        if has_duty_cycle:
            if test_case.evm_settings and test_case.evm_settings.evm_type != "DYNAMIC":
                errors.append(
                    "If duty_cycle_pct is provided, EVM evm_type must be DYNAMIC."
                )

        if test_case.evm_settings and test_case.evm_settings.evm_type == "DYNAMIC":
            if not has_duty_cycle:
                errors.append("Dynamic EVM test requires duty_cycle_pct.")

    elif test_case.test_type == "ACPR":
        if not has_servo_power:
            errors.append("ACPR test requires servo_output_power_dbm.")

        if has_input_power:
            warnings.append(
                "ACPR is typically servoed to output power. "
                "input_power_dbm may be ignored by the runner."
            )

        if not stim.acpr_band_pairs:
            errors.append("ACPR test requires at least one lower/upper ACPR band pair.")

        if stim.acpr_band_pairs:
            for index, pair in enumerate(stim.acpr_band_pairs, start=1):
                if pair.lower_offset_mhz == pair.upper_offset_mhz:
                    errors.append(
                        f"ACPR band pair {index} has identical lower and upper offsets."
                    )

                if pair.lower_offset_mhz > 0:
                    warnings.append(
                        f"ACPR band pair {index} lower_offset_mhz is positive. "
                        "Lower adjacent offset is usually negative."
                    )

                if pair.upper_offset_mhz < 0:
                    warnings.append(
                        f"ACPR band pair {index} upper_offset_mhz is negative. "
                        "Upper adjacent offset is usually positive."
                    )

                if pair.bandwidth_mhz is not None and pair.bandwidth_mhz <= 0:
                    errors.append(
                        f"ACPR band pair {index} bandwidth_mhz must be greater than 0."
                    )

    elif test_case.test_type == "CURRENT":
        if not has_input_power:
            errors.append("CURRENT test requires input_power_dbm.")

        if has_servo_power:
            warnings.append(
                "CURRENT tests are usually driven by input power. "
                "servo_output_power_dbm may not be used by the runner."
            )

    elif test_case.test_type == "P1DB":
        # P1dB uses power sweep parameters, not input_power_dbm or servo_output_power_dbm
        if has_input_power or has_servo_power:
            warnings.append(
                "P1dB tests use power sweep via pin_start_dbm, pin_stop_dbm, pin_step_dbm. "
                "input_power_dbm and servo_output_power_dbm may be ignored."
            )

        # Check for required P1dB measurement settings
        if test_case.measurement.pin_start_dbm is None:
            errors.append("P1dB test requires measurement.pin_start_dbm.")

        if test_case.measurement.pin_stop_dbm is None:
            errors.append("P1dB test requires measurement.pin_stop_dbm.")

        if test_case.measurement.pin_step_dbm is None:
            errors.append("P1dB test requires measurement.pin_step_dbm.")

        # Validate power sweep range
        if test_case.measurement.pin_start_dbm is not None and test_case.measurement.pin_stop_dbm is not None:
            if test_case.measurement.pin_start_dbm >= test_case.measurement.pin_stop_dbm:
                errors.append("P1dB pin_start_dbm must be less than pin_stop_dbm.")

        if test_case.measurement.pin_step_dbm is not None and test_case.measurement.pin_step_dbm <= 0:
            errors.append("P1dB pin_step_dbm must be greater than 0.")

    else:
        errors.append(f"Unsupported RF test_type: {test_case.test_type}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
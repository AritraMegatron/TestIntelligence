from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


RFTestType = Literal["GAIN", "EVM", "ACPR", "CURRENT", "S_PARAMETER"]
RFAdditionalMetric = Literal["PIN", "POUT", "CURRENT"]
BenchType = Literal["RF_BENCH", "S_PARAMETER_BENCH"]
SweepType = Literal["full_factorial", "zipped"]

SweepOrderAxis = Literal[
    "frequency_mhz",
    "power",
    "vcc_v",
    "vdd_v",
    "temperature_c",
    "duty_cycle_pct",
    "dut_mode_name",
    "modulation",
]


DEFAULT_SWEEP_EXPANSION_ORDER: List[SweepOrderAxis] = [
    "frequency_mhz",
    "vcc_v",
    "vdd_v",
    "temperature_c",
    "power",
    "duty_cycle_pct",
    "dut_mode_name",
    "modulation",
]


def _as_optional_list(value: Any):
    """
    Allows the LLM to accidentally return a scalar while the schema still
    normalizes it into list format.

    Examples:
        2450 -> [2450]
        "802.11ax" -> ["802.11ax"]
        [] -> None
        None -> None
    """
    if value is None:
        return None

    if isinstance(value, list):
        return value if len(value) > 0 else None

    return [value]


class ACPRBandPair(BaseModel):
    lower_offset_mhz: float
    upper_offset_mhz: float
    bandwidth_mhz: Optional[float] = None


class RFSweep(BaseModel):
    """
    Sweep execution metadata.

    Important:
    - This no longer stores values.
    - Values live in stimulus / dut_bias / environment as lists.
    - expansion_order only controls outer-to-inner execution order.
    """

    sweep_type: SweepType = "full_factorial"
    expansion_order: List[SweepOrderAxis] = Field(default_factory=list)


class RFStimulus(BaseModel):
    """
    RF stimulus variables.

    Convention:
        None        = not used / not applicable
        [x]         = fixed condition
        [x, y, z]   = swept condition

        Dynamic EVM convention:
        dut_mode_name = active DUT mode/state. This can be one or more modes.
        dut_alternate_mode_name = inactive/alternate DUT mode/state. This is a single value.
        duty_cycle_pct = active-state duty cycle percentage.
    """

    frequency_mhz: Optional[List[float]] = None
    dut_mode_name: Optional[List[str]] = None
    dut_alternate_mode_name: Optional[str] = None
    modulation: Optional[List[str]] = None

    # Only one of these may be non-null in a test.
    # expansion_order uses the abstract axis "power".
    input_power_dbm: Optional[List[float]] = None
    servo_output_power_dbm: Optional[List[float]] = None

    bandwidth_mhz: Optional[List[float]] = None
    duty_cycle_pct: Optional[List[float]] = None

    acpr_band_pairs: Optional[List[ACPRBandPair]] = None

    @field_validator(
        "frequency_mhz",
        "input_power_dbm",
        "servo_output_power_dbm",
        "bandwidth_mhz",
        "duty_cycle_pct",
        mode="before",
    )
    @classmethod
    def normalize_numeric_list(cls, value):
        return _as_optional_list(value)

    @field_validator(
        "dut_mode_name",
        "modulation",
        mode="before",
    )
    @classmethod
    def normalize_string_list(cls, value):
        return _as_optional_list(value)

    @field_validator("dut_alternate_mode_name", mode="before")
    @classmethod
    def normalize_alternate_mode(cls, value):
        """
        dut_alternate_mode_name must be one single mode/state.

        Accepts:
            "OFF" -> "OFF"
            ["OFF"] -> "OFF"

        Rejects:
            ["OFF", "SLEEP"] because dynamic EVM should use only one alternate mode.
        """
        if value is None:
            return None

        if isinstance(value, list):
            if len(value) == 0:
                return None
            if len(value) == 1:
                return str(value[0])
            raise ValueError("dut_alternate_mode_name must be a single string, not a list of multiple modes.")

        value = str(value).strip()
        return value if value else None


class RFBias(BaseModel):
    """
    Bias variables.

    Convention:
        None        = not used / not applicable
        [x]         = fixed condition
        [x, y, z]   = swept condition
    """

    vcc_v: Optional[List[float]] = None
    vdd_v: Optional[List[float]] = None

    @field_validator("vcc_v", "vdd_v", mode="before")
    @classmethod
    def normalize_numeric_list(cls, value):
        return _as_optional_list(value)


class RFEnvironment(BaseModel):
    """
    Environmental variables.

    Temperature defaults to [25.0], meaning fixed 25 C.
    """

    temperature_c: Optional[List[float]] = Field(default_factory=lambda: [25.0])

    @field_validator("temperature_c", mode="before")
    @classmethod
    def normalize_numeric_list(cls, value):
        return _as_optional_list(value)


class RFLimit(BaseModel):
    operator: Literal["<=", ">=", "==", "<", ">"]
    value: float
    unit: str


class RFMeasurement(BaseModel):
    metric: RFTestType
    limit: Optional[RFLimit] = None

    # Optional extra values to log at every bench point.
    # Example:
    #   metric = "EVM"
    #   additional_metrics = ["PIN", "POUT"]
    #
    # This means EVM is the primary measurement, while PIN/POUT are also captured.
    additional_metrics: List[RFAdditionalMetric] = Field(default_factory=list)


class RFEVMSettings(BaseModel):
    evm_type: Literal["STATIC", "DYNAMIC"]

class RFSParameterSettings(BaseModel):
    """
    VNA / S-parameter bench settings.

    Important:
    Frequency sweep here is handled internally by the VNA.
    This is different from RF bench frequency expansion.
    """

    ports: List[int]
    measurements: List[str]

    frequency_start_mhz: float
    frequency_stop_mhz: float

    num_points: int = 401
    if_bandwidth_hz: float = 1000
    vna_power_dbm: Optional[float] = -10

    calibration_type: Optional[Literal["SOLT", "TRL", "ECAL", "UNKNOWN"]] = "SOLT"
    sweep_mode: Optional[Literal["linear", "log"]] = "linear"
    result_format: Optional[Literal["DB_ANGLE", "MAG_ANGLE", "REAL_IMAG"]] = "DB_ANGLE"

class RFTestCase(BaseModel):
    schema_version: str = "inventide.rf.testcase.v1"
    domain: Literal["RF"] = "RF"

    test_id: str
    test_type: RFTestType
    bench_type: BenchType = "RF_BENCH"
    objective: str

    stimulus: RFStimulus
    dut_bias: RFBias
    environment: RFEnvironment = Field(default_factory=RFEnvironment)

    measurement: RFMeasurement
    evm_settings: Optional[RFEVMSettings] = None
    s_parameter_settings: Optional[RFSParameterSettings] = None

    # sweep only contains execution-order metadata.
    sweep: Optional[RFSweep] = None

    source_evidence: Optional[List[str]] = None
    confidence: Optional[Literal["low", "medium", "high"]] = None

    execution_target: dict = Field(
        default_factory=lambda: {
            "runner": "Generic_RF_TestRunner",
            "config_format": "csv",
        }
    )


class RFGenerationResponse(BaseModel):
    status: Literal["ready", "needs_clarification"]
    clarifying_questions: List[str] = Field(default_factory=list)
    default_assumptions: List[str] = Field(default_factory=list)
    test_case: Optional[RFTestCase] = None


class RFSpecGenerationResponse(BaseModel):
    status: Literal["ready", "no_tests_found"]
    summary: str
    generated_tests: List[RFTestCase] = Field(default_factory=list)
    rejected_candidates: List[dict] = Field(default_factory=list)
    default_assumptions: List[str] = Field(default_factory=list)
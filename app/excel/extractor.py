"""
Structured Excel workbook extraction for Inventide RF TestBridge MVP.

MVP contract:
- Only .xlsx workbooks are supported.
- Each visible sheet may contain an RF test table.
- Header row must include at least: test_name, test_type.
- For this MVP, supported workbook rows are EVM rows.
- Each non-empty row becomes one RFTestCase.

This is intentionally deterministic. The Excel workbook format is the contract,
so we do not need the LLM for this path.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from app.models.rf_models import RFSpecGenerationResponse, RFTestCase


SUPPORTED_EXCEL_TEST_TYPES = {"EVM", "ACPR"}
REQUIRED_COLUMNS = {"test_name", "test_type"}


def _normalize_header(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _cell_text(value: Any) -> str | None:
    if _is_blank(value):
        return None
    return str(value).strip()


def _number(value: Any) -> float | None:
    if _is_blank(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    text = text.replace("dBm", "").replace("dbm", "")
    text = text.replace("MHz", "").replace("mhz", "")
    text = text.replace("V", "").replace("v", "")
    text = text.replace("%", "")
    text = text.strip()

    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def _number_list(value: Any) -> list[float] | None:
    """
    Accepts either a single value or comma/semicolon/pipe-separated values.

    Examples:
        2450 -> [2450.0]
        "2400,2450,2500" -> [2400.0, 2450.0, 2500.0]
    """
    if _is_blank(value):
        return None

    if isinstance(value, (int, float)):
        return [float(value)]

    text = str(value).strip()
    parts = [p.strip() for p in re.split(r"[,;|]", text) if p.strip()]
    values: list[float] = []

    for part in parts:
        parsed = _number(part)
        if parsed is not None:
            values.append(parsed)

    return values or None


def _string_list(value: Any) -> list[str] | None:
    if _is_blank(value):
        return None

    text = str(value).strip()
    parts = [p.strip() for p in re.split(r"[,;|]", text) if p.strip()]
    return parts or None


def _additional_metrics(value: Any) -> list[str]:
    allowed = {"PIN", "POUT", "CURRENT"}
    metrics: list[str] = []

    for item in _string_list(value) or []:
        metric = item.upper().strip()
        if metric in allowed and metric not in metrics:
            metrics.append(metric)

    return metrics


def _find_header_row(sheet) -> tuple[int | None, dict[str, int]]:
    """
    Finds the first row that contains the MVP required column names.
    Returns 1-based row number and a header -> 1-based column index map.
    """
    for row_index, row in enumerate(sheet.iter_rows(values_only=True), start=1):
        headers = [_normalize_header(value) for value in row]
        header_set = set(h for h in headers if h)

        if REQUIRED_COLUMNS.issubset(header_set):
            mapping = {
                header: col_index
                for col_index, header in enumerate(headers, start=1)
                if header
            }
            return row_index, mapping

    return None, {}


def _row_dict(sheet, row_index: int, header_map: dict[str, int]) -> dict[str, Any]:
    return {
        header: sheet.cell(row_index, col_index).value
        for header, col_index in header_map.items()
    }


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", value or "")
    text = re.sub(r"_+", "_", text).strip("_").upper()
    return text[:48] or "TEST"


def _limit_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    value = _number(row.get("evm_limit_db") or row.get("limit_value") or row.get("acpr_limit_dbc"))
    if value is None:
        return None

    return {
        "operator": _cell_text(row.get("limit_operator")) or "<=",
        "value": value,
        "unit": _cell_text(row.get("limit_unit")) or "dB",
    }


def _modulation_from_row(row: dict[str, Any]) -> str | None:
    """
    For this EVM MVP workbook, burst_length carries the exact waveform/burst name,
    e.g. AX20-MCS11-300US. Prefer it over the base modulation column because the
    bench needs to know which burst waveform to load.
    """
    burst_length = _cell_text(row.get("burst_length"))
    if burst_length:
        return burst_length

    return _cell_text(row.get("modulation"))


def _acpr_band_pairs_from_row(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """
    Extracts ACPR band pairs from Excel row.
    
    Expected columns (based on ACPR_Test_Plan.xlsx format):
    - acpr_offset_mhz: primary offset frequency
    - acpr_adjacent_bw_mhz: bandwidth for adjacent channel
    - acpr_alternate_offset_mhz: alternate offset frequency (optional)
    
    Creates band pairs for ACPR measurement. If alternate offset is provided,
    creates two band pairs (primary and alternate).
    """
    primary_offset = _number(row.get("acpr_offset_mhz"))
    bandwidth = _number(row.get("acpr_adjacent_bw_mhz"))
    alternate_offset = _number(row.get("acpr_alternate_offset_mhz"))
    
    if not primary_offset:
        return None
    
    band_pairs = []
    
    # Primary band pair (lower and upper at same offset distance)
    band_pairs.append({
        "lower_offset_mhz": primary_offset,
        "upper_offset_mhz": primary_offset,
        "bandwidth_mhz": bandwidth,
    })
    
    # Alternate band pair if provided
    if alternate_offset:
        band_pairs.append({
            "lower_offset_mhz": alternate_offset,
            "upper_offset_mhz": alternate_offset,
            "bandwidth_mhz": bandwidth,
        })
    
    return band_pairs


def _build_acpr_test_case(
    row: dict[str, Any],
    *,
    source_file_name: str,
    sheet_name: str,
    row_index: int,
) -> RFTestCase:
    test_name = _cell_text(row.get("test_name")) or f"{sheet_name} row {row_index}"
    test_id = f"RF_XLSX_{_slug(sheet_name)}_{row_index:03d}_{str(uuid.uuid4())[:4].upper()}"

    servo_power = _number_list(row.get("servo_output_power_dbm"))
    input_power = None if servo_power else _number_list(row.get("input_power_dbm"))

    modulation = _cell_text(row.get("modulation"))
    
    data = {
        "schema_version": "inventide.rf.testcase.v1",
        "domain": "RF",
        "test_id": test_id,
        "test_type": "ACPR",
        "bench_type": "RF_BENCH",
        "objective": f"Run ACPR test from Excel workbook row: {test_name}",
        "stimulus": {
            "frequency_mhz": _number_list(row.get("frequency_mhz")),
            "dut_mode_name": _string_list(row.get("dut_mode_name")) or ["DEFAULT_TX_MODE"],
            "dut_alternate_mode_name": None,
            "modulation": [modulation] if modulation else None,
            "input_power_dbm": input_power,
            "servo_output_power_dbm": servo_power,
            "bandwidth_mhz": _number_list(row.get("bandwidth_mhz")),
            "duty_cycle_pct": None,
            "acpr_band_pairs": _acpr_band_pairs_from_row(row),
        },
        "dut_bias": {
            "vcc_v": _number_list(row.get("vcc_v")),
            "vdd_v": _number_list(row.get("vdd_v")),
        },
        "environment": {
            "temperature_c": _number_list(row.get("temperature_c")) or [25.0],
        },
        "measurement": {
            "metric": "ACPR",
            "limit": _limit_from_row(row),
            "additional_metrics": _additional_metrics(row.get("additional_metrics")),
        },
        "evm_settings": None,
        "s_parameter_settings": None,
        "sweep": None,
        "source_evidence": [
            f"{source_file_name} | sheet={sheet_name} | row={row_index} | test_name={test_name}"
        ],
        "confidence": "high",
        "execution_target": {
            "runner": "Generic_RF_TestRunner",
            "config_format": "csv",
        },
    }

    sweep_axes = []

    if data["stimulus"]["frequency_mhz"] and len(data["stimulus"]["frequency_mhz"]) > 1:
        sweep_axes.append("frequency_mhz")

    if data["stimulus"]["servo_output_power_dbm"] and len(data["stimulus"]["servo_output_power_dbm"]) > 1:
        sweep_axes.append("power")

    if data["stimulus"]["input_power_dbm"] and len(data["stimulus"]["input_power_dbm"]) > 1:
        sweep_axes.append("power")

    if data["dut_bias"]["vcc_v"] and len(data["dut_bias"]["vcc_v"]) > 1:
        sweep_axes.append("vcc_v")

    if data["dut_bias"]["vdd_v"] and len(data["dut_bias"]["vdd_v"]) > 1:
        sweep_axes.append("vdd_v")

    if data["environment"]["temperature_c"] and len(data["environment"]["temperature_c"]) > 1:
        sweep_axes.append("temperature_c")

    if data["stimulus"]["dut_mode_name"] and len(data["stimulus"]["dut_mode_name"]) > 1:
        sweep_axes.append("dut_mode_name")

    if data["stimulus"]["modulation"] and len(data["stimulus"]["modulation"]) > 1:
        sweep_axes.append("modulation")

    if sweep_axes:
        data["sweep"] = {
            "sweep_type": "full_factorial",
            "expansion_order": list(dict.fromkeys(sweep_axes)),
        }

    return RFTestCase.model_validate(data)


def _build_evm_test_case(
    row: dict[str, Any],
    *,
    source_file_name: str,
    sheet_name: str,
    row_index: int,
) -> RFTestCase:
    test_name = _cell_text(row.get("test_name")) or f"{sheet_name} row {row_index}"
    test_id = f"RF_XLSX_{_slug(sheet_name)}_{row_index:03d}_{str(uuid.uuid4())[:4].upper()}"

    servo_power = _number_list(row.get("servo_output_power_dbm"))
    input_power = None if servo_power else _number_list(row.get("input_power_dbm"))

    duty_cycle = _number_list(row.get("duty_cycle_pct"))
    evm_type = (_cell_text(row.get("evm_type")) or "DYNAMIC").upper()

    if duty_cycle:
        evm_type = "DYNAMIC"

    modulation = _modulation_from_row(row)

    data = {
        "schema_version": "inventide.rf.testcase.v1",
        "domain": "RF",
        "test_id": test_id,
        "test_type": "EVM",
        "bench_type": "RF_BENCH",
        "objective": f"Run EVM test from Excel workbook row: {test_name}",
        "stimulus": {
            "frequency_mhz": _number_list(row.get("frequency_mhz")),
            "dut_mode_name": _string_list(row.get("dut_mode_name")) or ["DEFAULT_TX_MODE"],
            "dut_alternate_mode_name": "OFF" if evm_type == "DYNAMIC" else None,
            "modulation": [modulation] if modulation else None,
            "input_power_dbm": input_power,
            "servo_output_power_dbm": servo_power,
            "bandwidth_mhz": _number_list(row.get("bandwidth_mhz")),
            "duty_cycle_pct": duty_cycle,
            "acpr_band_pairs": None,
        },
        "dut_bias": {
            "vcc_v": _number_list(row.get("vcc_v")),
            "vdd_v": _number_list(row.get("vdd_v")),
        },
        "environment": {
            "temperature_c": _number_list(row.get("temperature_c")) or [25.0],
        },
        "measurement": {
            "metric": "EVM",
            "limit": _limit_from_row(row),
            "additional_metrics": _additional_metrics(row.get("additional_metrics")),
        },
        "evm_settings": {
            "evm_type": evm_type,
        },
        "sweep": None,
        "source_evidence": [
            f"{source_file_name} | sheet={sheet_name} | row={row_index} | test_name={test_name}"
        ],
        "confidence": "high",
        "execution_target": {
            "runner": "Generic_RF_TestRunner",
            "config_format": "csv",
        },
    }

    sweep_axes = []

    if data["stimulus"]["frequency_mhz"] and len(data["stimulus"]["frequency_mhz"]) > 1:
        sweep_axes.append("frequency_mhz")

    if data["stimulus"]["servo_output_power_dbm"] and len(data["stimulus"]["servo_output_power_dbm"]) > 1:
        sweep_axes.append("power")

    if data["stimulus"]["input_power_dbm"] and len(data["stimulus"]["input_power_dbm"]) > 1:
        sweep_axes.append("power")

    if data["dut_bias"]["vcc_v"] and len(data["dut_bias"]["vcc_v"]) > 1:
        sweep_axes.append("vcc_v")

    if data["dut_bias"]["vdd_v"] and len(data["dut_bias"]["vdd_v"]) > 1:
        sweep_axes.append("vdd_v")

    if data["environment"]["temperature_c"] and len(data["environment"]["temperature_c"]) > 1:
        sweep_axes.append("temperature_c")

    if data["stimulus"]["duty_cycle_pct"] and len(data["stimulus"]["duty_cycle_pct"]) > 1:
        sweep_axes.append("duty_cycle_pct")

    if sweep_axes:
        data["sweep"] = {
            "sweep_type": "full_factorial",
            "expansion_order": list(dict.fromkeys(sweep_axes)),
        }

    return RFTestCase.model_validate(data)


def generate_rf_tests_from_excel_workbook(path: str | Path) -> RFSpecGenerationResponse:
    """
    Reads a structured MVP Excel workbook and returns RFSpecGenerationResponse.

    This mirrors the PDF spec generation response shape so the UI can use the
    same JSON preview, validation, and CSV generation pipeline.
    """
    workbook_path = Path(path)

    if workbook_path.suffix.lower() != ".xlsx":
        return RFSpecGenerationResponse(
            status="no_tests_found",
            summary=f"Unsupported workbook type: {workbook_path.suffix}. MVP Excel ingestion supports .xlsx only.",
            generated_tests=[],
            rejected_candidates=[
                {
                    "candidate": workbook_path.name,
                    "reason": "Only .xlsx workbooks are supported in the MVP Excel path.",
                }
            ],
            default_assumptions=[],
        )

    wb = load_workbook(workbook_path, data_only=True, read_only=False)

    generated_tests: list[RFTestCase] = []
    rejected_candidates: list[dict[str, str]] = []

    for sheet in wb.worksheets:
        if sheet.sheet_state != "visible":
            continue

        header_row, header_map = _find_header_row(sheet)

        if header_row is None:
            rejected_candidates.append(
                {
                    "candidate": f"Sheet {sheet.title}",
                    "reason": "No MVP header row found. Required columns: test_name, test_type.",
                }
            )
            continue

        for row_index in range(header_row + 1, sheet.max_row + 1):
            row = _row_dict(sheet, row_index, header_map)

            if all(_is_blank(value) for value in row.values()):
                continue

            test_type = (_cell_text(row.get("test_type")) or "").upper()

            if test_type not in SUPPORTED_EXCEL_TEST_TYPES:
                rejected_candidates.append(
                    {
                        "candidate": f"{sheet.title} row {row_index}",
                        "reason": f"Excel MVP currently supports only EVM and ACPR rows; got {test_type or 'blank'}.",
                    }
                )
                continue

            try:
                if test_type == "EVM":
                    generated_tests.append(
                        _build_evm_test_case(
                            row,
                            source_file_name=workbook_path.name,
                            sheet_name=sheet.title,
                            row_index=row_index,
                        )
                    )
                elif test_type == "ACPR":
                    generated_tests.append(
                        _build_acpr_test_case(
                            row,
                            source_file_name=workbook_path.name,
                            sheet_name=sheet.title,
                            row_index=row_index,
                        )
                    )
            except Exception as exc:
                rejected_candidates.append(
                    {
                        "candidate": f"{sheet.title} row {row_index}",
                        "reason": f"Could not convert row to RFTestCase: {exc}",
                    }
                )

    status = "ready" if generated_tests else "no_tests_found"

    summary = (
        f"Generated {len(generated_tests)} RF tests from structured Excel workbook {workbook_path.name}."
        if generated_tests
        else f"No supported RF tests were generated from Excel workbook {workbook_path.name}."
    )

    return RFSpecGenerationResponse(
        status=status,
        summary=summary,
        generated_tests=generated_tests,
        rejected_candidates=rejected_candidates,
        default_assumptions=[
            "Excel workbook ingestion used the structured MVP format; each populated row became one RFTestCase.",
            "burst_length was used as the bench modulation/waveform name when present (EVM tests).",
            "DYNAMIC EVM rows default dut_alternate_mode_name to OFF.",
        ],
    )
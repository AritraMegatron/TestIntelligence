import json
import re
import inspect
import asyncio
from pathlib import Path
from nicegui import ui

from app.adapters.rf.generator import generate_rf_test_from_chat
from app.adapters.rf.validator import validate_rf_test_case
from app.adapters.rf.config_generator import generate_labview_config_csv_multi
from app.adapters.rf.spec_generator import generate_rf_tests_from_spec_text
from app.excel.extractor import generate_rf_tests_from_excel_workbook
from app.pdf.extractor import extract_text_from_pdf
from app.db.duckdb_client import save_test_case, save_generated_config
from app.settings import UPLOAD_DIR


chat_messages = []

latest_test_cases = []
latest_config_path = None

# Lightweight MVP memory
pending_test_type = None
pending_rf_notes = []

# Fake bench connection state
connected_bench = None

# Uploaded spec paths.
# Key = slot name such as spec1/spec2/spec3.
# Value = saved PDF path under data/uploads.
uploaded_spec_paths_by_slot = {}


def detect_test_type(message: str):
    """
    Detect only clear test-type intent.
    Avoid treating DUT mode names like 'TX HIGH Gain' as a GAIN test.
    """
    msg = message.lower()

    if re.search(r"\b(evm|error vector magnitude)\b", msg):
        return "EVM"

    if re.search(r"\b(acpr|aclr|adjacent channel power|adjacent channel leakage)\b", msg):
        return "ACPR"

    if re.search(r"\b(current test|current consumption|supply current|icc|idd)\b", msg):
        return "CURRENT"

    gain_patterns = [
        r"\bgain test\b",
        r"\btest gain\b",
        r"\bmeasure gain\b",
        r"\bpower gain test\b",
        r"\bsmall signal gain\b",
    ]

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

    if any(re.search(pattern, msg) for pattern in gain_patterns):
        return "GAIN"

    return None


def get_dummy_last_calibration_date(bench_name: str) -> str:
    dummy_calibration_dates = {
        "RF Bench 01 - WLAN PA": "2026-05-15",
        "RF Bench 02 - 5G FEM": "2026-05-08",
        "RF Bench 03 - Generic SCPI": "2026-04-30",
        "Demo Bench - Local Simulator": "2026-06-01",
        "S-Parameter Bench": "2026-04-28",
    }

    return dummy_calibration_dates.get(bench_name, "Unknown")

def user_requested_defaults(message: str) -> bool:
    msg = message.lower()

    default_phrases = [
        "use default",
        "use defaults",
        "default values",
        "default parameters",
        "demo values",
        "make a demo",
        "generate with defaults",
        "with defaults",
        "other parameters use default",
        "other parameters use defaults",
        "for other parameters use default",
        "for other parameters use defaults",
        "keep everything else default",
        "everything else default",
    ]

    return any(phrase in msg for phrase in default_phrases)


def scroll_chat_to_bottom():
    ui.timer(
        0.1,
        lambda: ui.run_javascript("""
            const chatArea = document.getElementById('chat-area');
            if (chatArea) {
                chatArea.scrollTop = chatArea.scrollHeight;
            }
        """),
        once=True,
    )


def read_csv_preview(file_path: Path) -> str:
    if not file_path or not file_path.exists():
        return "No bench config CSV generated yet."

    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text()


def format_json_preview(test_cases: list) -> str:
    if not test_cases:
        return "No JSON generated yet."

    if len(test_cases) == 1:
        data = test_cases[0].model_dump()
    else:
        data = [tc.model_dump() for tc in test_cases]

    return "```json\n" + json.dumps(data, indent=2) + "\n```"


def format_validation_preview(validation_results: list) -> str:
    return "```json\n" + json.dumps(validation_results, indent=2) + "\n```"

def _is_active(values) -> bool:
    return values is not None and len(values) > 0


def _is_sweep(values) -> bool:
    return values is not None and len(values) > 1


def _value_text(values, unit: str = "") -> str:
    if not _is_active(values):
        return "not specified"

    suffix = f" {unit}" if unit else ""

    if len(values) == 1:
        return f"{values[0]}{suffix}"

    return f"{values}{suffix}"


def _scalar_text(value, unit: str = "") -> str:
    if value is None or value == "":
        return "not specified"

    suffix = f" {unit}" if unit else ""
    return f"{value}{suffix}"


def _active_power_field(test_case):
    stim = test_case.stimulus

    if _is_active(stim.input_power_dbm):
        return "input_power_dbm"

    if _is_active(stim.servo_output_power_dbm):
        return "servo_output_power_dbm"

    return None


def _active_power_values(test_case):
    stim = test_case.stimulus
    power_field = _active_power_field(test_case)

    if power_field == "input_power_dbm":
        return stim.input_power_dbm

    if power_field == "servo_output_power_dbm":
        return stim.servo_output_power_dbm

    return None


def _axis_values(test_case, axis_name: str):
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

    if axis_name == "bandwidth_mhz":
        return stim.bandwidth_mhz

    if axis_name == "duty_cycle_pct":
        return stim.duty_cycle_pct

    if axis_name == "dut_mode_name":
        return stim.dut_mode_name

    if axis_name == "modulation":
        return stim.modulation

    return None


def _axis_unit(axis_name: str) -> str:
    return {
        "frequency_mhz": "MHz",
        "power": "dBm",
        "vcc_v": "V",
        "vdd_v": "V",
        "temperature_c": "C",
        "duty_cycle_pct": "%",
        "dut_mode_name": "",
        "modulation": "",
    }.get(axis_name, "")


def format_power_strategy(test_case) -> str:
    stim = test_case.stimulus

    if _is_active(stim.servo_output_power_dbm):
        values = stim.servo_output_power_dbm

        if _is_sweep(values):
            return (
                f"Servo output power sweep: {_value_text(values, 'dBm')}\n"
                "- Expansion order uses the abstract axis 'power'.\n"
                "- input_power_dbm is null because the bench runner adjusts input power internally at each target output power."
            )

        return (
            f"Servo to output power: {_value_text(values, 'dBm')}\n"
            "- input_power_dbm is null because the bench runner adjusts input power internally."
        )

    if _is_active(stim.input_power_dbm):
        values = stim.input_power_dbm

        if _is_sweep(values):
            return (
                f"Input power sweep: {_value_text(values, 'dBm')}\n"
                "- Expansion order uses the abstract axis 'power'."
            )

        return f"Fixed input power: {_value_text(values, 'dBm')}"

    return "Power strategy: not specified"


def format_rf_test_chat_summary(test_case, validation: dict, default_assumptions: list | None = None) -> str:
    """
    Human-readable RF summary for the CHAT panel.

    This is intentionally different from the JSON preview.
    The goal is to make the chat feel useful to an RF engineer or RF manager.
    """

    stim = test_case.stimulus
    bias = test_case.dut_bias
    env = test_case.environment
    meas = test_case.measurement

    lines = []

    lines.append(f"Generated {test_case.test_type} test case: {test_case.test_id}")
    lines.append("")
    lines.append(f"Objective: {test_case.objective}")
    lines.append("")

    if test_case.test_type == "S_PARAMETER" and test_case.s_parameter_settings is not None:
        sp = test_case.s_parameter_settings

        lines.append("S-Parameter / VNA setup:")
        lines.append(f"- Bench type: {test_case.bench_type}")
        lines.append(f"- Ports: {sp.ports}")
        lines.append(f"- Measurements: {sp.measurements}")
        lines.append(f"- Frequency sweep: {sp.frequency_start_mhz} MHz to {sp.frequency_stop_mhz} MHz")
        lines.append(f"- Points: {sp.num_points}")
        lines.append(f"- IF bandwidth: {sp.if_bandwidth_hz} Hz")
        lines.append(f"- VNA power: {sp.vna_power_dbm} dBm")
        lines.append(f"- Calibration: {sp.calibration_type}")
        lines.append(f"- Result format: {sp.result_format}")
        lines.append("")

    lines.append("Stimulus:")
    lines.append(f"- Frequency: {_value_text(stim.frequency_mhz, 'MHz')}")
    lines.append(f"- DUT active mode: {_value_text(stim.dut_mode_name)}")

    alternate_mode = getattr(stim, "dut_alternate_mode_name", None)
    if alternate_mode:
        lines.append(f"- DUT alternate mode: {_scalar_text(alternate_mode)}")

    lines.append(f"- Modulation: {_value_text(stim.modulation)}")

    if _is_active(stim.bandwidth_mhz):
        if _is_sweep(stim.modulation) and len(stim.bandwidth_mhz) == len(stim.modulation):
            pairs = [
                f"{mod} → {bw} MHz"
                for mod, bw in zip(stim.modulation, stim.bandwidth_mhz)
            ]
            lines.append(f"- Bandwidth pairing: {', '.join(pairs)}")
        else:
            lines.append(f"- Bandwidth: {_value_text(stim.bandwidth_mhz, 'MHz')}")

    if _is_active(stim.duty_cycle_pct):
        lines.append(f"- Duty cycle: {_value_text(stim.duty_cycle_pct, '%')}")

    lines.append("")
    lines.append("Power strategy:")
    lines.append(f"- {format_power_strategy(test_case)}")

    if test_case.sweep and test_case.sweep.expansion_order:
        lines.append("")
        lines.append("Sweep plan:")
        lines.append(f"- Sweep type: {test_case.sweep.sweep_type}")
        lines.append(f"- Expansion order: {' > '.join(test_case.sweep.expansion_order)}")

        for axis in test_case.sweep.expansion_order:
            if axis == "bandwidth_mhz":
                # Safety: bandwidth is paired to modulation and should not be displayed as a sweep axis.
                continue

            values = _axis_values(test_case, axis)
            unit = _axis_unit(axis)
            lines.append(f"- {axis}: {_value_text(values, unit)}")

    lines.append("")
    lines.append("DUT bias:")
    lines.append(f"- VCC: {_value_text(bias.vcc_v, 'V')}")
    lines.append(f"- VDD: {_value_text(bias.vdd_v, 'V')}")

    lines.append("")
    lines.append("Environment:")
    lines.append(f"- Temperature: {_value_text(env.temperature_c, 'C')}")

    lines.append("")
    lines.append("Measurement:")
    lines.append(f"- Metric: {meas.metric}")

    additional_metrics = getattr(meas, "additional_metrics", None)
    if additional_metrics:
        lines.append(f"- Additional metrics: {', '.join(additional_metrics)}")

    if meas.limit is not None:
        lines.append(
            f"- Limit: {meas.limit.operator} {meas.limit.value} {meas.limit.unit}"
        )
    else:
        lines.append("- Limit: not specified")

    if test_case.test_type == "EVM" and test_case.evm_settings is not None:
        lines.append(f"- EVM type: {test_case.evm_settings.evm_type}")

        if test_case.evm_settings.evm_type == "DYNAMIC":
            lines.append(f"- Dynamic switching: active {_value_text(stim.dut_mode_name)} → alternate {_scalar_text(alternate_mode)}")

    if test_case.test_type == "ACPR" and stim.acpr_band_pairs:
        lines.append("")
        lines.append("ACPR offset pairs:")
        for index, pair in enumerate(stim.acpr_band_pairs, start=1):
            lines.append(
                f"- Pair {index}: lower {pair.lower_offset_mhz} MHz, "
                f"upper {pair.upper_offset_mhz} MHz, "
                f"bandwidth {pair.bandwidth_mhz} MHz"
            )

    lines.append("")
    lines.append("Validation:")
    lines.append(f"- {'PASS' if validation.get('is_valid') else 'FAIL'}")

    if validation.get("errors"):
        lines.append("")
        lines.append("Errors:")
        for error in validation["errors"]:
            lines.append(f"- {error}")

    if validation.get("warnings"):
        lines.append("")
        lines.append("Warnings:")
        for warning in validation["warnings"]:
            lines.append(f"- {warning}")

    if default_assumptions:
        lines.append("")
        lines.append("Assumptions used:")
        for assumption in default_assumptions:
            lines.append(f"- {assumption}")

    lines.append("")
    lines.append("JSON and validation details are available in the right-side panels.")

    return "\n".join(lines)

def format_clarification_response(result) -> str:
    """
    Better clarification response for the chat panel.
    """

    lines = []
    lines.append("I can generate this RF test, but I need a few missing bench parameters first.")
    lines.append("")

    if result.clarifying_questions:
        lines.append("Missing information:")
        for question in result.clarifying_questions:
            lines.append(f"- {question}")
    else:
        lines.append("Missing information:")
        lines.append("- Please provide the remaining RF test details.")

    if result.default_assumptions:
        lines.append("")
        lines.append("Available demo defaults:")
        for assumption in result.default_assumptions:
            lines.append(f"- {assumption}")

    lines.append("")
    lines.append("You can also reply: use defaults")

    return "\n".join(lines)

async def save_uploaded_file(upload_event, slot_name: str) -> Path:
    """
    Saves NiceGUI uploaded file to data/uploads.
    Handles NiceGUI versions where upload data is stored in e.file.
    """

    file_obj = getattr(upload_event, "file", None)

    if file_obj is None:
        raise ValueError(
            "Upload event did not include .file. Check NiceGUI upload event structure."
        )

    original_name = (
        getattr(file_obj, "filename", None)
        or getattr(file_obj, "name", None)
        or f"{slot_name}_uploaded_spec.pdf"
    )

    safe_name = f"{slot_name}_{original_name}".replace(" ", "_")
    destination = UPLOAD_DIR / safe_name

    # Most likely file_obj is a Starlette UploadFile.
    # It may expose async read().
    if hasattr(file_obj, "seek"):
        result = file_obj.seek(0)
        if hasattr(result, "__await__"):
            await result

    if hasattr(file_obj, "read"):
        data = file_obj.read()
        if hasattr(data, "__await__"):
            data = await data
    elif hasattr(file_obj, "file"):
        inner_file = file_obj.file
        inner_file.seek(0)
        data = inner_file.read()
    else:
        raise ValueError("Could not read uploaded file data.")

    if isinstance(data, str):
        data = data.encode("utf-8")

    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"Upload content must be bytes, got {type(data)}")

    with destination.open("wb") as f:
        f.write(data)

    return destination


def create_chat_page():
    global latest_test_cases, latest_config_path
    global pending_test_type, pending_rf_notes, connected_bench
    global uploaded_spec_paths_by_slot

    ui.label("Inventide RF TestBridge").classes("text-3xl font-bold text-yellow-500")
    ui.label("AI Driven RF Test Assistant").classes("text-gray-300 text-lg")

    # -----------------------------
    # Top bench connection bar
    # -----------------------------
    with ui.row().classes("w-full items-center justify-between mt-4"):
        with ui.row().classes("items-center gap-3"):
            def on_bench_change(e):
                selected_bench = getattr(e, "value", bench_dropdown.value)

                bench_calibration_label.text = (
                    f"Last calibrated: {get_dummy_last_calibration_date(selected_bench)}"
                )
                bench_calibration_label.update()

            bench_dropdown = ui.select(
                options=[
                    "RF Bench 01 - WLAN PA",
                    "RF Bench 02 - 5G FEM",
                    "RF Bench 03 - Generic SCPI",
                    "Demo Bench - Local Simulator",
                    "S-Parameter Bench",
                ],
                value="Demo Bench - Local Simulator",
                label="Bench Name",
                on_change=on_bench_change,
            ).props("outlined dense").classes("w-72")

            connect_button = ui.button("Connect").classes(
                "bg-blue-500 text-white font-bold"
            )

            bench_calibration_label = ui.label(
                f"Last calibrated: {get_dummy_last_calibration_date(bench_dropdown.value)}"
            ).classes("text-sm text-gray-300")

        with ui.row().classes("items-center gap-2"):
            ui.label("Connection").classes("text-sm text-gray-300")

            connection_dot = ui.element("div").classes(
                "w-8 h-8 rounded-full bg-red-500 border-4 border-blue-500"
            )

            connection_text = ui.label("Disconnected").classes(
                "text-sm text-red-400 font-bold"
            )

    def connect_bench():
        global connected_bench

        selected = bench_dropdown.value
        connected_bench = selected

        connection_dot.classes(remove="bg-red-500")
        connection_dot.classes(add="bg-green-500")

        connection_text.text = f"Connected: {selected}"
        connection_text.classes(remove="text-red-400")
        connection_text.classes(add="text-green-400")

        ui.notify(f"Connected to {selected}", type="positive")

    connect_button.on("click", connect_bench)

    # -----------------------------
    # Main working area
    # Layout:
    # Left: SPEC/CHAT tabs
    # Middle: JSON preview
    # Right: Validation + Generate Bench Config button
    # -----------------------------
    with ui.row().classes("w-full gap-4 mt-6 items-start"):

        # -------------------------------------------------
        # Left panel: SPEC / CHAT tabs
        # -------------------------------------------------
        with ui.column().classes("w-[46%]"):
            with ui.tabs().classes("w-full") as tabs:
                specs_tab = ui.tab("SPEC")
                chat_tab = ui.tab("CHAT")

            with ui.tab_panels(tabs, value=specs_tab).classes(
                "w-full h-[430px] border rounded-lg bg-gray-900"
            ):
                # -----------------------------
                # Specs tab
                # -----------------------------
                with ui.tab_panel(specs_tab).classes("p-4 pb-8"):
                    ui.label("Upload up to 3 RF spec sheets or datasheets").classes(
                        "text-lg font-bold text-yellow-500"
                    )

                    spec_status = ui.markdown("No specs uploaded yet.").classes(
                        "text-sm text-gray-300"
                    )

                    process_status = ui.label("").classes("text-sm text-yellow-400 mt-1")

                    generation_mode = ui.select(
                        options=["Demo Defaults", "Conservative"],
                        value="Demo Defaults",
                        label="Generation Mode",
                    ).props("outlined dense").classes("w-64 mt-2")

                    async def handle_spec_upload(e):
                        global uploaded_spec_paths_by_slot

                        print("UPLOAD EVENT TYPE:", type(e))
                        print("UPLOAD EVENT DIR:", dir(e))
                        print("UPLOAD EVENT DICT:", getattr(e, "__dict__", None))

                        if len(uploaded_spec_paths_by_slot) >= 3:
                            ui.notify("Maximum 3 spec sheets allowed for MVP.", type="warning")
                            return

                        slot_name = f"spec{len(uploaded_spec_paths_by_slot) + 1}"

                        try:
                            saved_path = await save_uploaded_file(e, slot_name)
                        except Exception as ex:
                            ui.notify(f"Upload failed: {str(ex)}", type="negative")
                            print(f"[UPLOAD ERROR] {ex}")
                            return

                        uploaded_spec_paths_by_slot[slot_name] = saved_path

                        uploaded_files = list(uploaded_spec_paths_by_slot.values())

                        spec_status.set_content(
                            "\n".join([f"- {p.name}" for p in uploaded_files])
                        )

                        print(f"[UPLOAD] {slot_name}: {saved_path}")
                        print("Current uploaded files:", uploaded_spec_paths_by_slot)

                        ui.notify(f"Uploaded {saved_path.name}", type="positive")

                    def clear_uploaded_specs():
                        global uploaded_spec_paths_by_slot

                        uploaded_spec_paths_by_slot = {}
                        spec_status.set_content("No sources uploaded yet.")
                        ui.notify("Cleared uploaded sources.", type="info")

                    with ui.column().classes("w-full gap-2 mt-4"):
                        ui.upload(
                            label="Upload RF Specs / Datasheets / Excel Workbooks",
                            on_upload=handle_spec_upload,
                            multiple=True,
                            max_files=3,
                        ).props("accept=.pdf,.xlsx").classes("w-full")

                    with ui.row().classes("w-full items-center gap-3 mt-3 mb-10"):
                        process_button = ui.button("Process Sources").classes(
                            "bg-blue-500 text-white font-bold"
                        )

                        ui.button("Clear Specs", on_click=clear_uploaded_specs).classes(
                            "bg-gray-700 text-white font-bold"
                        )

                # -----------------------------
                # Chat tab
                # -----------------------------
                with ui.tab_panel(chat_tab).classes("p-3"):
                    chat_area = ui.column().classes(
                        "w-full h-[340px] border rounded-lg p-4 overflow-auto bg-gray-950"
                    ).props('id="chat-area"')

                    with ui.row().classes("w-full items-center gap-2 mt-2"):
                        user_input = ui.input(
                            placeholder="Example: servo to output power 25 dBm, duty cycle 45%, DUT mode TX HIGH GAIN. Keep everything else default"
                        ).props("outlined clearable dense").classes(
                            "flex-grow text-white bg-gray-800"
                        )

                        send_button = ui.button("Send").classes(
                            "bg-blue-500 text-white font-bold"
                        )

        # -------------------------------------------------
        # Middle panel: JSON
        # -------------------------------------------------
        with ui.column().classes("w-[32%]"):
            ui.label("Json Format").classes("h-[48px] flex items-end text-xl font-bold")

            json_preview = ui.markdown("No JSON generated yet.").classes(
                "w-full h-[430px] border rounded-lg p-4 overflow-auto bg-gray-950 text-sm"
            )

        # -------------------------------------------------
        # Right panel: Validation + Generate Bench Config button
        # -------------------------------------------------
        with ui.column().classes("w-[20%]"):
            ui.label("Validation").classes("h-[48px] flex items-end text-xl font-bold")

            # Total height matches JSON/chat panel: 430px.
            # justify-between keeps the button bottom aligned with JSON/chat panels.
            with ui.column().classes("w-full h-[430px] justify-between"):
                validation_preview = ui.markdown("No validation yet.").classes(
                    "w-full h-[360px] border rounded-lg p-4 overflow-auto bg-gray-950 text-sm"
                )

                def export_config():
                    global latest_test_cases, latest_config_path

                    if not latest_test_cases:
                        ui.notify("No test cases generated yet.", type="warning")
                        return

                    if len(latest_test_cases) == 1:
                        file_name = f"{latest_test_cases[0].test_id}_bench_config.csv"
                    else:
                        file_name = "rf_spec_generated_bench_config.csv"

                    file_path = generate_labview_config_csv_multi(
                        latest_test_cases,
                        file_name=file_name,
                    )

                    latest_config_path = file_path

                    for tc in latest_test_cases:
                        save_generated_config(tc.test_id, str(file_path))

                    csv_text = read_csv_preview(file_path)

                    csv_preview.set_content(
                        "```csv\n"
                        + csv_text
                        + "\n```"
                    )

                    ui.notify(f"Generated bench config: {file_path.name}", type="positive")

                ui.button("Generate Bench Config", on_click=export_config).classes(
                    "w-full h-[48px] bg-blue-500 text-white font-bold"
                )

    # -----------------------------
    # Bottom CSV preview panel
    # -----------------------------
    with ui.column().classes("w-full mt-6"):
        ui.label("Generated Bench Config CSV").classes("text-xl font-bold")

        csv_preview = ui.markdown("No bench config CSV generated yet.").classes(
            "w-full h-56 border rounded-lg p-4 overflow-auto bg-gray-950 text-sm"
        )

        # -----------------------------
        # Chat send logic
        # -----------------------------
        async def send_message(e=None):
            """
            Async chat handler.

            The LLM call is blocking, so it is executed in a background thread using
            asyncio.to_thread(). All NiceGUI UI updates remain on the main async path.
            """
            global latest_test_cases, latest_config_path
            global pending_test_type, pending_rf_notes

            msg = (user_input.value or "").strip()
            if not msg:
                return

            # Prevent double-submit while the LLM call is running.
            send_button.disable()
            send_button.text = "Generating..."
            send_button.update()

            user_input.value = ""
            user_input.update()

            detected_type = detect_test_type(msg)

            if detected_type:
                pending_test_type = detected_type
                pending_rf_notes = [f"Selected test type: {pending_test_type}"]

            if pending_test_type:
                pending_rf_notes.append(f"User detail: {msg}")

            chat_messages.append({"role": "user", "content": msg})

            with chat_area:
                ui.chat_message(text=msg, name="You", sent=True)
            scroll_chat_to_bottom()

            effective_msg = msg
            pending_context_text = "\n".join(pending_rf_notes)

            if user_requested_defaults(msg) and pending_test_type:
                effective_msg = f"""
    Continue the existing RF test request.

    The selected test type is {pending_test_type}.

    Preserve all user-provided parameters from this pending context:
    {pending_context_text}

    For all remaining missing parameters, use demo default values.

    Important:
    - Do not change the test type.
    - If the selected test type is EVM, generate an EVM test.
    - If the selected test type is GAIN, generate a GAIN test.
    - If the selected test type is ACPR, generate an ACPR test.
    - If the selected test type is CURRENT, generate a CURRENT test.
    - If the user provided DUT mode name, preserve it exactly.
    - If the user provided servo output power, preserve it exactly.
    - If the user provided duty cycle, preserve it exactly.
    - If the user provided frequency, modulation, input power, VCC, VDD, or bandwidth, preserve them exactly.
    """

            recent_context = "\n".join(
                [f"{m['role']}: {m['content']}" for m in chat_messages[-8:]]
            )

            try:
                # This is the important async change.
                # generate_rf_test_from_chat() calls the OpenAI API and is blocking,
                # so we move it off the UI event loop.
                result = await asyncio.to_thread(
                    generate_rf_test_from_chat,
                    effective_msg,
                    recent_context,
                )

                if result.status == "needs_clarification":
                    response_text = format_clarification_response(result)

                    chat_messages.append({"role": "assistant", "content": response_text})

                    with chat_area:
                        ui.chat_message(
                            text=response_text,
                            name="TestBridge AI",
                            sent=False,
                        )

                    scroll_chat_to_bottom()
                    return

                test_case = result.test_case
                latest_test_cases = [test_case]
                latest_config_path = None

                pending_test_type = None
                pending_rf_notes = []

                validation = validate_rf_test_case(test_case)

                # Local DB write is fast, but keep it off the UI loop too.
                await asyncio.to_thread(save_test_case, test_case)

                response_text = format_rf_test_chat_summary(
                    test_case=test_case,
                    validation=validation,
                    default_assumptions=result.default_assumptions,
                )

                chat_messages.append({"role": "assistant", "content": response_text})

                with chat_area:
                    ui.chat_message(
                        text=response_text,
                        name="TestBridge AI",
                        sent=False,
                    )
                scroll_chat_to_bottom()

                json_preview.set_content(format_json_preview(latest_test_cases))
                validation_preview.set_content(format_validation_preview([validation]))
                csv_preview.set_content("No bench config CSV generated yet.")

            except Exception as ex:
                error_text = f"Error generating RF test case: {str(ex)}"

                chat_messages.append({"role": "assistant", "content": error_text})

                with chat_area:
                    ui.chat_message(
                        text=error_text,
                        name="TestBridge AI",
                        sent=False,
                    )
                scroll_chat_to_bottom()

                validation_preview.set_content(
                    "```json\n"
                    + json.dumps({"error": error_text}, indent=2)
                    + "\n```"
                )

                ui.notify(error_text, type="negative")

            finally:
                send_button.enable()
                send_button.text = "Send"
                send_button.update()

        send_button.on("click", send_message)
        user_input.on("keydown.enter", send_message)

    # -----------------------------
    # Source processing logic: PDF specs + structured XLSX workbooks
    # -----------------------------
    async def process_specs():
        global latest_test_cases, latest_config_path

        uploaded_source_paths = list(uploaded_spec_paths_by_slot.values())

        print("Uploaded files:", uploaded_spec_paths_by_slot)

        if not uploaded_source_paths:
            ui.notify("Please upload at least one PDF spec sheet or XLSX workbook.", type="warning")
            process_status.text = "No uploaded source found. Please upload a PDF spec sheet or XLSX workbook first."
            process_status.update()
            return

        process_button.disable()
        process_button.text = "Processing..."
        process_button.update()

        process_status.text = "Reading uploaded source files..."
        process_status.update()
        ui.notify("Reading uploaded source files...", type="info")
        await asyncio.sleep(0.05)

        try:
            pdf_paths = [p for p in uploaded_source_paths[:3] if p.suffix.lower() == ".pdf"]
            workbook_paths = [p for p in uploaded_source_paths[:3] if p.suffix.lower() == ".xlsx"]

            unsupported_paths = [
                p for p in uploaded_source_paths[:3]
                if p.suffix.lower() not in {".pdf", ".xlsx"}
            ]

            all_generated_tests = []
            all_default_assumptions = []
            all_rejected_candidates = []
            source_summaries = []

            for path in unsupported_paths:
                all_rejected_candidates.append(
                    {
                        "candidate": path.name,
                        "reason": "Unsupported file type. MVP source ingestion supports .pdf and .xlsx.",
                    }
                )

            # -----------------------------
            # Structured Excel workbook path
            # -----------------------------
            for workbook_path in workbook_paths:
                process_status.text = f"Extracting EVM tests from workbook {workbook_path.name}..."
                process_status.update()
                ui.notify(f"Processing workbook {workbook_path.name}...", type="info")
                await asyncio.sleep(0.05)

                workbook_result = await asyncio.to_thread(
                    generate_rf_tests_from_excel_workbook,
                    workbook_path,
                )

                source_summaries.append(workbook_result.summary)
                all_generated_tests.extend(workbook_result.generated_tests)
                all_rejected_candidates.extend(workbook_result.rejected_candidates)

                for assumption in workbook_result.default_assumptions:
                    if assumption not in all_default_assumptions:
                        all_default_assumptions.append(assumption)

            # -----------------------------
            # PDF spec/datasheet path
            # -----------------------------
            if pdf_paths:
                combined_text_parts = []

                for path in pdf_paths:
                    print(f"Extracting text from: {path}")

                    process_status.text = f"Extracting text from {path.name}..."
                    process_status.update()
                    await asyncio.sleep(0.05)

                    text = await asyncio.to_thread(extract_text_from_pdf, path)

                    if not text or not text.strip():
                        print(f"No extractable text found in: {path.name}")
                        combined_text_parts.append(
                            f"\n\n===== FILE: {path.name} =====\n"
                            f"No extractable text found in this PDF."
                        )
                    else:
                        print(f"Extracted {len(text)} characters from {path.name}")
                        combined_text_parts.append(
                            f"\n\n===== FILE: {path.name} =====\n{text}"
                        )

                combined_text = "\n".join(combined_text_parts)

                print("COMBINED PDF TEXT PREVIEW:")
                print(combined_text[:2000])

                if combined_text.strip():
                    process_status.text = "Sending extracted PDF specs to Test Engine..."
                    process_status.update()
                    ui.notify("Generating RF tests from PDF specs...", type="info")
                    print("Calling LLM for PDF spec generation...")
                    await asyncio.sleep(0.05)

                    spec_result = await asyncio.to_thread(
                        generate_rf_tests_from_spec_text,
                        combined_text,
                        [p.name for p in pdf_paths],
                    )

                    source_summaries.append(spec_result.summary)
                    all_generated_tests.extend(spec_result.generated_tests)
                    all_rejected_candidates.extend(spec_result.rejected_candidates)

                    for assumption in spec_result.default_assumptions:
                        if assumption not in all_default_assumptions:
                            all_default_assumptions.append(assumption)
                else:
                    source_summaries.append("No text could be extracted from uploaded PDF specs.")

            process_status.text = "Validating generated RF tests..."
            process_status.update()
            await asyncio.sleep(0.05)

            latest_test_cases = all_generated_tests
            latest_config_path = None

            if not latest_test_cases:
                validation_summary = {
                    "source_summary": source_summaries,
                    "generated_count": 0,
                    "default_assumptions": all_default_assumptions,
                    "rejected_candidates": all_rejected_candidates,
                    "validation_results": [],
                }

                json_preview.set_content("No valid RF test JSON generated from uploaded sources.")
                validation_preview.set_content(
                    "```json\n" + json.dumps(validation_summary, indent=2) + "\n```"
                )
                csv_preview.set_content("No bench config CSV generated yet.")

                process_status.text = "No valid RF tests were generated from the uploaded sources."
                process_status.update()

                ui.notify("No valid RF tests were generated from the uploaded sources.", type="warning")
                return

            validation_results = []

            for tc in latest_test_cases:
                validation = validate_rf_test_case(tc)
                validation_results.append(
                    {
                        "test_id": tc.test_id,
                        "test_type": tc.test_type,
                        **validation,
                    }
                )

                save_test_case(tc)

            json_preview.set_content(format_json_preview(latest_test_cases))

            validation_summary = {
                "source_summary": source_summaries,
                "generated_count": len(latest_test_cases),
                "default_assumptions": all_default_assumptions,
                "rejected_candidates": all_rejected_candidates,
                "validation_results": validation_results,
            }

            validation_preview.set_content(
                "```json\n" + json.dumps(validation_summary, indent=2) + "\n```"
            )

            csv_preview.set_content("No bench config CSV generated yet.")

            process_status.text = f"Done. Generated {len(latest_test_cases)} suggested RF tests."
            process_status.update()

            ui.notify(
                f"Generated {len(latest_test_cases)} suggested RF tests from uploaded sources.",
                type="positive",
            )

        except Exception as e:
            error_text = f"Error processing uploaded sources: {str(e)}"
            print(error_text)

            process_status.text = error_text
            process_status.update()

            validation_preview.set_content(
                "```json\n"
                + json.dumps({"error": error_text}, indent=2)
                + "\n```"
            )

            ui.notify(error_text, type="negative")

        finally:
            process_button.enable()
            process_button.text = "Process Sources"
            process_button.update()

    process_button.on("click", process_specs)

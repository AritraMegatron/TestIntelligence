from __future__ import annotations

import os

from nicegui import ui

from app.core.llm_client import (
    configure_llm_provider,
    get_llm_model,
    get_llm_provider,
)
from app.db.duckdb_client import init_db
from app.ui.chat_page import create_chat_page


def configure_llm_from_environment() -> None:
    """
    Select the application-wide LLM provider.

    Set one of these in .env:
        LLM_PROVIDER=openai
        LLM_PROVIDER=ollama
    """
    requested_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    match requested_provider:
        case "openai":
            configure_llm_provider("openai")
        case "ollama":
            configure_llm_provider("ollama")
        case _:
            raise RuntimeError(
                f"Invalid LLM_PROVIDER '{requested_provider}'. "
                "Expected 'openai' or 'ollama'."
            )


@ui.page("/")
def index():
    ui.dark_mode().enable()

    with ui.header().classes("bg-black text-white items-center"):
        ui.label("Inventide Test Intelligence").classes(
            "text-xl font-bold text-yellow-400"
        )
        ui.space()
        ui.label(
            f"LLM: {get_llm_provider().upper()} · {get_llm_model()}"
        ).classes("text-sm text-gray-300")

    with ui.column().classes("w-full p-0"):
        create_chat_page()


def main():
    configure_llm_from_environment()
    init_db()

    ui.run(
        title="Inventide Test Intelligence",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8090)),
        reload=False,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()

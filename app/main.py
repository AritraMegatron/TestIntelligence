from __future__ import annotations

import os

from nicegui import ui

from app.db.duckdb_client import init_db
from app.ui.chat_page import create_chat_page


@ui.page("/")
def index():
    ui.dark_mode().enable()

    with ui.header().classes("bg-black text-white"):
        ui.label("Inventide Test Intelligence").classes(
            "text-xl font-bold text-yellow-400"
        )

    with ui.column().classes("w-full p-0"):
        create_chat_page()


def main():
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
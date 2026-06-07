from nicegui import ui
import os
from app.db.duckdb_client import init_db
from app.ui.chat_page import create_chat_page


def main():
    init_db()

    ui.dark_mode().enable()

    with ui.header().classes("bg-black text-white"):
        ui.label("Inventide Test Intelligence").classes("text-xl font-bold text-yellow-500")

    with ui.column().classes("w-full p-6"):
        create_chat_page()

    ui.run(
        title="Inventide Test Intelligence",
        #host="127.0.0.1",
        #port=8090,
        #reload=True,
        host="0.0.0.0",
        port = int(os.environ.get("PORT", 8090)),
        reload = False
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
import json
import duckdb
from datetime import datetime
from app.settings import DB_PATH
from app.models.rf_models import RFTestCase


def get_connection():
    return duckdb.connect(str(DB_PATH))


def init_db():
    con = get_connection()

    con.execute("""
        CREATE TABLE IF NOT EXISTS generated_test_cases (
            id VARCHAR,
            test_id VARCHAR,
            domain VARCHAR,
            test_type VARCHAR,
            test_json JSON,
            created_at TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS generated_configs (
            id VARCHAR,
            test_id VARCHAR,
            file_path VARCHAR,
            created_at TIMESTAMP
        )
    """)

    con.close()


def save_test_case(test_case: RFTestCase):
    con = get_connection()

    con.execute(
        """
        INSERT INTO generated_test_cases
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            test_case.test_id,
            test_case.test_id,
            test_case.domain,
            test_case.test_type,
            json.dumps(test_case.model_dump()),
            datetime.now(),
        ],
    )

    con.close()


def save_generated_config(test_id: str, file_path: str):
    con = get_connection()

    con.execute(
        """
        INSERT INTO generated_configs
        VALUES (?, ?, ?, ?)
        """,
        [
            test_id,
            test_id,
            file_path,
            datetime.now(),
        ],
    )

    con.close()
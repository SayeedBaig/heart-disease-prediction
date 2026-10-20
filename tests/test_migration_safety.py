import importlib.util
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "alembic"
    / "versions"
    / "85178fb28e85_add_patient_auth_fields.py"
)
SPEC = importlib.util.spec_from_file_location("patient_auth_migration", MIGRATION_PATH)
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)


def test_patient_auth_migration_refuses_existing_accounts_before_schema_changes():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE patients "
                "(id INTEGER PRIMARY KEY, email VARCHAR(255) NOT NULL)"
            )
        )
        connection.execute(
            text("INSERT INTO patients (email) VALUES (:email)"),
            {"email": "synthetic-patient@example.test"},
        )

        before_columns = {column["name"] for column in inspect(connection).get_columns("patients")}
        context = MigrationContext.configure(connection)

        with Operations.context(context):
            with pytest.raises(RuntimeError, match="explicitly reviewed password recovery plan"):
                MIGRATION.upgrade()

        after_columns = {column["name"] for column in inspect(connection).get_columns("patients")}

    assert after_columns == before_columns

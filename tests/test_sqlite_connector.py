import hashlib
import inspect
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import app.streamlit_app as streamlit_app
from src.agents.plan_generator import PlanGenerationResult
from src.agents.capability_guard import Capability, CapabilityCheckResult
from src.connectors import DataConnector, SQLiteConnector, SQLiteConnectorError
from src.schemas.analysis_plan import parse_analysis_plan


def _create_database(tmp_path: Path) -> Path:
    database_path = tmp_path / "sales.db"
    with sqlite3.connect(database_path) as connection:
        pd.DataFrame(
            {
                "category": ["A", "B", "A"],
                "region": ["North", "South", "South"],
                "revenue": [100, 200, 50],
            }
        ).to_sql("orders", connection, index=False)
        pd.DataFrame({"customer": ["Acme"]}).to_sql(
            "customers",
            connection,
            index=False,
        )
    return database_path


def _valid_plan():
    return parse_analysis_plan(
        {
            "version": "1.0",
            "goal": "Rank categories by revenue",
            "operations": [
                {"operation": "groupby", "columns": ["category"]},
                {
                    "operation": "aggregate",
                    "metrics": [
                        {
                            "column": "revenue",
                            "function": "sum",
                            "alias": "total_revenue",
                        }
                    ],
                },
                {
                    "operation": "top_n",
                    "sort_by": "total_revenue",
                    "n": 1,
                    "ascending": False,
                },
            ],
        }
    )


def test_sqlite_connector_implements_data_connector(tmp_path) -> None:
    connector = SQLiteConnector(_create_database(tmp_path), "orders")

    assert isinstance(connector, DataConnector)


def test_list_tables_uses_sqlite_metadata(tmp_path) -> None:
    connector = SQLiteConnector(_create_database(tmp_path))

    assert connector.list_tables() == ["customers", "orders"]


def test_load_selected_table_into_dataframe(tmp_path) -> None:
    connector = SQLiteConnector(_create_database(tmp_path), "orders")

    dataframe = connector.load()

    assert dataframe.to_dict("records") == [
        {"category": "A", "region": "North", "revenue": 100},
        {"category": "B", "region": "South", "revenue": 200},
        {"category": "A", "region": "South", "revenue": 50},
    ]


def test_get_schema_reuses_standard_schema_summary(tmp_path) -> None:
    connector = SQLiteConnector(_create_database(tmp_path), "orders")

    summary = connector.get_schema()

    assert summary["column_names"] == ["category", "region", "revenue"]
    assert summary["number_of_rows"] == 3
    assert summary["schema_table"].columns.tolist() == [
        "column name",
        "data type",
        "missing values",
        "missing percentage",
    ]


def test_missing_table_selection_fails(tmp_path) -> None:
    connector = SQLiteConnector(_create_database(tmp_path))

    with pytest.raises(SQLiteConnectorError, match="Select a SQLite table"):
        connector.load()


def test_unavailable_table_fails_without_changing_database(tmp_path) -> None:
    database_path = _create_database(tmp_path)
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()
    connector = SQLiteConnector(database_path, "orders; DROP TABLE orders")

    with pytest.raises(SQLiteConnectorError, match="unavailable"):
        connector.load()

    after = hashlib.sha256(database_path.read_bytes()).hexdigest()
    assert after == before


def test_successful_read_does_not_modify_database(tmp_path) -> None:
    database_path = _create_database(tmp_path)
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()

    connector = SQLiteConnector(database_path, "orders")
    connector.load()
    connector.get_schema()

    after = hashlib.sha256(database_path.read_bytes()).hexdigest()
    assert after == before


def test_missing_database_file_fails(tmp_path) -> None:
    connector = SQLiteConnector(tmp_path / "missing.db", "orders")

    with pytest.raises(SQLiteConnectorError, match="not found"):
        connector.list_tables()


def test_invalid_database_file_fails(tmp_path) -> None:
    database_path = tmp_path / "invalid.db"
    database_path.write_bytes(b"not a sqlite database")
    connector = SQLiteConnector(database_path, "orders")

    with pytest.raises(SQLiteConnectorError, match="metadata could not be read"):
        connector.list_tables()


def test_loaded_dataframe_is_cached(tmp_path) -> None:
    connector = SQLiteConnector(_create_database(tmp_path), "orders")

    assert connector.load() is connector.load()


def test_sqlite_connector_dataframe_runs_existing_v5_workflow(tmp_path, monkeypatch) -> None:
    connector = SQLiteConnector(_create_database(tmp_path), "orders")
    dataframe = connector.load()
    schema_summary = connector.get_schema()
    monkeypatch.setattr(
        streamlit_app,
        "generate_structured_plan",
        lambda summary, question: PlanGenerationResult(success=True, plan=_valid_plan()),
    )
    monkeypatch.setattr(
        streamlit_app,
        "check_capability_boundary",
        lambda question, plan: CapabilityCheckResult(
            allowed=True,
            capability=Capability.RANKING,
            plan_matches_intent=True,
            message="Request capability and generated plan are compatible.",
        ),
    )

    preparation = streamlit_app.prepare_v5_plan(schema_summary, "Rank categories")
    result = streamlit_app.execute_v5_plan(
        dataframe,
        preparation.validated_plan,
        preparation.schema_signature,
    )

    assert preparation.success is True
    assert result.success is True
    assert result.dataframe.to_dict("records") == [
        {"category": "B", "total_revenue": 200}
    ]


def test_demo_database_contains_orders_table() -> None:
    database_path = Path("data/demo_sales.db")
    connector = SQLiteConnector(database_path, "orders")

    dataframe = connector.load()

    assert {"date", "product", "category", "region", "quantity", "revenue"} <= set(
        dataframe.columns
    )
    assert len(dataframe) == 24


def test_streamlit_exposes_csv_and_sqlite_source_selection() -> None:
    source = inspect.getsource(streamlit_app)

    assert "CSV Upload" in source
    assert "SQLite Database" in source
    assert "SQLiteConnector" in source

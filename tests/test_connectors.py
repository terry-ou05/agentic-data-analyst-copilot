from io import BytesIO
import inspect

import pandas as pd
import pytest

import app.streamlit_app as streamlit_app
import src.connectors.csv_connector as csv_connector_module
from src.agents.plan_generator import PlanGenerationResult
from src.connectors.base import DataConnector
from src.connectors.csv_connector import CsvConnector
from src.data.loader import CsvLoadError
from src.schemas.analysis_plan import parse_analysis_plan


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


def test_data_connector_is_an_abstract_interface() -> None:
    with pytest.raises(TypeError):
        DataConnector()


def test_csv_connector_loads_dataframe_from_uploaded_bytes() -> None:
    connector = CsvConnector(BytesIO(b"category,revenue\nA,10\nB,20\n"))

    dataframe = connector.load()

    assert dataframe.to_dict("records") == [
        {"category": "A", "revenue": 10},
        {"category": "B", "revenue": 20},
    ]


def test_csv_connector_reuses_existing_loader_once(monkeypatch) -> None:
    expected = pd.DataFrame({"category": ["A"], "revenue": [10]})
    calls = []

    def fake_load_csv(source):
        calls.append(source)
        return expected

    monkeypatch.setattr(csv_connector_module, "load_csv", fake_load_csv)
    source = BytesIO(b"category,revenue\nA,10\n")
    connector = CsvConnector(source)

    first = connector.load()
    second = connector.load()

    assert first is expected
    assert second is expected
    assert calls == [source]


def test_csv_connector_schema_uses_standard_schema_summary() -> None:
    connector = CsvConnector(BytesIO(b"category,revenue\nA,10\n"))

    summary = connector.get_schema()

    assert summary["column_names"] == ["category", "revenue"]
    assert summary["number_of_rows"] == 1
    assert summary["schema_table"].columns.tolist() == [
        "column name",
        "data type",
        "missing values",
        "missing percentage",
    ]


def test_get_schema_loads_connector_when_needed(monkeypatch) -> None:
    expected = pd.DataFrame({"category": ["A"]})
    calls = []

    def fake_load_csv(source):
        calls.append(source)
        return expected

    monkeypatch.setattr(csv_connector_module, "load_csv", fake_load_csv)
    connector = CsvConnector(BytesIO(b"category\nA\n"))

    summary = connector.get_schema()

    assert summary["column_names"] == ["category"]
    assert len(calls) == 1


@pytest.mark.parametrize("payload", [b"", b"   \r\n\t"])
def test_csv_connector_preserves_empty_csv_errors(payload) -> None:
    connector = CsvConnector(BytesIO(payload))

    with pytest.raises(CsvLoadError, match="empty or contains only whitespace"):
        connector.load()


def test_csv_connector_preserves_invalid_csv_errors() -> None:
    connector = CsvConnector(BytesIO(b'category,revenue\n"unterminated,10\n'))

    with pytest.raises(CsvLoadError, match="invalid or malformed"):
        connector.load()


def test_connector_dataframe_still_runs_v5_workflow(monkeypatch) -> None:
    connector = CsvConnector(
        BytesIO(b"category,revenue\nA,100\nB,200\nA,50\n")
    )
    dataframe = connector.load()
    schema_summary = connector.get_schema()
    monkeypatch.setattr(
        streamlit_app,
        "generate_structured_plan",
        lambda summary, question: PlanGenerationResult(success=True, plan=_valid_plan()),
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


def test_streamlit_uses_csv_connector_for_data_entry() -> None:
    source = inspect.getsource(streamlit_app)

    assert "CsvConnector(" in source
    assert "load_csv(" not in source

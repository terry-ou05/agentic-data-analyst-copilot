from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.cases import default_evaluation_cases
from src.evaluation.runner import EvaluationRunner, run_synthetic_evaluation
from src.evaluation.synthetic_data import (
    SYNTHETIC_TABLE_NAME,
    create_synthetic_sqlite_database,
    generate_synthetic_business_dataframe,
)


def test_synthetic_business_data_is_deterministic_and_business_shaped():
    dataframe = generate_synthetic_business_dataframe()

    assert len(dataframe) == 45
    assert dataframe.columns.tolist() == [
        "date",
        "product",
        "category",
        "region",
        "channel",
        "quantity",
        "revenue",
    ]
    assert dataframe["revenue"].dtype.kind in {"i", "u", "f"}
    assert dataframe.groupby("category")["revenue"].sum().idxmax() == "Computer"
    assert dataframe.groupby("category")["quantity"].mean().idxmax() == "Accessory"


def test_synthetic_database_is_loadable_by_existing_sqlite_connector(tmp_path):
    database_path = create_synthetic_sqlite_database(tmp_path / "evaluation.db")

    from src.connectors.sqlite_connector import SQLiteConnector

    connector = SQLiteConnector(database_path, SYNTHETIC_TABLE_NAME)
    dataframe = connector.load()
    schema = connector.get_schema()

    assert connector.list_tables() == [SYNTHETIC_TABLE_NAME]
    assert len(dataframe) == 45
    assert schema["column_names"] == dataframe.columns.tolist()


def test_synthetic_database_refuses_to_overwrite_existing_file(tmp_path):
    database_path = tmp_path / "existing.db"
    database_path.write_bytes(b"existing database placeholder")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_synthetic_sqlite_database(database_path)


def test_synthetic_database_rejects_invalid_table_name(tmp_path):
    with pytest.raises(ValueError, match="non-empty string"):
        create_synthetic_sqlite_database(tmp_path / "evaluation.db", table_name=" ")


def test_default_cases_are_versioned_structured_plan_cases():
    cases = default_evaluation_cases()

    assert len(cases) == 3
    assert {case.name for case in cases} == {
        "top_category_revenue",
        "online_region_revenue",
        "highest_average_quantity_category",
    }
    assert all(case.plan_payload["version"] == "1.0" for case in cases)
    assert all(case.plan_payload["operations"] for case in cases)


def test_full_evaluation_runs_existing_v5_v6_workflow_offline(tmp_path, monkeypatch):
    database_path = tmp_path / "evaluation.db"

    # The runner patches the module-local LLM call sites. This guard proves the
    # outer client is not used and no real network call is made.
    def unexpected_llm_call(*_args, **_kwargs):
        raise AssertionError("The real LLM client must not be called by evaluation.")

    monkeypatch.setattr("src.llm.client.generate_chat_completion", unexpected_llm_call)
    report = run_synthetic_evaluation(database_path)

    assert report.setup_error == ""
    assert report.total_cases == 3
    assert report.successful_cases == 3
    assert report.success_rate == 1.0
    assert report.data_load_duration_ms >= 0
    assert report.average_execution_duration_ms >= 0
    assert report.average_total_duration_ms >= 0
    for result in report.results:
        assert result.success is True
        assert result.error == ""
        assert result.plan_success is True
        assert result.validation_success is True
        assert result.execution_success is True
        assert result.profile_success is True
        assert result.visualization_success is True
        assert result.chart_success is True
        assert result.insight_success is True
        assert result.input_rows == 45
        assert result.output_rows == 1
        assert result.executed_operations
        assert result.total_duration_ms >= result.plan_duration_ms


def test_evaluation_reports_expected_business_results(tmp_path):
    report = run_synthetic_evaluation(tmp_path / "evaluation.db")
    records = report.to_dict()["results"]

    assert [record["name"] for record in records] == [
        "top_category_revenue",
        "online_region_revenue",
        "highest_average_quantity_category",
    ]
    assert all(record["metrics"]["chart_type"] == "bar" for record in records)
    assert all(record["phases"].values() for record in records)


def test_failed_business_assertion_is_reported_without_crashing(tmp_path):
    database_path = create_synthetic_sqlite_database(tmp_path / "evaluation.db")
    incorrect_case = replace(
        default_evaluation_cases()[0],
        expected_top_value="Not the generated answer",
    )

    report = EvaluationRunner(database_path).run([incorrect_case])

    assert report.total_cases == 1
    assert report.successful_cases == 0
    assert report.success_rate == 0.0
    assert report.results[0].success is False
    assert report.results[0].error == "Result did not match the expected business outcome."
    assert report.results[0].insight_success is True


def test_runner_handles_missing_database_as_structured_setup_failure(tmp_path):
    report = EvaluationRunner(tmp_path / "missing.db").run()

    assert report.total_cases == 0
    assert report.success_rate == 0.0
    assert report.setup_error == "Evaluation database could not be loaded."


def test_runner_accepts_an_explicit_empty_case_collection(tmp_path):
    database_path = create_synthetic_sqlite_database(tmp_path / "evaluation.db")
    report = EvaluationRunner(database_path).run(())

    assert report.setup_error == ""
    assert report.total_cases == 0
    assert report.success_rate == 0.0


def test_evaluation_does_not_mutate_synthetic_source_dataframe(tmp_path):
    database_path = create_synthetic_sqlite_database(tmp_path / "evaluation.db")
    source = generate_synthetic_business_dataframe()
    before = source.copy(deep=True)

    report = EvaluationRunner(database_path).run()

    pd.testing.assert_frame_equal(source, before)
    assert report.success_rate == 1.0

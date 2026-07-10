import pandas as pd

import app.streamlit_app as streamlit_app
from src.agents.insight_generator import InsightGenerationResult
from src.agents.plan_generator import PlanGenerationResult
from src.analysis.chart_generator import ChartGenerationError
from src.analysis.profiler import AggregatedResultSummary, AnalysisProfile
from src.analysis.visualization import VisualizationPlan
from src.data.schema import build_schema_summary
from src.schemas.analysis_plan import (
    create_validated_analysis_plan,
    parse_analysis_plan,
)
from src.utils.schema_signature import build_schema_signature


def _valid_plan():
    return parse_analysis_plan(
        {
            "version": "1.0",
            "goal": "Find the highest revenue category",
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


def _dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["A", "B", "A"],
            "revenue": [100, 200, 50],
        }
    )


def _mock_valid_plan_generation(monkeypatch) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "generate_structured_plan",
        lambda schema_summary, question: PlanGenerationResult(
            success=True,
            plan=_valid_plan(),
        ),
    )


def test_plan_generation_is_mocked_and_independently_validated(monkeypatch) -> None:
    dataframe = _dataframe()
    summary = build_schema_summary(dataframe)
    captured = {}

    def fake_generation(schema_summary, question):
        captured["schema_summary"] = schema_summary
        captured["question"] = question
        return PlanGenerationResult(success=True, plan=_valid_plan())

    monkeypatch.setattr(
        streamlit_app,
        "generate_structured_plan",
        fake_generation,
    )

    preparation = streamlit_app.prepare_v5_plan(
        summary,
        "Which category has the highest revenue?",
    )

    assert preparation.success is True
    assert preparation.validation_result.valid is True
    assert preparation.validated_plan is not None
    assert preparation.schema_signature == build_schema_signature(summary)
    assert captured["schema_summary"] is summary
    assert captured["question"] == "Which category has the highest revenue?"


def test_invalid_plan_cannot_reach_executor(monkeypatch) -> None:
    dataframe = _dataframe()
    summary = build_schema_summary(dataframe)
    invalid_plan = parse_analysis_plan(
        {
            "version": "1.0",
            "goal": "Use a missing column",
            "operations": [
                {
                    "operation": "filter",
                    "column": "missing_column",
                    "operator": "eq",
                    "value": "x",
                }
            ],
        }
    )
    monkeypatch.setattr(
        streamlit_app,
        "generate_structured_plan",
        lambda schema_summary, question: PlanGenerationResult(
            success=True,
            plan=invalid_plan,
        ),
    )

    preparation = streamlit_app.prepare_v5_plan(summary, "Use missing data")

    assert preparation.success is False
    assert preparation.validation_result.valid is False
    assert preparation.validated_plan is None

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Executor must not run an invalid plan")

    monkeypatch.setattr(streamlit_app, "execute_analysis_plan", fail_if_called)
    result = streamlit_app.execute_v5_plan(
        dataframe,
        preparation.validated_plan,
        preparation.schema_signature,
    )

    assert result.success is False
    assert result.error_code == "INVALID_PLAN_TYPE"


def test_validated_plan_executes_to_expected_dataframe(monkeypatch) -> None:
    dataframe = _dataframe()
    summary = build_schema_summary(dataframe)
    _mock_valid_plan_generation(monkeypatch)

    preparation = streamlit_app.prepare_v5_plan(summary, "Rank categories")
    result = streamlit_app.execute_v5_plan(
        dataframe,
        preparation.validated_plan,
        preparation.schema_signature,
    )

    expected = pd.DataFrame({"category": ["B"], "total_revenue": [200]})
    assert result.success is True
    pd.testing.assert_frame_equal(
        result.dataframe.reset_index(drop=True),
        expected,
    )


def test_schema_change_rejects_bound_plan(monkeypatch) -> None:
    dataframe = _dataframe()
    summary = build_schema_summary(dataframe)
    _mock_valid_plan_generation(monkeypatch)
    preparation = streamlit_app.prepare_v5_plan(summary, "Rank categories")

    changed_dataframe = dataframe.rename(columns={"category": "segment"})

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Executor must not run a schema-stale plan")

    monkeypatch.setattr(streamlit_app, "execute_analysis_plan", fail_if_called)
    result = streamlit_app.execute_v5_plan(
        changed_dataframe,
        preparation.validated_plan,
        preparation.schema_signature,
    )

    assert result.success is False
    assert result.error_code == "SCHEMA_CHANGED"
    assert result.message == "Dataset changed. Please regenerate analysis plan."


def test_v5_workflow_isolated_from_legacy_code_generator(monkeypatch) -> None:
    dataframe = _dataframe()
    summary = build_schema_summary(dataframe)
    _mock_valid_plan_generation(monkeypatch)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("V5 must not call generate_analysis_code")

    monkeypatch.setattr(streamlit_app, "generate_analysis_code", fail_if_called)

    preparation = streamlit_app.prepare_v5_plan(summary, "Rank categories")
    result = streamlit_app.execute_v5_plan(
        dataframe,
        preparation.validated_plan,
        preparation.schema_signature,
    )

    assert preparation.success is True
    assert result.success is True

    monkeypatch.setattr(
        streamlit_app,
        "generate_insight",
        lambda profile, summary, visualization: InsightGenerationResult(
            success=True,
            insight="Category B leads the result.",
        ),
    )
    report = streamlit_app.build_v52_report(
        result.dataframe,
        preparation.validated_plan,
    )

    assert report.insight_result.success is True


def test_new_dataset_clears_previous_v5_state() -> None:
    state = {}
    streamlit_app.initialize_v5_session_state(state)
    state.update(
        {
            "v5_plan": _valid_plan(),
            "v5_validation_result": object(),
            "v5_execution_result": object(),
            "v5_schema_signature": "schema-a",
            "v5_validated_plan": object(),
            "v5_dataset_identity": "file-a",
            "v5_analysis_profile": object(),
            "v5_visualization_plan": object(),
            "v5_chart": object(),
            "v5_chart_error": "old error",
            "v5_insight_result": object(),
        }
    )

    changed = streamlit_app.synchronize_v5_dataset_state(
        state,
        dataset_identity="file-b",
        current_schema_signature="schema-b",
    )

    assert changed is True
    assert state["v5_dataset_identity"] == "file-b"
    assert state["v5_plan"] is None
    assert state["v5_validation_result"] is None
    assert state["v5_execution_result"] is None
    assert state["v5_schema_signature"] is None
    assert state["v5_validated_plan"] is None
    assert state["v5_analysis_profile"] is None
    assert state["v5_visualization_plan"] is None
    assert state["v5_chart"] is None
    assert state["v5_chart_error"] == ""
    assert state["v5_insight_result"] is None


def test_rerun_with_same_dataset_preserves_v5_state() -> None:
    plan = _valid_plan()
    state = {}
    streamlit_app.initialize_v5_session_state(state)
    state.update(
        {
            "v5_plan": plan,
            "v5_schema_signature": "schema-a",
            "v5_dataset_identity": "file-a",
        }
    )

    changed = streamlit_app.synchronize_v5_dataset_state(
        state,
        dataset_identity="file-a",
        current_schema_signature="schema-a",
    )

    assert changed is False
    assert state["v5_plan"] is plan


def test_schema_signature_changes_with_rows_columns_or_types() -> None:
    base = _dataframe()
    base_signature = build_schema_signature(build_schema_summary(base))

    assert build_schema_signature(build_schema_summary(base.iloc[:2])) != base_signature
    assert (
        build_schema_signature(
            build_schema_summary(base.rename(columns={"category": "segment"}))
        )
        != base_signature
    )
    assert (
        build_schema_signature(
            build_schema_summary(base.assign(revenue=base["revenue"].astype(str)))
        )
        != base_signature
    )


def test_v52_report_chains_profile_chart_and_insight(monkeypatch) -> None:
    dataframe = pd.DataFrame(
        {"category": ["A", "B"], "total_revenue": [150, 200]}
    )
    validated_plan = create_validated_analysis_plan(
        _valid_plan(),
        {"column_names": ["category", "revenue"]},
    )
    monkeypatch.setattr(
        streamlit_app,
        "generate_insight",
        lambda profile, summary, visualization: InsightGenerationResult(
            success=True,
            insight="Category B leads the aggregated result.",
        ),
    )

    report = streamlit_app.build_v52_report(dataframe, validated_plan)

    assert isinstance(report.profile, AnalysisProfile)
    assert isinstance(report.visualization_plan, VisualizationPlan)
    assert report.visualization_plan.chart_type.value == "bar"
    assert report.chart is not None
    assert report.chart_error == ""
    assert report.insight_result.success is True


def test_v52_report_passes_only_metadata_to_insight(monkeypatch) -> None:
    dataframe = pd.DataFrame(
        {"category": ["A", "B"], "total_revenue": [150, 200]}
    )
    validated_plan = create_validated_analysis_plan(
        _valid_plan(),
        {"column_names": ["category", "revenue"]},
    )
    captured = {}

    def fake_insight(profile, summary, visualization):
        captured["profile"] = profile
        captured["summary"] = summary
        captured["visualization"] = visualization
        return InsightGenerationResult(success=True, insight="Metadata-only insight.")

    monkeypatch.setattr(streamlit_app, "generate_insight", fake_insight)

    streamlit_app.build_v52_report(dataframe, validated_plan)

    assert isinstance(captured["profile"], AnalysisProfile)
    assert isinstance(captured["summary"], AggregatedResultSummary)
    assert isinstance(captured["visualization"], VisualizationPlan)
    assert not isinstance(captured["profile"], pd.DataFrame)
    assert not isinstance(captured["summary"], pd.DataFrame)


def test_non_aggregate_report_withholds_rows_from_insight(monkeypatch) -> None:
    dataframe = pd.DataFrame(
        {"customer_name": ["Sensitive Person"], "revenue": [100]}
    )
    plan = parse_analysis_plan(
        {
            "version": "1.0",
            "goal": "Rank revenue",
            "operations": [
                {
                    "operation": "top_n",
                    "sort_by": "revenue",
                    "n": 1,
                    "ascending": False,
                }
            ],
        }
    )
    validated_plan = create_validated_analysis_plan(
        plan,
        {"column_names": ["customer_name", "revenue"]},
    )

    def fake_insight(profile, summary, visualization):
        assert summary.is_aggregated is False
        assert summary.rows == ()
        assert "Sensitive Person" not in str(summary.to_dict())
        return InsightGenerationResult(success=True, insight="Profile-only insight.")

    monkeypatch.setattr(streamlit_app, "generate_insight", fake_insight)

    report = streamlit_app.build_v52_report(dataframe, validated_plan)

    assert report.insight_result.success is True


def test_chart_failure_does_not_block_insight(monkeypatch) -> None:
    dataframe = pd.DataFrame(
        {"category": ["A", "B"], "total_revenue": [150, 200]}
    )
    validated_plan = create_validated_analysis_plan(
        _valid_plan(),
        {"column_names": ["category", "revenue"]},
    )

    def fail_chart(plan, result_dataframe):
        raise ChartGenerationError("CHART_FAILED", "Chart could not be generated.")

    monkeypatch.setattr(streamlit_app, "generate_chart", fail_chart)
    monkeypatch.setattr(
        streamlit_app,
        "generate_insight",
        lambda profile, summary, visualization: InsightGenerationResult(
            success=True,
            insight="Insight remains available.",
        ),
    )

    report = streamlit_app.build_v52_report(dataframe, validated_plan)

    assert report.chart is None
    assert report.chart_error == "Chart could not be generated."
    assert report.insight_result.success is True


def test_empty_report_returns_no_chart_and_structured_insight_failure() -> None:
    dataframe = pd.DataFrame(
        {
            "category": pd.Series(dtype="object"),
            "total_revenue": pd.Series(dtype="float64"),
        }
    )
    validated_plan = create_validated_analysis_plan(
        _valid_plan(),
        {"column_names": ["category", "revenue"]},
    )

    report = streamlit_app.build_v52_report(dataframe, validated_plan)

    assert report.profile.empty is True
    assert report.visualization_plan is None
    assert report.chart is None
    assert report.insight_result.error_code == "EMPTY_RESULT"

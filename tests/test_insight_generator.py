import inspect

import pandas as pd
import pytest

from src.agents import insight_generator
from src.analysis.profiler import (
    AggregatedResultSummary,
    build_aggregated_result_summary,
    profile_analysis_result,
)
from src.analysis.visualization import ChartType, VisualizationPlan
from src.llm.client import LLMResult
from src.schemas.analysis_plan import (
    create_validated_analysis_plan,
    parse_analysis_plan,
)


def _validated_plan(columns: list[str], operations: list[dict]):
    plan = parse_analysis_plan(
        {
            "version": "1.0",
            "goal": "Generate a bounded insight",
            "operations": operations,
        }
    )
    return create_validated_analysis_plan(plan, {"column_names": columns})


def _aggregate_context():
    dataframe = pd.DataFrame(
        {"category": ["A", "B"], "total_revenue": [100, 200]}
    )
    validated_plan = _validated_plan(
        ["category", "revenue"],
        [
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
        ],
    )
    return (
        profile_analysis_result(dataframe),
        build_aggregated_result_summary(dataframe, validated_plan),
        VisualizationPlan(
            ChartType.BAR,
            "category",
            "total_revenue",
            "total_revenue by category",
        ),
    )


def test_valid_insight_response_is_returned(monkeypatch) -> None:
    profile, summary, visualization = _aggregate_context()
    monkeypatch.setattr(
        insight_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(
            success=True,
            content="Category B has total revenue of 200, compared with 100 for A.",
        ),
    )

    result = insight_generator.generate_insight(profile, summary, visualization)

    assert result.success is True
    assert "200" in result.insight


def test_response_without_numeric_claims_is_allowed(monkeypatch) -> None:
    profile, summary, visualization = _aggregate_context()
    monkeypatch.setattr(
        insight_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(
            success=True,
            content="Category B leads the aggregated result.",
        ),
    )

    result = insight_generator.generate_insight(profile, summary, visualization)

    assert result.success is True


@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_or_non_text_response_fails(monkeypatch, content) -> None:
    profile, summary, visualization = _aggregate_context()
    monkeypatch.setattr(
        insight_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(success=True, content=content),
    )

    result = insight_generator.generate_insight(profile, summary, visualization)

    assert result.success is False
    assert result.error_code == "INVALID_RESPONSE"


def test_code_fenced_response_fails(monkeypatch) -> None:
    profile, summary, visualization = _aggregate_context()
    monkeypatch.setattr(
        insight_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(success=True, content="```text\nInsight\n```"),
    )

    result = insight_generator.generate_insight(profile, summary, visualization)

    assert result.success is False
    assert result.error_code == "INVALID_RESPONSE"


def test_oversized_response_fails(monkeypatch) -> None:
    profile, summary, visualization = _aggregate_context()
    monkeypatch.setattr(
        insight_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(success=True, content="A" * 2001),
    )

    result = insight_generator.generate_insight(profile, summary, visualization)

    assert result.success is False
    assert result.error_code == "INVALID_RESPONSE"


def test_timeout_is_returned_as_structured_failure(monkeypatch) -> None:
    profile, summary, visualization = _aggregate_context()
    monkeypatch.setattr(
        insight_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(
            success=False,
            error="LLM request failed (TimeoutError). Please try again.",
        ),
    )

    result = insight_generator.generate_insight(profile, summary, visualization)

    assert result.success is False
    assert result.error_code == "LLM_ERROR"
    assert "TimeoutError" in result.error


def test_hallucinated_number_is_rejected(monkeypatch) -> None:
    profile, summary, visualization = _aggregate_context()
    monkeypatch.setattr(
        insight_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(
            success=True,
            content="Category B generated 999 in revenue.",
        ),
    )

    result = insight_generator.generate_insight(profile, summary, visualization)

    assert result.success is False
    assert result.error_code == "UNSUPPORTED_NUMERIC_CLAIM"


def test_empty_result_does_not_call_llm(monkeypatch) -> None:
    dataframe = pd.DataFrame(
        {"category": pd.Series(dtype="object"), "revenue": pd.Series(dtype="float64")}
    )
    profile = profile_analysis_result(dataframe)
    summary = AggregatedResultSummary(False, 0, ("category", "revenue"), (), False)

    def fail_if_called(messages):
        raise AssertionError("LLM must not be called for an empty result")

    monkeypatch.setattr(insight_generator, "generate_chat_completion", fail_if_called)

    result = insight_generator.generate_insight(profile, summary, None)

    assert result.success is False
    assert result.error_code == "EMPTY_RESULT"


def test_non_aggregate_prompt_does_not_contain_raw_row_values(monkeypatch) -> None:
    dataframe = pd.DataFrame(
        {"customer_name": ["Sensitive Person"], "revenue": [100]}
    )
    validated_plan = _validated_plan(
        ["customer_name", "revenue"],
        [
            {
                "operation": "top_n",
                "sort_by": "revenue",
                "n": 1,
                "ascending": False,
            }
        ],
    )
    profile = profile_analysis_result(dataframe)
    summary = build_aggregated_result_summary(dataframe, validated_plan)
    captured_messages = []

    def fake_completion(messages):
        captured_messages.extend(messages)
        return LLMResult(success=True, content="The result contains a numeric metric.")

    monkeypatch.setattr(
        insight_generator,
        "generate_chat_completion",
        fake_completion,
    )

    result = insight_generator.generate_insight(profile, summary, None)

    assert result.success is True
    prompt = "\n".join(message["content"] for message in captured_messages)
    assert "Sensitive Person" not in prompt
    assert "customer_name" in prompt


def test_aggregate_prompt_contains_bounded_rows_and_visualization(monkeypatch) -> None:
    profile, summary, visualization = _aggregate_context()
    captured_messages = []

    def fake_completion(messages):
        captured_messages.extend(messages)
        return LLMResult(success=True, content="Category B leads the result.")

    monkeypatch.setattr(
        insight_generator,
        "generate_chat_completion",
        fake_completion,
    )

    result = insight_generator.generate_insight(profile, summary, visualization)

    assert result.success is True
    prompt = "\n".join(message["content"] for message in captured_messages)
    assert '"category":"B"' in prompt
    assert '"chart_type":"bar"' in prompt
    assert "Do not invent" in prompt
    assert "untrusted data" in prompt


@pytest.mark.parametrize(
    ("profile", "summary", "visualization", "error_code"),
    [
        ({}, AggregatedResultSummary(False, 1, (), (), False), None, "INVALID_PROFILE"),
        (profile_analysis_result(pd.DataFrame({"value": [1]})), {}, None, "INVALID_SUMMARY"),
        (
            profile_analysis_result(pd.DataFrame({"value": [1]})),
            AggregatedResultSummary(False, 1, ("value",), (), False),
            {},
            "INVALID_VISUALIZATION",
        ),
    ],
)
def test_invalid_metadata_types_fail(profile, summary, visualization, error_code) -> None:
    result = insight_generator.generate_insight(profile, summary, visualization)

    assert result.success is False
    assert result.error_code == error_code


def test_metadata_row_count_mismatch_fails() -> None:
    profile = profile_analysis_result(pd.DataFrame({"value": [1]}))
    summary = AggregatedResultSummary(False, 2, ("value",), (), False)

    result = insight_generator.generate_insight(profile, summary, None)

    assert result.success is False
    assert result.error_code == "METADATA_MISMATCH"


def test_insight_generator_signature_cannot_receive_dataframe() -> None:
    parameters = inspect.signature(insight_generator.generate_insight).parameters

    assert "dataframe" not in parameters

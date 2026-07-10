import json

import pandas as pd
import pytest

from src.agents import plan_generator
from src.data.schema import build_schema_summary
from src.llm.client import LLMResult


@pytest.fixture
def schema_summary() -> dict:
    dataframe = pd.DataFrame(
        {
            "category": ["Computer"],
            "region": ["North"],
            "revenue": [100],
        }
    )
    return build_schema_summary(dataframe)


def _valid_payload() -> dict:
    return {
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
                "n": 5,
                "ascending": False,
            },
        ],
    }


def test_valid_json_plan(schema_summary, monkeypatch) -> None:
    monkeypatch.setattr(
        plan_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(
            success=True,
            content=json.dumps(_valid_payload()),
        ),
    )

    result = plan_generator.generate_structured_plan(
        schema_summary,
        "Which categories have the highest revenue?",
    )

    assert result.success is True
    assert result.plan is not None
    assert result.plan.goal == "Rank categories by revenue"
    assert result.validation_errors == ()


def test_json_fence_is_supported(schema_summary, monkeypatch) -> None:
    fenced_content = f"```json\n{json.dumps(_valid_payload())}\n```"
    monkeypatch.setattr(
        plan_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(success=True, content=fenced_content),
    )

    result = plan_generator.generate_structured_plan(schema_summary, "Rank revenue")

    assert result.success is True
    assert result.plan is not None


def test_empty_response_fails(schema_summary, monkeypatch) -> None:
    monkeypatch.setattr(
        plan_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(success=True, content=""),
    )

    result = plan_generator.generate_structured_plan(schema_summary, "Rank revenue")

    assert result.success is False
    assert "JSON is invalid" in result.error


def test_invalid_json_fails(schema_summary, monkeypatch) -> None:
    monkeypatch.setattr(
        plan_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(success=True, content="{not valid json}"),
    )

    result = plan_generator.generate_structured_plan(schema_summary, "Rank revenue")

    assert result.success is False
    assert "JSON is invalid" in result.error


def test_timeout_is_returned_without_retry(schema_summary, monkeypatch) -> None:
    monkeypatch.setattr(
        plan_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(
            success=False,
            error="LLM request failed (TimeoutError). Please try again.",
        ),
    )

    result = plan_generator.generate_structured_plan(schema_summary, "Rank revenue")

    assert result.success is False
    assert "TimeoutError" in result.error


def test_prompt_requires_json_and_forbids_code(schema_summary, monkeypatch) -> None:
    captured_messages = []

    def fake_completion(messages):
        captured_messages.extend(messages)
        return LLMResult(success=True, content=json.dumps(_valid_payload()))

    monkeypatch.setattr(plan_generator, "generate_chat_completion", fake_completion)

    result = plan_generator.generate_structured_plan(schema_summary, "Rank revenue")

    assert result.success is True
    combined_prompt = "\n".join(message["content"] for message in captured_messages)
    assert "Return exactly one JSON object" in combined_prompt
    assert "Do not return" in combined_prompt
    assert "Python" in combined_prompt
    assert "SQL" in combined_prompt


def test_invalid_plan_is_not_auto_repaired(schema_summary, monkeypatch) -> None:
    payload = _valid_payload()
    payload["unexpected"] = "do not remove me"
    monkeypatch.setattr(
        plan_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(success=True, content=json.dumps(payload)),
    )

    result = plan_generator.generate_structured_plan(schema_summary, "Rank revenue")

    assert result.success is False
    assert "unknown fields" in result.error

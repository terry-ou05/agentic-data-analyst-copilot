import json

import pytest

from src.agents import capability_guard
from src.agents.capability_guard import Capability
from src.llm.client import LLMResult
from src.schemas.analysis_plan import parse_analysis_plan


def _ranking_plan():
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


def _assessment(capability: str, matches: bool = True) -> str:
    return json.dumps(
        {
            "capability": capability,
            "plan_matches_intent": matches,
            "reason": "Mocked semantic assessment.",
        }
    )


def _plan_with_operations(operations: list[dict]):
    return parse_analysis_plan(
        {
            "version": "1.0",
            "goal": "Mock capability plan",
            "operations": operations,
        }
    )


def test_supported_ranking_request_with_matching_plan_is_allowed(monkeypatch) -> None:
    captured_messages = []
    monkeypatch.setattr(
        capability_guard,
        "generate_chat_completion",
        lambda messages: (
            captured_messages.extend(messages)
            or LLMResult(success=True, content=_assessment("ranking"))
        ),
    )

    result = capability_guard.check_capability_boundary(
        "Which category generates the highest revenue?",
        _ranking_plan(),
    )

    assert result.allowed is True
    assert result.capability is Capability.RANKING
    assert result.plan_matches_intent is True
    prompt = "\n".join(message["content"] for message in captured_messages)
    assert "Classify the user's intended capability semantically" in prompt
    assert "PLAN_JSON" in prompt
    assert "forecasting" in prompt


@pytest.mark.parametrize(
    ("capability", "operations"),
    [
        (
            "aggregation",
            [
                {
                    "operation": "aggregate",
                    "metrics": [
                        {
                            "column": "revenue",
                            "function": "sum",
                            "alias": "total_revenue",
                        }
                    ],
                }
            ],
        ),
        (
            "filtering",
            [
                {
                    "operation": "filter",
                    "column": "region",
                    "operator": "eq",
                    "value": "North",
                }
            ],
        ),
        (
            "grouping_analysis",
            [
                {"operation": "groupby", "columns": ["region"]},
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
        ),
        (
            "summary_analysis",
            [
                {
                    "operation": "aggregate",
                    "metrics": [
                        {
                            "column": "revenue",
                            "function": "mean",
                            "alias": "average_revenue",
                        }
                    ],
                }
            ],
        ),
    ],
)
def test_supported_capabilities_with_required_operations_are_allowed(
    monkeypatch,
    capability,
    operations,
) -> None:
    monkeypatch.setattr(
        capability_guard,
        "generate_chat_completion",
        lambda messages: LLMResult(success=True, content=_assessment(capability)),
    )

    result = capability_guard.check_capability_boundary(
        "Run a supported business analysis.",
        _plan_with_operations(operations),
    )

    assert result.allowed is True
    assert result.capability is Capability(capability)


@pytest.mark.parametrize(
    ("question", "capability"),
    [
        ("Predict next year's revenue.", "forecasting"),
        ("Predict customer churn.", "prediction"),
        ("Update all revenue values.", "data_modification"),
        ("Delete all database records.", "deletion"),
        ("Insert a new order.", "insertion"),
        ("Train a machine learning model.", "machine_learning_training"),
    ],
)
def test_unsupported_capabilities_are_rejected(monkeypatch, question, capability) -> None:
    monkeypatch.setattr(
        capability_guard,
        "generate_chat_completion",
        lambda messages: LLMResult(success=True, content=_assessment(capability)),
    )

    result = capability_guard.check_capability_boundary(question, _ranking_plan())

    assert result.allowed is False
    assert result.capability is Capability(capability)
    assert "outside the supported analysis capabilities" in result.message
    assert f"Unsupported capability: {capability}." in result.errors


def test_semantic_plan_mismatch_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        capability_guard,
        "generate_chat_completion",
        lambda messages: LLMResult(
            success=True,
            content=_assessment("aggregation", matches=False),
        ),
    )

    result = capability_guard.check_capability_boundary(
        "Show total revenue.",
        _ranking_plan(),
    )

    assert result.allowed is False
    assert result.capability is Capability.AGGREGATION
    assert result.plan_matches_intent is False
    assert "does not match your request" in result.message


def test_ranking_requires_top_n_even_when_assessment_claims_match(monkeypatch) -> None:
    aggregate_only_plan = parse_analysis_plan(
        {
            "version": "1.0",
            "goal": "Rank categories",
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
            ],
        }
    )
    monkeypatch.setattr(
        capability_guard,
        "generate_chat_completion",
        lambda messages: LLMResult(success=True, content=_assessment("ranking")),
    )

    result = capability_guard.check_capability_boundary(
        "Which category is highest by revenue?",
        aggregate_only_plan,
    )

    assert result.allowed is False
    assert result.capability is Capability.RANKING
    assert result.errors == ("Missing required operations: top_n.",)


def test_invalid_guard_response_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        capability_guard,
        "generate_chat_completion",
        lambda messages: LLMResult(
            success=True,
            content=json.dumps(
                {
                    "capability": "unsupported_value",
                    "plan_matches_intent": True,
                    "reason": "Invalid capability.",
                }
            ),
        ),
    )

    result = capability_guard.check_capability_boundary(
        "Which category is highest by revenue?",
        _ranking_plan(),
    )

    assert result.allowed is False
    assert result.capability is None
    assert "Unable to verify" in result.message


def test_guard_timeout_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        capability_guard,
        "generate_chat_completion",
        lambda messages: LLMResult(
            success=False,
            error="LLM request failed (TimeoutError). Please try again.",
        ),
    )

    result = capability_guard.check_capability_boundary(
        "Which category is highest by revenue?",
        _ranking_plan(),
    )

    assert result.allowed is False
    assert result.capability is None
    assert result.errors == ("Capability assessment request failed.",)

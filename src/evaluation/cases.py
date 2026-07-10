from dataclasses import dataclass
from typing import Any, Mapping

from src.analysis.visualization import ChartType


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    question: str
    plan_payload: Mapping[str, Any]
    expected_top_column: str | None = None
    expected_top_value: str | None = None
    expected_chart_type: ChartType | None = None


@dataclass(frozen=True)
class MetricRequirement:
    """A required source metric, independent of the generated alias."""

    column: str
    function: str


@dataclass(frozen=True)
class SemanticEvaluationCase:
    """One business intent with natural-language variants and required capability."""

    intent_name: str
    description: str
    question_variations: tuple[str, ...]
    plan_payloads: tuple[Mapping[str, Any], ...]
    expected_operations: tuple[str, ...]
    expected_columns: tuple[str, ...]
    expected_metrics: tuple[MetricRequirement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.intent_name, str) or not self.intent_name.strip():
            raise ValueError("intent_name must be a non-empty string.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string.")
        if not self.question_variations or any(
            not isinstance(question, str) or not question.strip()
            for question in self.question_variations
        ):
            raise ValueError("question_variations must contain non-empty strings.")
        if len(self.question_variations) != len(self.plan_payloads):
            raise ValueError(
                "question_variations and plan_payloads must have the same length."
            )
        if not self.expected_operations:
            raise ValueError("expected_operations must not be empty.")
        if any(
            not isinstance(column, str) or not column.strip()
            for column in self.expected_columns
        ):
            raise ValueError("expected_columns must contain non-empty strings.")


def _plan(goal: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
    return {"version": "1.0", "goal": goal, "operations": operations}


def default_evaluation_cases() -> tuple[EvaluationCase, ...]:
    """Return deterministic business questions and expected workflow outcomes."""
    return (
        EvaluationCase(
            name="top_category_revenue",
            question="Which category has the highest total revenue?",
            plan_payload=_plan(
                "Rank categories by total revenue.",
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
                    {
                        "operation": "top_n",
                        "sort_by": "total_revenue",
                        "n": 1,
                        "ascending": False,
                    },
                ],
            ),
            expected_top_column="category",
            expected_top_value="Computer",
            expected_chart_type=ChartType.BAR,
        ),
        EvaluationCase(
            name="online_region_revenue",
            question="Which region has the highest online revenue?",
            plan_payload=_plan(
                "Rank regions by online revenue.",
                [
                    {
                        "operation": "filter",
                        "column": "channel",
                        "operator": "eq",
                        "value": "Online",
                    },
                    {"operation": "groupby", "columns": ["region"]},
                    {
                        "operation": "aggregate",
                        "metrics": [
                            {
                                "column": "revenue",
                                "function": "sum",
                                "alias": "online_revenue",
                            }
                        ],
                    },
                    {
                        "operation": "top_n",
                        "sort_by": "online_revenue",
                        "n": 1,
                        "ascending": False,
                    },
                ],
            ),
            expected_top_column="region",
            expected_top_value="North",
            expected_chart_type=ChartType.BAR,
        ),
        EvaluationCase(
            name="highest_average_quantity_category",
            question="Which category has the highest average quantity?",
            plan_payload=_plan(
                "Rank categories by average quantity.",
                [
                    {"operation": "groupby", "columns": ["category"]},
                    {
                        "operation": "aggregate",
                        "metrics": [
                            {
                                "column": "quantity",
                                "function": "mean",
                                "alias": "average_quantity",
                            }
                        ],
                    },
                    {
                        "operation": "top_n",
                        "sort_by": "average_quantity",
                        "n": 1,
                        "ascending": False,
                    },
                ],
            ),
            expected_top_column="category",
            expected_top_value="Accessory",
            expected_chart_type=ChartType.BAR,
        ),
    )


def default_semantic_evaluation_cases() -> tuple[SemanticEvaluationCase, ...]:
    """Return offline semantic-robustness cases without production question rules."""
    return (
        SemanticEvaluationCase(
            intent_name="highest_revenue_category",
            description="Identify the category with the greatest total revenue.",
            question_variations=(
                "Which category has the highest revenue?",
                "What product category makes the most money?",
                "Find the top performing category by sales.",
            ),
            plan_payloads=(
                _plan(
                    "Rank categories by total revenue.",
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
                        {
                            "operation": "top_n",
                            "sort_by": "total_revenue",
                            "n": 1,
                            "ascending": False,
                        },
                    ],
                ),
                _plan(
                    "Find the category that generates the most revenue.",
                    [
                        {"operation": "groupby", "columns": ["category"]},
                        {
                            "operation": "aggregate",
                            "metrics": [
                                {
                                    "column": "revenue",
                                    "function": "sum",
                                    "alias": "category_sales",
                                }
                            ],
                        },
                        {
                            "operation": "top_n",
                            "sort_by": "category_sales",
                            "n": 1,
                            "ascending": False,
                        },
                    ],
                ),
                _plan(
                    "Return the leading category by revenue.",
                    [
                        {"operation": "groupby", "columns": ["category"]},
                        {
                            "operation": "aggregate",
                            "metrics": [
                                {
                                    "column": "revenue",
                                    "function": "sum",
                                    "alias": "revenue_total",
                                }
                            ],
                        },
                        {
                            "operation": "top_n",
                            "sort_by": "revenue_total",
                            "n": 1,
                            "ascending": False,
                        },
                    ],
                ),
            ),
            expected_operations=("groupby", "aggregate", "top_n"),
            expected_columns=("category", "revenue"),
            expected_metrics=(MetricRequirement("revenue", "sum"),),
        ),
        SemanticEvaluationCase(
            intent_name="highest_online_revenue_region",
            description="Identify the region with the greatest online revenue.",
            question_variations=(
                "Which region has the highest online revenue?",
                "For online sales, what region earns the most?",
                "Show the best region for revenue from the online channel.",
            ),
            plan_payloads=(
                _plan(
                    "Rank online revenue by region.",
                    [
                        {
                            "operation": "filter",
                            "column": "channel",
                            "operator": "eq",
                            "value": "Online",
                        },
                        {"operation": "groupby", "columns": ["region"]},
                        {
                            "operation": "aggregate",
                            "metrics": [
                                {
                                    "column": "revenue",
                                    "function": "sum",
                                    "alias": "online_revenue",
                                }
                            ],
                        },
                        {
                            "operation": "top_n",
                            "sort_by": "online_revenue",
                            "n": 1,
                            "ascending": False,
                        },
                    ],
                ),
                _plan(
                    "Find the top online region by revenue.",
                    [
                        {
                            "operation": "filter",
                            "column": "channel",
                            "operator": "eq",
                            "value": "Online",
                        },
                        {"operation": "groupby", "columns": ["region"]},
                        {
                            "operation": "aggregate",
                            "metrics": [
                                {
                                    "column": "revenue",
                                    "function": "sum",
                                    "alias": "regional_online_sales",
                                }
                            ],
                        },
                        {
                            "operation": "top_n",
                            "sort_by": "regional_online_sales",
                            "n": 1,
                            "ascending": False,
                        },
                    ],
                ),
                _plan(
                    "Return the online revenue leader by region.",
                    [
                        {
                            "operation": "filter",
                            "column": "channel",
                            "operator": "eq",
                            "value": "Online",
                        },
                        {"operation": "groupby", "columns": ["region"]},
                        {
                            "operation": "aggregate",
                            "metrics": [
                                {
                                    "column": "revenue",
                                    "function": "sum",
                                    "alias": "online_revenue_total",
                                }
                            ],
                        },
                        {
                            "operation": "top_n",
                            "sort_by": "online_revenue_total",
                            "n": 1,
                            "ascending": False,
                        },
                    ],
                ),
            ),
            expected_operations=("filter", "groupby", "aggregate", "top_n"),
            expected_columns=("channel", "region", "revenue"),
            expected_metrics=(MetricRequirement("revenue", "sum"),),
        ),
        SemanticEvaluationCase(
            intent_name="highest_average_quantity_category",
            description="Identify the category with the greatest average quantity.",
            question_variations=(
                "Which category has the highest average quantity?",
                "What product type sells the most units on average?",
                "Find the category with the best average order quantity.",
            ),
            plan_payloads=(
                _plan(
                    "Rank categories by average quantity.",
                    [
                        {"operation": "groupby", "columns": ["category"]},
                        {
                            "operation": "aggregate",
                            "metrics": [
                                {
                                    "column": "quantity",
                                    "function": "mean",
                                    "alias": "average_quantity",
                                }
                            ],
                        },
                        {
                            "operation": "top_n",
                            "sort_by": "average_quantity",
                            "n": 1,
                            "ascending": False,
                        },
                    ],
                ),
                _plan(
                    "Find the product category with the highest mean quantity.",
                    [
                        {"operation": "groupby", "columns": ["category"]},
                        {
                            "operation": "aggregate",
                            "metrics": [
                                {
                                    "column": "quantity",
                                    "function": "mean",
                                    "alias": "mean_units",
                                }
                            ],
                        },
                        {
                            "operation": "top_n",
                            "sort_by": "mean_units",
                            "n": 1,
                            "ascending": False,
                        },
                    ],
                ),
                _plan(
                    "Return the category leading in average quantity.",
                    [
                        {"operation": "groupby", "columns": ["category"]},
                        {
                            "operation": "aggregate",
                            "metrics": [
                                {
                                    "column": "quantity",
                                    "function": "mean",
                                    "alias": "quantity_average",
                                }
                            ],
                        },
                        {
                            "operation": "top_n",
                            "sort_by": "quantity_average",
                            "n": 1,
                            "ascending": False,
                        },
                    ],
                ),
            ),
            expected_operations=("groupby", "aggregate", "top_n"),
            expected_columns=("category", "quantity"),
            expected_metrics=(MetricRequirement("quantity", "mean"),),
        ),
    )

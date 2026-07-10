import pytest

from src.schemas.analysis_plan import (
    OperationType,
    PlanParseError,
    parse_analysis_plan,
    validate_analysis_plan,
)


SCHEMA_SUMMARY = {
    "column_names": ["category", "region", "revenue", "channel"],
}


def _plan(operations):
    return {
        "version": "1.0",
        "goal": "Test structured analysis plan",
        "operations": operations,
    }


@pytest.mark.parametrize(
    "operations",
    [
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
        [
            {
                "operation": "filter",
                "column": "channel",
                "operator": "eq",
                "value": "Online",
            }
        ],
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
        [
            {
                "operation": "top_n",
                "sort_by": "revenue",
                "n": 5,
                "ascending": False,
            }
        ],
    ],
)
def test_valid_operation_plans(operations) -> None:
    plan = parse_analysis_plan(_plan(operations))
    result = validate_analysis_plan(plan, SCHEMA_SUMMARY)

    assert result.valid is True
    assert result.errors == ()


def test_operation_type_enum_contains_only_supported_operations() -> None:
    assert {operation.value for operation in OperationType} == {
        "filter",
        "groupby",
        "aggregate",
        "top_n",
    }


def test_unknown_operation_fails() -> None:
    with pytest.raises(PlanParseError, match="Unsupported operation"):
        parse_analysis_plan(_plan([{"operation": "join", "column": "region"}]))


def test_missing_column_fails_validation() -> None:
    plan = parse_analysis_plan(
        _plan(
            [
                {
                    "operation": "filter",
                    "column": "missing_column",
                    "operator": "eq",
                    "value": "x",
                }
            ]
        )
    )

    result = validate_analysis_plan(plan, SCHEMA_SUMMARY)

    assert result.valid is False
    assert "Filter column does not exist: missing_column" in result.errors


@pytest.mark.parametrize(
    "payload",
    [
        {
            "version": "1.0",
            "goal": "Wrong operations type",
            "operations": "filter",
        },
        _plan(
            [
                {
                    "operation": "groupby",
                    "columns": "category",
                }
            ]
        ),
        _plan(
            [
                {
                    "operation": "top_n",
                    "sort_by": "revenue",
                    "n": "5",
                    "ascending": False,
                }
            ]
        ),
    ],
)
def test_wrong_parameter_types_fail(payload) -> None:
    with pytest.raises(PlanParseError):
        parse_analysis_plan(payload)


def test_invalid_aggregation_function_fails() -> None:
    plan = parse_analysis_plan(
        _plan(
            [
                {
                    "operation": "aggregate",
                    "metrics": [
                        {
                            "column": "revenue",
                            "function": "median",
                            "alias": "median_revenue",
                        }
                    ],
                }
            ]
        )
    )

    result = validate_analysis_plan(plan, SCHEMA_SUMMARY)

    assert result.valid is False
    assert "Unsupported aggregate function: median" in result.errors


def test_invalid_filter_operator_fails() -> None:
    plan = parse_analysis_plan(
        _plan(
            [
                {
                    "operation": "filter",
                    "column": "region",
                    "operator": "contains",
                    "value": "North",
                }
            ]
        )
    )

    result = validate_analysis_plan(plan, SCHEMA_SUMMARY)

    assert result.valid is False
    assert "Unsupported filter operator: contains" in result.errors


@pytest.mark.parametrize("n", [0, 101])
def test_top_n_out_of_range_fails(n) -> None:
    plan = parse_analysis_plan(
        _plan(
            [
                {
                    "operation": "top_n",
                    "sort_by": "revenue",
                    "n": n,
                    "ascending": False,
                }
            ]
        )
    )

    result = validate_analysis_plan(plan, SCHEMA_SUMMARY)

    assert result.valid is False
    assert "top_n n must be between 1 and 100." in result.errors


@pytest.mark.parametrize(
    "payload",
    [
        {
            "version": "1.0",
            "goal": "Unknown top-level field",
            "operations": [],
            "unexpected": True,
        },
        _plan(
            [
                {
                    "operation": "filter",
                    "column": "region",
                    "operator": "eq",
                    "value": "North",
                    "unexpected": True,
                }
            ]
        ),
    ],
)
def test_unknown_json_fields_fail(payload) -> None:
    with pytest.raises(PlanParseError, match="unknown fields"):
        parse_analysis_plan(payload)


def test_invalid_operation_order_fails() -> None:
    plan = parse_analysis_plan(
        _plan(
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
                },
                {
                    "operation": "filter",
                    "column": "region",
                    "operator": "eq",
                    "value": "North",
                },
            ]
        )
    )

    result = validate_analysis_plan(plan, SCHEMA_SUMMARY)

    assert result.valid is False
    assert "operations[1] is out of order." in result.errors


def test_groupby_without_aggregate_fails() -> None:
    plan = parse_analysis_plan(
        _plan([{"operation": "groupby", "columns": ["category"]}])
    )

    result = validate_analysis_plan(plan, SCHEMA_SUMMARY)

    assert result.valid is False
    assert "A groupby operation must be followed by aggregate." in result.errors

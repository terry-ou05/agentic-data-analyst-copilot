import pandas as pd
import pytest

from src.analysis.operations import (
    OperationExecutionError,
    apply_aggregate,
    apply_filter,
    apply_groupby,
    apply_top_n,
)
from src.schemas.analysis_plan import (
    create_validated_analysis_plan,
    parse_analysis_plan,
)


@pytest.fixture
def dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["A", "B", "A", None],
            "region": ["North", "South", "South", "North"],
            "revenue": [100, 200, 50, 150],
            "profit": [10.0, 30.0, 5.0, None],
        }
    )


def _validated_operations(dataframe: pd.DataFrame, operations: list[dict]):
    plan = parse_analysis_plan(
        {
            "version": "1.0",
            "goal": "Test trusted operations",
            "operations": operations,
        }
    )
    validated = create_validated_analysis_plan(
        plan,
        {"column_names": list(dataframe.columns)},
    )
    return validated.operations


@pytest.mark.parametrize(
    ("operator", "value", "expected_indices"),
    [
        ("eq", "A", [0, 2]),
        ("ne", "A", [1, 3]),
        ("gt", 100, [1, 3]),
        ("gte", 100, [0, 1, 3]),
        ("lt", 100, [2]),
        ("lte", 100, [0, 2]),
        ("in", ["A", "B"], [0, 1, 2]),
        ("not_in", ["A"], [1, 3]),
        ("is_null", None, [3]),
        ("not_null", None, [0, 1, 2]),
    ],
)
def test_allowlisted_filters(
    dataframe: pd.DataFrame,
    operator: str,
    value,
    expected_indices: list[int],
) -> None:
    column = "revenue" if operator in {"gt", "gte", "lt", "lte"} else "category"
    operation = _validated_operations(
        dataframe,
        [
            {
                "operation": "filter",
                "column": column,
                "operator": operator,
                "value": value,
            }
        ],
    )[0]

    result = apply_filter(dataframe, operation)

    assert result.index.tolist() == expected_indices


def test_groupby_then_aggregate_matches_pandas(dataframe: pd.DataFrame) -> None:
    operations = _validated_operations(
        dataframe,
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

    grouped_input = apply_groupby(dataframe, operations[0])
    result = apply_aggregate(grouped_input, operations[1], ["category"])
    expected = dataframe.groupby(
        ["category"], dropna=False, as_index=False
    ).agg(total_revenue=("revenue", "sum"))

    pd.testing.assert_frame_equal(result, expected)


def test_all_aggregate_functions_are_supported(dataframe: pd.DataFrame) -> None:
    operation = _validated_operations(
        dataframe,
        [
            {
                "operation": "aggregate",
                "metrics": [
                    {"column": "revenue", "function": "sum", "alias": "sum_value"},
                    {"column": "revenue", "function": "mean", "alias": "mean_value"},
                    {"column": "revenue", "function": "min", "alias": "min_value"},
                    {"column": "revenue", "function": "max", "alias": "max_value"},
                    {"column": "profit", "function": "count", "alias": "count_value"},
                ],
            }
        ],
    )[0]

    result = apply_aggregate(dataframe, operation)

    assert result.to_dict("records") == [
        {
            "sum_value": 500,
            "mean_value": 125.0,
            "min_value": 50,
            "max_value": 200,
            "count_value": 3,
        }
    ]


def test_top_n_matches_pandas(dataframe: pd.DataFrame) -> None:
    operation = _validated_operations(
        dataframe,
        [
            {
                "operation": "top_n",
                "sort_by": "revenue",
                "n": 2,
                "ascending": False,
            }
        ],
    )[0]

    result = apply_top_n(dataframe, operation)
    expected = dataframe.sort_values("revenue", ascending=False).head(2).copy()

    pd.testing.assert_frame_equal(result, expected)


def test_chinese_and_special_column_names_are_supported() -> None:
    dataframe = pd.DataFrame(
        {
            "产品-类型": ["电脑", "手机", "电脑"],
            "订单 金额": [100, 200, 50],
        }
    )
    operations = _validated_operations(
        dataframe,
        [
            {"operation": "groupby", "columns": ["产品-类型"]},
            {
                "operation": "aggregate",
                "metrics": [
                    {
                        "column": "订单 金额",
                        "function": "sum",
                        "alias": "总金额",
                    }
                ],
            },
        ],
    )

    result = apply_aggregate(
        apply_groupby(dataframe, operations[0]),
        operations[1],
        ["产品-类型"],
    )

    assert result.set_index("产品-类型")["总金额"].to_dict() == {
        "电脑": 150,
        "手机": 200,
    }


def test_operations_do_not_modify_input_dataframe(dataframe: pd.DataFrame) -> None:
    original = dataframe.copy(deep=True)
    operation = _validated_operations(
        dataframe,
        [
            {
                "operation": "filter",
                "column": "category",
                "operator": "eq",
                "value": "A",
            }
        ],
    )[0]

    result = apply_filter(dataframe, operation)
    result.loc[:, "revenue"] = 0

    pd.testing.assert_frame_equal(dataframe, original)


def test_operation_rejects_raw_dict(dataframe: pd.DataFrame) -> None:
    with pytest.raises(OperationExecutionError, match="validated filter operation"):
        apply_filter(dataframe, {})

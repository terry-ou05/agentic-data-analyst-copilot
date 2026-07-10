import pandas as pd
import pytest

from src.analysis.profiler import (
    build_aggregated_result_summary,
    profile_analysis_result,
)
from src.schemas.analysis_plan import (
    create_validated_analysis_plan,
    parse_analysis_plan,
)


def _validated_plan(columns: list[str], operations: list[dict]):
    plan = parse_analysis_plan(
        {
            "version": "1.0",
            "goal": "Profile an analysis result",
            "operations": operations,
        }
    )
    return create_validated_analysis_plan(plan, {"column_names": columns})


def _aggregate_plan():
    return _validated_plan(
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


def test_numeric_column_profile_contains_safe_statistics() -> None:
    profile = profile_analysis_result(pd.DataFrame({"revenue": [10, 20, 30]}))

    assert profile.columns[0].data_type == "numeric"
    assert profile.numeric_summary[0].to_dict() == {
        "count": 3,
        "sum": 60,
        "mean": 20.0,
        "min": 10,
        "max": 30,
    }


def test_all_missing_numeric_column_uses_null_statistics() -> None:
    dataframe = pd.DataFrame({"revenue": pd.Series([None, None], dtype="float64")})

    profile = profile_analysis_result(dataframe)

    assert profile.numeric_summary[0].to_dict() == {
        "count": 0,
        "sum": None,
        "mean": None,
        "min": None,
        "max": None,
    }


def test_categorical_column_is_identified() -> None:
    profile = profile_analysis_result(pd.DataFrame({"category": ["A", "B", "A"]}))

    assert profile.columns[0].data_type == "categorical"
    assert profile.columns[0].unique_values == 2
    assert profile.numeric_summary == ()


@pytest.mark.parametrize(
    ("series", "expected_type"),
    [
        (pd.Series([True, False]), "boolean"),
        (pd.Series(pd.to_datetime(["2025-01-01", "2025-01-02"])), "datetime"),
    ],
)
def test_boolean_and_datetime_columns_are_identified(series, expected_type) -> None:
    profile = profile_analysis_result(pd.DataFrame({"value": series}))

    assert profile.columns[0].data_type == expected_type


def test_missing_values_are_reported_per_column() -> None:
    dataframe = pd.DataFrame({"category": ["A", None], "revenue": [10, None]})

    profile_dict = profile_analysis_result(dataframe).to_dict()

    assert profile_dict["statistics"]["missing_values"] == {
        "category": 1,
        "revenue": 1,
    }


def test_empty_dataframe_is_supported() -> None:
    dataframe = pd.DataFrame(
        {
            "category": pd.Series(dtype="object"),
            "revenue": pd.Series(dtype="float64"),
        }
    )

    profile = profile_analysis_result(dataframe)

    assert profile.empty is True
    assert profile.rows == 0
    assert len(profile.columns) == 2
    assert profile.numeric_summary[0].count == 0


def test_unicode_and_special_character_columns_are_preserved() -> None:
    dataframe = pd.DataFrame({"产品-类型": ["电脑"], "订单 金额": [100]})

    profile = profile_analysis_result(dataframe)

    assert [column.name for column in profile.columns] == ["产品-类型", "订单 金额"]


def test_date_named_string_column_is_safely_identified_as_datetime() -> None:
    dataframe = pd.DataFrame({"order_date": ["2025-01-01", "2025-01-02"]})

    profile = profile_analysis_result(dataframe)

    assert profile.columns[0].data_type == "datetime"


def test_date_like_category_without_time_name_remains_categorical() -> None:
    dataframe = pd.DataFrame({"category": ["2025-01-01", "2025-01-02"]})

    profile = profile_analysis_result(dataframe)

    assert profile.columns[0].data_type == "categorical"


def test_profiler_does_not_modify_dataframe() -> None:
    dataframe = pd.DataFrame({"category": ["A", None], "revenue": [10, 20]})
    original = dataframe.copy(deep=True)

    profile_analysis_result(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_profiler_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        profile_analysis_result([{"revenue": 10}])


def test_non_aggregate_result_never_exposes_rows() -> None:
    dataframe = pd.DataFrame(
        {"customer_name": ["Sensitive Person"], "revenue": [100]}
    )
    plan = _validated_plan(
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

    summary = build_aggregated_result_summary(dataframe, plan)

    assert summary.is_aggregated is False
    assert summary.rows == ()
    assert "Sensitive Person" not in str(summary.to_dict())


def test_aggregate_result_exposes_only_bounded_result_rows() -> None:
    dataframe = pd.DataFrame(
        {"category": ["A", "B"], "total_revenue": [150, 200]}
    )

    summary = build_aggregated_result_summary(dataframe, _aggregate_plan())

    assert summary.is_aggregated is True
    assert summary.to_dict()["rows"] == [
        {"category": "A", "total_revenue": 150},
        {"category": "B", "total_revenue": 200},
    ]


def test_aggregate_summary_marks_row_truncation() -> None:
    dataframe = pd.DataFrame(
        {"category": ["A", "B", "C"], "total_revenue": [10, 20, 30]}
    )

    summary = build_aggregated_result_summary(
        dataframe,
        _aggregate_plan(),
        max_rows=2,
    )

    assert len(summary.rows) == 2
    assert summary.truncated is True


def test_aggregate_summary_marks_column_truncation() -> None:
    dataframe = pd.DataFrame(
        {"category": ["A"], "total_revenue": [10], "extra": [1]}
    )

    summary = build_aggregated_result_summary(
        dataframe,
        _aggregate_plan(),
        max_columns=2,
    )

    assert summary.columns == ("category", "total_revenue")
    assert summary.truncated is True


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_rows", 0), ("max_rows", True), ("max_columns", 0)],
)
def test_aggregate_summary_rejects_invalid_limits(field, value) -> None:
    kwargs = {field: value}

    with pytest.raises(ValueError, match="positive integer"):
        build_aggregated_result_summary(
            pd.DataFrame({"category": ["A"], "total_revenue": [10]}),
            _aggregate_plan(),
            **kwargs,
        )

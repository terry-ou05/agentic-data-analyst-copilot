import pandas as pd
import pytest

from src.analysis.profiler import profile_analysis_result
from src.analysis.visualization import ChartType, plan_visualization
from src.schemas.analysis_plan import (
    create_validated_analysis_plan,
    parse_analysis_plan,
)


def _validated_plan(columns: list[str], operations: list[dict]):
    plan = parse_analysis_plan(
        {
            "version": "1.0",
            "goal": "Choose a deterministic visualization",
            "operations": operations,
        }
    )
    return create_validated_analysis_plan(plan, {"column_names": columns})


def _aggregate_plan(group_column: str, metric_column: str, alias: str):
    return _validated_plan(
        [group_column, metric_column],
        [
            {"operation": "groupby", "columns": [group_column]},
            {
                "operation": "aggregate",
                "metrics": [
                    {
                        "column": metric_column,
                        "function": "sum",
                        "alias": alias,
                    }
                ],
            },
        ],
    )


def test_categorical_metric_result_uses_bar_chart() -> None:
    dataframe = pd.DataFrame(
        {"category": ["A", "B"], "total_revenue": [100, 200]}
    )

    plan = plan_visualization(
        profile_analysis_result(dataframe),
        _aggregate_plan("category", "revenue", "total_revenue"),
    )

    assert plan.chart_type is ChartType.BAR
    assert plan.x == "category"
    assert plan.y == "total_revenue"


def test_datetime_metric_result_uses_line_chart() -> None:
    dataframe = pd.DataFrame(
        {
            "order_date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "total_revenue": [100, 200],
        }
    )

    plan = plan_visualization(
        profile_analysis_result(dataframe),
        _aggregate_plan("order_date", "revenue", "total_revenue"),
    )

    assert plan.chart_type is ChartType.LINE
    assert plan.x == "order_date"
    assert plan.y == "total_revenue"


def test_date_string_metric_result_uses_line_chart() -> None:
    dataframe = pd.DataFrame(
        {
            "order_date": ["2025-01-01", "2025-01-02"],
            "total_revenue": [100, 200],
        }
    )

    plan = plan_visualization(
        profile_analysis_result(dataframe),
        _aggregate_plan("order_date", "revenue", "total_revenue"),
    )

    assert plan.chart_type is ChartType.LINE
    assert plan.x == "order_date"


def test_numeric_pair_result_uses_scatter_chart() -> None:
    dataframe = pd.DataFrame({"revenue": [10, 20], "profit": [2, 5]})
    plan_contract = _validated_plan(
        ["revenue", "profit"],
        [
            {
                "operation": "top_n",
                "sort_by": "revenue",
                "n": 2,
                "ascending": False,
            }
        ],
    )

    plan = plan_visualization(profile_analysis_result(dataframe), plan_contract)

    assert plan.chart_type is ChartType.SCATTER
    assert plan.x == "revenue"
    assert plan.y == "profit"


def test_aggregate_alias_is_preferred_as_metric() -> None:
    dataframe = pd.DataFrame(
        {
            "category": ["A", "B"],
            "other_numeric": [1, 2],
            "total_revenue": [100, 200],
        }
    )

    plan = plan_visualization(
        profile_analysis_result(dataframe),
        _aggregate_plan("category", "revenue", "total_revenue"),
    )

    assert plan.y == "total_revenue"


def test_groupby_column_is_preferred_as_dimension() -> None:
    dataframe = pd.DataFrame(
        {
            "other_category": ["X", "Y"],
            "category": ["A", "B"],
            "total_revenue": [100, 200],
        }
    )

    plan = plan_visualization(
        profile_analysis_result(dataframe),
        _aggregate_plan("category", "revenue", "total_revenue"),
    )

    assert plan.x == "category"


def test_unicode_columns_are_preserved_in_plan() -> None:
    dataframe = pd.DataFrame({"产品-类型": ["电脑", "手机"], "总金额": [100, 200]})

    plan = plan_visualization(
        profile_analysis_result(dataframe),
        _aggregate_plan("产品-类型", "订单 金额", "总金额"),
    )

    assert plan.to_dict() == {
        "chart_type": "bar",
        "x": "产品-类型",
        "y": "总金额",
        "title": "总金额 by 产品-类型",
    }


@pytest.mark.parametrize(
    "dataframe",
    [
        pd.DataFrame({"revenue": pd.Series(dtype="float64")}),
        pd.DataFrame({"revenue": [10]}),
        pd.DataFrame({"category": ["A", "B"]}),
        pd.DataFrame({"active": [True, False], "revenue": [10, 20]}),
    ],
)
def test_unsupported_results_return_no_chart(dataframe) -> None:
    plan_contract = _validated_plan(
        list(dataframe.columns),
        [
            {
                "operation": "top_n",
                "sort_by": list(dataframe.columns)[0],
                "n": 1,
                "ascending": False,
            }
        ],
    )

    assert plan_visualization(profile_analysis_result(dataframe), plan_contract) is None


def test_planning_is_deterministic() -> None:
    dataframe = pd.DataFrame({"category": ["A", "B"], "revenue": [10, 20]})
    profile = profile_analysis_result(dataframe)
    plan_contract = _aggregate_plan("category", "raw_revenue", "revenue")

    first = plan_visualization(profile, plan_contract)
    second = plan_visualization(profile, plan_contract)

    assert first == second


def test_planner_rejects_non_profile() -> None:
    with pytest.raises(TypeError, match="AnalysisProfile"):
        plan_visualization({}, _aggregate_plan("category", "revenue", "total"))


def test_planner_rejects_unvalidated_plan() -> None:
    profile = profile_analysis_result(pd.DataFrame({"category": ["A"], "value": [1]}))

    with pytest.raises(TypeError, match="ValidatedAnalysisPlan"):
        plan_visualization(profile, {})

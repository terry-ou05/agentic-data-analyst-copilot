import inspect

import pandas as pd
import pytest
from plotly.graph_objects import Figure

import src.analysis.chart_generator as chart_generator_module
from src.analysis.chart_generator import ChartGenerationError, generate_chart
from src.analysis.visualization import ChartType, VisualizationPlan


@pytest.mark.parametrize(
    ("chart_type", "dataframe", "x", "y", "expected_trace"),
    [
        (
            ChartType.BAR,
            pd.DataFrame({"category": ["A", "B"], "revenue": [10, 20]}),
            "category",
            "revenue",
            "bar",
        ),
        (
            ChartType.LINE,
            pd.DataFrame(
                {
                    "date": pd.to_datetime(["2025-01-02", "2025-01-01"]),
                    "revenue": [20, 10],
                }
            ),
            "date",
            "revenue",
            "scatter",
        ),
        (
            ChartType.SCATTER,
            pd.DataFrame({"revenue": [10, 20], "profit": [2, 5]}),
            "revenue",
            "profit",
            "scatter",
        ),
    ],
)
def test_valid_predefined_charts(
    chart_type,
    dataframe,
    x,
    y,
    expected_trace,
) -> None:
    plan = VisualizationPlan(chart_type, x, y, "Test chart")

    figure = generate_chart(plan, dataframe)

    assert isinstance(figure, Figure)
    assert figure.data[0].type == expected_trace
    assert figure.layout.title.text == "Test chart"


def test_line_chart_sorts_datetime_x_values() -> None:
    dataframe = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-01"]),
            "revenue": [20, 10],
        }
    )

    figure = generate_chart(
        VisualizationPlan(ChartType.LINE, "date", "revenue", "Revenue trend"),
        dataframe,
    )

    assert list(figure.data[0].y) == [10, 20]


def test_line_chart_parses_date_strings_on_dataframe_copy() -> None:
    dataframe = pd.DataFrame(
        {"order_date": ["2025-01-02", "2025-01-01"], "revenue": [20, 10]}
    )
    original = dataframe.copy(deep=True)

    figure = generate_chart(
        VisualizationPlan(ChartType.LINE, "order_date", "revenue", "Revenue trend"),
        dataframe,
    )

    assert list(figure.data[0].y) == [10, 20]
    pd.testing.assert_frame_equal(dataframe, original)


def test_line_chart_rejects_invalid_date_strings() -> None:
    dataframe = pd.DataFrame(
        {"order_date": ["not-a-date"], "revenue": [10]}
    )

    with pytest.raises(ChartGenerationError) as exc_info:
        generate_chart(
            VisualizationPlan(
                ChartType.LINE,
                "order_date",
                "revenue",
                "Revenue trend",
            ),
            dataframe,
        )

    assert exc_info.value.error_code == "INVALID_COLUMN_TYPE"


@pytest.mark.parametrize("missing_column", ["missing_x", "missing_y"])
def test_missing_chart_column_fails(missing_column) -> None:
    dataframe = pd.DataFrame({"category": ["A"], "revenue": [10]})
    x = missing_column if missing_column == "missing_x" else "category"
    y = missing_column if missing_column == "missing_y" else "revenue"

    with pytest.raises(ChartGenerationError, match=missing_column) as exc_info:
        generate_chart(
            VisualizationPlan(ChartType.BAR, x, y, "Missing column"),
            dataframe,
        )

    assert exc_info.value.error_code == "MISSING_COLUMN"


def test_non_numeric_y_column_fails() -> None:
    dataframe = pd.DataFrame({"category": ["A"], "label": ["high"]})

    with pytest.raises(ChartGenerationError) as exc_info:
        generate_chart(
            VisualizationPlan(ChartType.BAR, "category", "label", "Invalid y"),
            dataframe,
        )

    assert exc_info.value.error_code == "INVALID_COLUMN_TYPE"


def test_scatter_non_numeric_x_column_fails() -> None:
    dataframe = pd.DataFrame({"category": ["A"], "revenue": [10]})

    with pytest.raises(ChartGenerationError) as exc_info:
        generate_chart(
            VisualizationPlan(ChartType.SCATTER, "category", "revenue", "Invalid x"),
            dataframe,
        )

    assert exc_info.value.error_code == "INVALID_COLUMN_TYPE"


def test_empty_result_fails_cleanly() -> None:
    dataframe = pd.DataFrame(
        {"category": pd.Series(dtype="object"), "revenue": pd.Series(dtype="float64")}
    )

    with pytest.raises(ChartGenerationError) as exc_info:
        generate_chart(
            VisualizationPlan(ChartType.BAR, "category", "revenue", "Empty"),
            dataframe,
        )

    assert exc_info.value.error_code == "EMPTY_RESULT"


def test_chart_generation_does_not_modify_input() -> None:
    dataframe = pd.DataFrame({"category": ["A", "B"], "revenue": [10, 20]})
    original = dataframe.copy(deep=True)

    generate_chart(
        VisualizationPlan(ChartType.BAR, "category", "revenue", "Revenue"),
        dataframe,
    )

    pd.testing.assert_frame_equal(dataframe, original)


def test_unicode_chart_columns_are_supported() -> None:
    dataframe = pd.DataFrame({"产品-类型": ["电脑", "手机"], "订单 金额": [10, 20]})

    figure = generate_chart(
        VisualizationPlan(ChartType.BAR, "产品-类型", "订单 金额", "订单金额"),
        dataframe,
    )

    assert list(figure.data[0].x) == ["电脑", "手机"]


def test_chart_generator_rejects_wrong_plan_type() -> None:
    with pytest.raises(ChartGenerationError) as exc_info:
        generate_chart({}, pd.DataFrame({"value": [1]}))

    assert exc_info.value.error_code == "INVALID_PLAN_TYPE"


def test_chart_generator_rejects_unknown_chart_type() -> None:
    invalid_plan = VisualizationPlan("pie", "category", "revenue", "Invalid")

    with pytest.raises(ChartGenerationError) as exc_info:
        generate_chart(
            invalid_plan,
            pd.DataFrame({"category": ["A"], "revenue": [1]}),
        )

    assert exc_info.value.error_code == "UNSUPPORTED_CHART_TYPE"


def test_plotly_failure_is_converted_to_safe_error(monkeypatch) -> None:
    def fail_plot(*args, **kwargs):
        raise RuntimeError("internal plotting details")

    monkeypatch.setattr(chart_generator_module.px, "bar", fail_plot)

    with pytest.raises(ChartGenerationError) as exc_info:
        generate_chart(
            VisualizationPlan(ChartType.BAR, "category", "revenue", "Revenue"),
            pd.DataFrame({"category": ["A"], "revenue": [1]}),
        )

    assert exc_info.value.error_code == "CHART_GENERATION_FAILED"
    assert "internal plotting details" not in str(exc_info.value)


def test_chart_generator_contains_no_dynamic_execution() -> None:
    source = inspect.getsource(chart_generator_module)

    for forbidden in ("eval(", "exec(", "compile(", ".query(", "getattr("):
        assert forbidden not in source

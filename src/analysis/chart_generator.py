import pandas as pd
import plotly.express as px
from pandas.api.types import is_numeric_dtype
from plotly.graph_objects import Figure

from src.analysis.visualization import ChartType, VisualizationPlan


class ChartGenerationError(ValueError):
    """A safe failure raised before or during predefined chart generation."""

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        super().__init__(message)


def _validate_chart_inputs(
    visualization_plan: VisualizationPlan,
    dataframe: pd.DataFrame,
) -> None:
    if not isinstance(visualization_plan, VisualizationPlan):
        raise ChartGenerationError(
            "INVALID_PLAN_TYPE",
            "Chart generation requires a VisualizationPlan.",
        )
    if not isinstance(visualization_plan.chart_type, ChartType):
        raise ChartGenerationError(
            "UNSUPPORTED_CHART_TYPE",
            "The requested chart type is not supported.",
        )
    if not isinstance(dataframe, pd.DataFrame):
        raise ChartGenerationError(
            "INVALID_DATAFRAME",
            "Chart input must be a pandas DataFrame.",
        )
    if dataframe.empty:
        raise ChartGenerationError(
            "EMPTY_RESULT",
            "An empty analysis result cannot be charted.",
        )

    missing_columns = [
        column
        for column in (visualization_plan.x, visualization_plan.y)
        if column not in dataframe.columns
    ]
    if missing_columns:
        raise ChartGenerationError(
            "MISSING_COLUMN",
            f"Chart columns are missing: {', '.join(missing_columns)}",
        )
    if not is_numeric_dtype(dataframe[visualization_plan.y].dtype):
        raise ChartGenerationError(
            "INVALID_COLUMN_TYPE",
            f"Chart y column must be numeric: {visualization_plan.y}",
        )
    if (
        visualization_plan.chart_type is ChartType.SCATTER
        and not is_numeric_dtype(dataframe[visualization_plan.x].dtype)
    ):
        raise ChartGenerationError(
            "INVALID_COLUMN_TYPE",
            f"Scatter x column must be numeric: {visualization_plan.x}",
        )


def generate_chart(
    visualization_plan: VisualizationPlan,
    dataframe: pd.DataFrame,
) -> Figure:
    """Generate one chart through fixed, predefined Plotly Express calls."""
    _validate_chart_inputs(visualization_plan, dataframe)
    chart_data = dataframe.copy(deep=True)

    try:
        if visualization_plan.chart_type is ChartType.BAR:
            return px.bar(
                chart_data,
                x=visualization_plan.x,
                y=visualization_plan.y,
                title=visualization_plan.title,
            )
        if visualization_plan.chart_type is ChartType.LINE:
            if not pd.api.types.is_datetime64_any_dtype(
                chart_data[visualization_plan.x].dtype
            ):
                parsed_x = pd.to_datetime(
                    chart_data[visualization_plan.x],
                    errors="coerce",
                )
                invalid_values = (
                    chart_data[visualization_plan.x].notna() & parsed_x.isna()
                )
                if invalid_values.any():
                    raise ChartGenerationError(
                        "INVALID_COLUMN_TYPE",
                        f"Line x column must contain datetime values: "
                        f"{visualization_plan.x}",
                    )
                chart_data[visualization_plan.x] = parsed_x
            ordered_data = chart_data.sort_values(by=visualization_plan.x)
            return px.line(
                ordered_data,
                x=visualization_plan.x,
                y=visualization_plan.y,
                title=visualization_plan.title,
            )
        if visualization_plan.chart_type is ChartType.SCATTER:
            return px.scatter(
                chart_data,
                x=visualization_plan.x,
                y=visualization_plan.y,
                title=visualization_plan.title,
            )
    except ChartGenerationError:
        raise
    except Exception as exc:
        raise ChartGenerationError(
            "CHART_GENERATION_FAILED",
            "The chart could not be generated from this analysis result.",
        ) from exc

    raise ChartGenerationError(
        "UNSUPPORTED_CHART_TYPE",
        "The requested chart type is not supported.",
    )

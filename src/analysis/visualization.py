from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.analysis.profiler import AnalysisProfile
from src.schemas.analysis_plan import OperationType, ValidatedAnalysisPlan


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"


@dataclass(frozen=True)
class VisualizationPlan:
    chart_type: ChartType
    x: str
    y: str
    title: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chart_type": self.chart_type.value,
            "x": self.x,
            "y": self.y,
            "title": self.title,
        }


def _ordered_preferred_columns(
    preferred: list[str],
    available: list[str],
) -> list[str]:
    ordered: list[str] = []
    for column in preferred + available:
        if column in available and column not in ordered:
            ordered.append(column)
    return ordered


def plan_visualization(
    profile: AnalysisProfile,
    validated_plan: ValidatedAnalysisPlan,
) -> VisualizationPlan | None:
    """Choose one chart deterministically from validated result metadata."""
    if not isinstance(profile, AnalysisProfile):
        raise TypeError("An AnalysisProfile is required.")
    if not isinstance(validated_plan, ValidatedAnalysisPlan):
        raise TypeError("A ValidatedAnalysisPlan is required.")
    if profile.empty:
        return None

    columns_by_type: dict[str, list[str]] = {
        "categorical": [],
        "datetime": [],
        "numeric": [],
    }
    for column in profile.columns:
        if column.data_type in columns_by_type:
            columns_by_type[column.data_type].append(column.name)

    groupby_columns: list[str] = []
    aggregate_aliases: list[str] = []
    for operation in validated_plan.operations:
        if operation.operation_type is OperationType.GROUPBY:
            groupby_columns.extend(operation.parameters["columns"])
        elif operation.operation_type is OperationType.AGGREGATE:
            aggregate_aliases.extend(
                metric["alias"] for metric in operation.parameters["metrics"]
            )

    datetime_columns = _ordered_preferred_columns(
        groupby_columns,
        columns_by_type["datetime"],
    )
    categorical_columns = _ordered_preferred_columns(
        groupby_columns,
        columns_by_type["categorical"],
    )
    numeric_columns = _ordered_preferred_columns(
        aggregate_aliases,
        columns_by_type["numeric"],
    )

    if datetime_columns and numeric_columns:
        x_column = datetime_columns[0]
        y_column = numeric_columns[0]
        return VisualizationPlan(
            chart_type=ChartType.LINE,
            x=x_column,
            y=y_column,
            title=f"{y_column} by {x_column}",
        )

    if categorical_columns and numeric_columns:
        x_column = categorical_columns[0]
        y_column = numeric_columns[0]
        return VisualizationPlan(
            chart_type=ChartType.BAR,
            x=x_column,
            y=y_column,
            title=f"{y_column} by {x_column}",
        )

    if len(numeric_columns) >= 2 and profile.rows >= 2:
        x_column, y_column = numeric_columns[:2]
        return VisualizationPlan(
            chart_type=ChartType.SCATTER,
            x=x_column,
            y=y_column,
            title=f"{y_column} vs {x_column}",
        )

    return None

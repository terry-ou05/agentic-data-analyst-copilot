import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from src.schemas.analysis_plan import OperationType, ValidatedAnalysisPlan


MAX_SUMMARY_ROWS = 10
MAX_SUMMARY_COLUMNS = 12
DATETIME_NAME_MARKERS = (
    "date",
    "time",
    "day",
    "month",
    "year",
    "日期",
    "时间",
    "月份",
    "年度",
)


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    data_type: str
    pandas_dtype: str
    missing_values: int
    unique_values: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.data_type,
            "pandas_dtype": self.pandas_dtype,
            "missing_values": self.missing_values,
            "unique_values": self.unique_values,
        }


@dataclass(frozen=True)
class NumericStatistics:
    column: str
    count: int
    sum: int | float | None
    mean: int | float | None
    min: int | float | None
    max: int | float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "sum": self.sum,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
        }


@dataclass(frozen=True)
class AnalysisProfile:
    rows: int
    columns: tuple[ColumnProfile, ...]
    numeric_summary: tuple[NumericStatistics, ...]

    @property
    def empty(self) -> bool:
        return self.rows == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "column_count": len(self.columns),
            "columns": [column.to_dict() for column in self.columns],
            "statistics": {
                "numeric_summary": {
                    item.column: item.to_dict() for item in self.numeric_summary
                },
                "missing_values": {
                    column.name: column.missing_values for column in self.columns
                },
            },
        }


@dataclass(frozen=True)
class AggregatedResultSummary:
    is_aggregated: bool
    total_rows: int
    columns: tuple[str, ...]
    rows: tuple[tuple[tuple[str, Any], ...], ...]
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_aggregated": self.is_aggregated,
            "total_rows": self.total_rows,
            "columns": list(self.columns),
            "rows": [dict(row) for row in self.rows],
            "truncated": self.truncated,
        }


def _classify_column(series: pd.Series) -> str:
    if is_bool_dtype(series.dtype):
        return "boolean"
    if is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if is_numeric_dtype(series.dtype):
        return "numeric"
    column_name = str(series.name).casefold()
    if (
        (is_object_dtype(series.dtype) or is_string_dtype(series.dtype))
        and any(marker in column_name for marker in DATETIME_NAME_MARKERS)
    ):
        non_missing = series.dropna()
        if not non_missing.empty:
            parsed = pd.to_datetime(non_missing, errors="coerce")
            if parsed.notna().all():
                return "datetime"
    return "categorical"


def _safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    return str(value)


def _numeric_statistics(column: str, series: pd.Series) -> NumericStatistics:
    non_missing = series.dropna()
    if non_missing.empty:
        return NumericStatistics(
            column=column,
            count=0,
            sum=None,
            mean=None,
            min=None,
            max=None,
        )

    return NumericStatistics(
        column=column,
        count=int(non_missing.count()),
        sum=_safe_scalar(non_missing.sum()),
        mean=_safe_scalar(non_missing.mean()),
        min=_safe_scalar(non_missing.min()),
        max=_safe_scalar(non_missing.max()),
    )


def profile_analysis_result(dataframe: pd.DataFrame) -> AnalysisProfile:
    """Create bounded analytical metadata without retaining dataframe rows."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Analysis result must be a pandas DataFrame.")

    working_copy = dataframe.copy(deep=True)
    columns: list[ColumnProfile] = []
    numeric_summary: list[NumericStatistics] = []

    for raw_column in working_copy.columns:
        column = str(raw_column)
        series = working_copy[raw_column]
        data_type = _classify_column(series)
        columns.append(
            ColumnProfile(
                name=column,
                data_type=data_type,
                pandas_dtype=str(series.dtype),
                missing_values=int(series.isna().sum()),
                unique_values=int(series.nunique(dropna=True)),
            )
        )
        if data_type == "numeric":
            numeric_summary.append(_numeric_statistics(column, series))

    return AnalysisProfile(
        rows=int(len(working_copy)),
        columns=tuple(columns),
        numeric_summary=tuple(numeric_summary),
    )


def build_aggregated_result_summary(
    dataframe: pd.DataFrame,
    validated_plan: ValidatedAnalysisPlan,
    max_rows: int = MAX_SUMMARY_ROWS,
    max_columns: int = MAX_SUMMARY_COLUMNS,
) -> AggregatedResultSummary:
    """Expose bounded rows only when the validated plan performed aggregation."""
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Analysis result must be a pandas DataFrame.")
    if not isinstance(validated_plan, ValidatedAnalysisPlan):
        raise TypeError("A ValidatedAnalysisPlan is required.")
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer.")
    if (
        not isinstance(max_columns, int)
        or isinstance(max_columns, bool)
        or max_columns < 1
    ):
        raise ValueError("max_columns must be a positive integer.")

    is_aggregated = any(
        operation.operation_type is OperationType.AGGREGATE
        for operation in validated_plan.operations
    )
    selected_columns = tuple(
        str(column) for column in list(dataframe.columns)[:max_columns]
    )
    if not is_aggregated:
        return AggregatedResultSummary(
            is_aggregated=False,
            total_rows=int(len(dataframe)),
            columns=selected_columns,
            rows=(),
            truncated=False,
        )

    selected_frame = dataframe.iloc[:max_rows, :max_columns]
    rows = tuple(
        tuple(
            (str(column), _safe_scalar(value))
            for column, value in zip(selected_frame.columns, row, strict=True)
        )
        for row in selected_frame.itertuples(index=False, name=None)
    )
    return AggregatedResultSummary(
        is_aggregated=True,
        total_rows=int(len(dataframe)),
        columns=selected_columns,
        rows=rows,
        truncated=(len(dataframe) > max_rows or len(dataframe.columns) > max_columns),
    )

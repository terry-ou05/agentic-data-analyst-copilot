from collections.abc import Mapping, Sequence
from typing import Any, Callable

import pandas as pd

from src.schemas.analysis_plan import (
    AGGREGATE_FUNCTIONS,
    FILTER_OPERATORS,
    MAX_TOP_N,
    Operation,
    OperationType,
)


class OperationExecutionError(ValueError):
    """A safe, structured failure raised by a trusted dataframe operation."""

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = error_code
        super().__init__(message)


def _require_dataframe(dataframe: pd.DataFrame) -> None:
    if not isinstance(dataframe, pd.DataFrame):
        raise OperationExecutionError(
            "INVALID_DATAFRAME",
            "Operation input must be a pandas DataFrame.",
        )


def _require_operation(operation: Operation, expected: OperationType) -> Mapping[str, Any]:
    if not isinstance(operation, Operation) or operation.operation_type is not expected:
        raise OperationExecutionError(
            "INVALID_OPERATION",
            f"Expected a validated {expected.value} operation.",
        )
    if not isinstance(operation.parameters, Mapping):
        raise OperationExecutionError(
            "INVALID_PARAMETER",
            f"{expected.value} parameters must be a mapping.",
        )
    return operation.parameters


def _require_existing_columns(
    dataframe: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    missing = sorted({column for column in columns if column not in dataframe.columns})
    if missing:
        raise OperationExecutionError(
            "MISSING_COLUMN",
            f"Required columns are missing: {', '.join(missing)}",
        )


def _require_column_list(value: Any, field_name: str) -> list[str]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise OperationExecutionError(
            "INVALID_PARAMETER",
            f"{field_name} must be a non-empty list of column names.",
        )
    return list(value)


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def apply_filter(dataframe: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    """Apply one allowlisted filter and return a new DataFrame."""
    _require_dataframe(dataframe)
    parameters = _require_operation(operation, OperationType.FILTER)
    column = parameters.get("column")
    operator = parameters.get("operator")
    value = parameters.get("value")

    if not isinstance(column, str) or not column:
        raise OperationExecutionError(
            "INVALID_PARAMETER",
            "Filter column must be a non-empty string.",
        )
    if operator not in FILTER_OPERATORS:
        raise OperationExecutionError(
            "UNSUPPORTED_FILTER_OPERATOR",
            f"Unsupported filter operator: {operator}",
        )
    _require_existing_columns(dataframe, [column])

    if operator in {"in", "not_in"} and (
        not isinstance(value, list)
        or not value
        or any(not _is_json_scalar(item) for item in value)
    ):
        raise OperationExecutionError(
            "INVALID_PARAMETER",
            f"Filter operator {operator} requires a non-empty scalar list.",
        )
    if operator in {"is_null", "not_null"} and value is not None:
        raise OperationExecutionError(
            "INVALID_PARAMETER",
            f"Filter operator {operator} requires a null value.",
        )
    if (
        operator not in {"in", "not_in", "is_null", "not_null"}
        and not _is_json_scalar(value)
    ):
        raise OperationExecutionError(
            "INVALID_PARAMETER",
            f"Filter operator {operator} requires a scalar value.",
        )

    series = dataframe[column]
    if operator == "eq":
        mask = series == value
    elif operator == "ne":
        mask = series != value
    elif operator == "gt":
        mask = series > value
    elif operator == "gte":
        mask = series >= value
    elif operator == "lt":
        mask = series < value
    elif operator == "lte":
        mask = series <= value
    elif operator == "in":
        mask = series.isin(value)
    elif operator == "not_in":
        mask = ~series.isin(value)
    elif operator == "is_null":
        mask = series.isna()
    else:
        mask = series.notna()

    return dataframe.loc[mask.fillna(False)].copy(deep=True)


def apply_groupby(dataframe: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    """Validate group keys and return an isolated input for aggregation."""
    _require_dataframe(dataframe)
    parameters = _require_operation(operation, OperationType.GROUPBY)
    columns = _require_column_list(parameters.get("columns"), "Groupby columns")
    _require_existing_columns(dataframe, columns)
    return dataframe.copy(deep=True)


def _aggregate_sum(series: pd.Series) -> Any:
    return series.sum()


def _aggregate_mean(series: pd.Series) -> Any:
    return series.mean()


def _aggregate_min(series: pd.Series) -> Any:
    return series.min()


def _aggregate_max(series: pd.Series) -> Any:
    return series.max()


def _aggregate_count(series: pd.Series) -> Any:
    return series.count()


_AGGREGATE_DISPATCH: dict[str, Callable[[pd.Series], Any]] = {
    "sum": _aggregate_sum,
    "mean": _aggregate_mean,
    "min": _aggregate_min,
    "max": _aggregate_max,
    "count": _aggregate_count,
}


def _require_metrics(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise OperationExecutionError(
            "INVALID_PARAMETER",
            "Aggregate metrics must be a non-empty list.",
        )

    metrics: list[Mapping[str, Any]] = []
    aliases: set[str] = set()
    for metric in value:
        if not isinstance(metric, Mapping):
            raise OperationExecutionError(
                "INVALID_PARAMETER",
                "Each aggregate metric must be a mapping.",
            )
        column = metric.get("column")
        function = metric.get("function")
        alias = metric.get("alias")
        if not isinstance(column, str) or not column:
            raise OperationExecutionError(
                "INVALID_PARAMETER",
                "Aggregate metric column must be a non-empty string.",
            )
        if function not in AGGREGATE_FUNCTIONS:
            raise OperationExecutionError(
                "UNSUPPORTED_AGGREGATE_FUNCTION",
                f"Unsupported aggregate function: {function}",
            )
        if not isinstance(alias, str) or not alias:
            raise OperationExecutionError(
                "INVALID_PARAMETER",
                "Aggregate alias must be a non-empty string.",
            )
        if alias in aliases:
            raise OperationExecutionError(
                "INVALID_PARAMETER",
                f"Duplicate aggregate alias: {alias}",
            )
        aliases.add(alias)
        metrics.append(metric)
    return metrics


def apply_aggregate(
    dataframe: pd.DataFrame,
    operation: Operation,
    groupby_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Apply fixed aggregate functions, optionally using validated group keys."""
    _require_dataframe(dataframe)
    parameters = _require_operation(operation, OperationType.AGGREGATE)
    metrics = _require_metrics(parameters.get("metrics"))
    grouping_columns = list(groupby_columns)
    if any(not isinstance(column, str) or not column for column in grouping_columns):
        raise OperationExecutionError(
            "INVALID_PARAMETER",
            "Groupby columns must contain only non-empty strings.",
        )

    metric_columns = [metric["column"] for metric in metrics]
    _require_existing_columns(dataframe, grouping_columns + metric_columns)

    if grouping_columns:
        named_aggregations = {
            metric["alias"]: pd.NamedAgg(
                column=metric["column"],
                aggfunc=_AGGREGATE_DISPATCH[metric["function"]],
            )
            for metric in metrics
        }
        return (
            dataframe.groupby(grouping_columns, dropna=False, as_index=False)
            .agg(**named_aggregations)
            .copy(deep=True)
        )

    result_row = {
        metric["alias"]: _AGGREGATE_DISPATCH[metric["function"]](
            dataframe[metric["column"]]
        )
        for metric in metrics
    }
    return pd.DataFrame([result_row])


def apply_top_n(dataframe: pd.DataFrame, operation: Operation) -> pd.DataFrame:
    """Sort by one validated column and return at most 100 rows."""
    _require_dataframe(dataframe)
    parameters = _require_operation(operation, OperationType.TOP_N)
    sort_by = parameters.get("sort_by")
    n = parameters.get("n")
    ascending = parameters.get("ascending")

    if not isinstance(sort_by, str) or not sort_by:
        raise OperationExecutionError(
            "INVALID_PARAMETER",
            "top_n sort_by must be a non-empty string.",
        )
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= MAX_TOP_N:
        raise OperationExecutionError(
            "INVALID_PARAMETER",
            f"top_n n must be between 1 and {MAX_TOP_N}.",
        )
    if not isinstance(ascending, bool):
        raise OperationExecutionError(
            "INVALID_PARAMETER",
            "top_n ascending must be a boolean.",
        )
    _require_existing_columns(dataframe, [sort_by])

    return dataframe.sort_values(by=sort_by, ascending=ascending).head(n).copy(deep=True)

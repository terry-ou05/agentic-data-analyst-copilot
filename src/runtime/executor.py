from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.analysis.operations import (
    OperationExecutionError,
    apply_aggregate,
    apply_filter,
    apply_groupby,
    apply_top_n,
)
from src.schemas.analysis_plan import Operation, OperationType, ValidatedAnalysisPlan


@dataclass(frozen=True)
class AnalysisResult:
    success: bool
    dataframe: pd.DataFrame | None
    error_code: str
    message: str
    executed_operations: tuple[str, ...]
    input_rows: int
    output_rows: int
    warnings: tuple[str, ...]


OperationHandler = Callable[..., pd.DataFrame]

_OPERATION_DISPATCH: dict[OperationType, OperationHandler] = {
    OperationType.FILTER: apply_filter,
    OperationType.GROUPBY: apply_groupby,
    OperationType.AGGREGATE: apply_aggregate,
    OperationType.TOP_N: apply_top_n,
}


def _failure_result(
    error_code: str,
    message: str,
    input_rows: int,
    executed_operations: list[str],
    warnings: list[str],
) -> AnalysisResult:
    return AnalysisResult(
        success=False,
        dataframe=None,
        error_code=error_code,
        message=message,
        executed_operations=tuple(executed_operations),
        input_rows=input_rows,
        output_rows=0,
        warnings=tuple(warnings),
    )


def execute_analysis_plan(
    dataframe: pd.DataFrame,
    validated_plan: ValidatedAnalysisPlan,
) -> AnalysisResult:
    """Execute only trusted operations from a validated structured plan."""
    if not isinstance(dataframe, pd.DataFrame):
        return _failure_result(
            error_code="INVALID_DATAFRAME",
            message="Analysis input must be a pandas DataFrame.",
            input_rows=0,
            executed_operations=[],
            warnings=[],
        )

    input_rows = len(dataframe)
    if not isinstance(validated_plan, ValidatedAnalysisPlan):
        return _failure_result(
            error_code="INVALID_PLAN_TYPE",
            message="Executor requires a ValidatedAnalysisPlan.",
            input_rows=input_rows,
            executed_operations=[],
            warnings=[],
        )

    current = dataframe.copy(deep=True)
    executed_operations: list[str] = []
    warnings: list[str] = []
    pending_groupby_columns: tuple[str, ...] = ()
    if current.empty:
        warnings.append("Input DataFrame is empty.")

    for index, operation in enumerate(validated_plan.operations):
        if not isinstance(operation, Operation) or not isinstance(
            operation.operation_type, OperationType
        ):
            return _failure_result(
                error_code="UNSUPPORTED_OPERATION",
                message=f"Operation at position {index} is not supported.",
                input_rows=input_rows,
                executed_operations=executed_operations,
                warnings=warnings,
            )

        operation_type = operation.operation_type
        handler = _OPERATION_DISPATCH.get(operation_type)
        if handler is None:
            return _failure_result(
                error_code="UNSUPPORTED_OPERATION",
                message=f"Operation {operation_type.value} is not supported.",
                input_rows=input_rows,
                executed_operations=executed_operations,
                warnings=warnings,
            )

        previous_rows = len(current)
        try:
            if operation_type is OperationType.GROUPBY:
                current = handler(current, operation)
                pending_groupby_columns = tuple(operation.parameters["columns"])
            elif operation_type is OperationType.AGGREGATE:
                current = handler(current, operation, pending_groupby_columns)
                pending_groupby_columns = ()
            else:
                current = handler(current, operation)
        except OperationExecutionError as exc:
            return _failure_result(
                error_code=exc.error_code,
                message=str(exc),
                input_rows=input_rows,
                executed_operations=executed_operations,
                warnings=warnings,
            )
        except Exception:
            return _failure_result(
                error_code="OPERATION_FAILED",
                message=f"Operation {operation_type.value} could not be completed.",
                input_rows=input_rows,
                executed_operations=executed_operations,
                warnings=warnings,
            )

        executed_operations.append(operation_type.value)
        if previous_rows and current.empty:
            warnings.append(f"Operation {operation_type.value} produced an empty result.")

    if pending_groupby_columns:
        return _failure_result(
            error_code="INVALID_OPERATION_ORDER",
            message="A groupby operation must be followed by aggregate.",
            input_rows=input_rows,
            executed_operations=executed_operations,
            warnings=warnings,
        )

    return AnalysisResult(
        success=True,
        dataframe=current,
        error_code="",
        message="Analysis plan executed successfully.",
        executed_operations=tuple(executed_operations),
        input_rows=input_rows,
        output_rows=len(current),
        warnings=tuple(warnings),
    )

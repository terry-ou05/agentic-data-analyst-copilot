from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


PLAN_VERSION = "1.0"
MAX_TOP_N = 100
FILTER_OPERATORS = frozenset(
    {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in"}
)
AGGREGATE_FUNCTIONS = frozenset({"sum", "mean", "min", "max", "count"})


class OperationType(str, Enum):
    FILTER = "filter"
    GROUPBY = "groupby"
    AGGREGATE = "aggregate"
    TOP_N = "top_n"


@dataclass(frozen=True)
class Operation:
    operation_type: OperationType
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class AnalysisPlan:
    version: str
    goal: str
    operations: tuple[Operation, ...]


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


class PlanParseError(ValueError):
    """Raised when a structured plan does not match the JSON contract."""


def _require_exact_fields(
    payload: Mapping[str, Any],
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise PlanParseError(f"{context} is missing required fields: {', '.join(missing)}")

    unknown = sorted(set(payload) - required)
    if unknown:
        raise PlanParseError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanParseError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PlanParseError(f"{field_name} must be a non-empty list of strings.")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise PlanParseError(f"{field_name} must contain only non-empty strings.")
    return [item.strip() for item in value]


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _parse_filter_operation(payload: Mapping[str, Any], index: int) -> Operation:
    context = f"operations[{index}]"
    _require_exact_fields(
        payload,
        {"operation", "column", "operator", "value"},
        context,
    )
    column = _require_non_empty_string(payload["column"], f"{context}.column")
    operator = _require_non_empty_string(payload["operator"], f"{context}.operator")
    value = payload["value"]
    if isinstance(value, list):
        if not value or any(not _is_json_scalar(item) for item in value):
            raise PlanParseError(
                f"{context}.value must be a JSON scalar or a non-empty list of scalars."
            )
    elif not _is_json_scalar(value):
        raise PlanParseError(
            f"{context}.value must be a JSON scalar or a non-empty list of scalars."
        )

    return Operation(
        operation_type=OperationType.FILTER,
        parameters={"column": column, "operator": operator, "value": value},
    )


def _parse_groupby_operation(payload: Mapping[str, Any], index: int) -> Operation:
    context = f"operations[{index}]"
    _require_exact_fields(payload, {"operation", "columns"}, context)
    columns = _require_string_list(payload["columns"], f"{context}.columns")
    return Operation(
        operation_type=OperationType.GROUPBY,
        parameters={"columns": columns},
    )


def _parse_aggregate_operation(payload: Mapping[str, Any], index: int) -> Operation:
    context = f"operations[{index}]"
    _require_exact_fields(payload, {"operation", "metrics"}, context)
    metrics = payload["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise PlanParseError(f"{context}.metrics must be a non-empty list.")

    parsed_metrics = []
    for metric_index, metric in enumerate(metrics):
        metric_context = f"{context}.metrics[{metric_index}]"
        if not isinstance(metric, dict):
            raise PlanParseError(f"{metric_context} must be an object.")
        _require_exact_fields(
            metric,
            {"column", "function", "alias"},
            metric_context,
        )
        parsed_metrics.append(
            {
                "column": _require_non_empty_string(
                    metric["column"], f"{metric_context}.column"
                ),
                "function": _require_non_empty_string(
                    metric["function"], f"{metric_context}.function"
                ),
                "alias": _require_non_empty_string(
                    metric["alias"], f"{metric_context}.alias"
                ),
            }
        )

    return Operation(
        operation_type=OperationType.AGGREGATE,
        parameters={"metrics": parsed_metrics},
    )


def _parse_top_n_operation(payload: Mapping[str, Any], index: int) -> Operation:
    context = f"operations[{index}]"
    _require_exact_fields(
        payload,
        {"operation", "sort_by", "n", "ascending"},
        context,
    )
    sort_by = _require_non_empty_string(payload["sort_by"], f"{context}.sort_by")
    n = payload["n"]
    ascending = payload["ascending"]
    if not isinstance(n, int) or isinstance(n, bool):
        raise PlanParseError(f"{context}.n must be an integer.")
    if not isinstance(ascending, bool):
        raise PlanParseError(f"{context}.ascending must be a boolean.")
    return Operation(
        operation_type=OperationType.TOP_N,
        parameters={"sort_by": sort_by, "n": n, "ascending": ascending},
    )


def _parse_operation(payload: Any, index: int) -> Operation:
    context = f"operations[{index}]"
    if not isinstance(payload, dict):
        raise PlanParseError(f"{context} must be an object.")

    operation_name = _require_non_empty_string(
        payload.get("operation"),
        f"{context}.operation",
    )
    try:
        operation_type = OperationType(operation_name)
    except ValueError as exc:
        raise PlanParseError(f"Unsupported operation: {operation_name}") from exc

    parsers = {
        OperationType.FILTER: _parse_filter_operation,
        OperationType.GROUPBY: _parse_groupby_operation,
        OperationType.AGGREGATE: _parse_aggregate_operation,
        OperationType.TOP_N: _parse_top_n_operation,
    }
    return parsers[operation_type](payload, index)


def parse_analysis_plan(payload: Any) -> AnalysisPlan:
    """Parse a JSON-compatible object into the V5.1 analysis plan contract."""
    if not isinstance(payload, dict):
        raise PlanParseError("Analysis plan must be a JSON object.")
    _require_exact_fields(payload, {"version", "goal", "operations"}, "plan")

    version = _require_non_empty_string(payload["version"], "plan.version")
    goal = _require_non_empty_string(payload["goal"], "plan.goal")
    raw_operations = payload["operations"]
    if not isinstance(raw_operations, list):
        raise PlanParseError("plan.operations must be a list.")

    operations = tuple(
        _parse_operation(operation, index)
        for index, operation in enumerate(raw_operations)
    )
    return AnalysisPlan(version=version, goal=goal, operations=operations)


def validate_analysis_plan(
    plan: AnalysisPlan,
    schema_summary: Mapping[str, Any],
) -> ValidationResult:
    """Validate plan semantics against the current dataset schema."""
    errors: list[str] = []
    raw_columns = schema_summary.get("column_names")
    if not isinstance(raw_columns, list) or any(
        not isinstance(column, str) for column in raw_columns
    ):
        return ValidationResult(False, ("Schema column_names must be a list of strings.",))

    schema_columns = set(raw_columns)
    available_columns = set(raw_columns)
    groupby_columns: list[str] = []
    operation_counts = {operation_type: 0 for operation_type in OperationType}
    phase_order = {
        OperationType.FILTER: 0,
        OperationType.GROUPBY: 1,
        OperationType.AGGREGATE: 2,
        OperationType.TOP_N: 3,
    }

    if plan.version != PLAN_VERSION:
        errors.append(f"Unsupported plan version: {plan.version}")
    if not plan.operations:
        errors.append("Plan must contain at least one operation.")

    previous_phase = 0
    for index, operation in enumerate(plan.operations):
        if not isinstance(operation.operation_type, OperationType):
            errors.append(f"operations[{index}] has an unsupported operation type.")
            continue

        operation_counts[operation.operation_type] += 1
        current_phase = phase_order[operation.operation_type]
        if current_phase < previous_phase:
            errors.append(f"operations[{index}] is out of order.")
        previous_phase = max(previous_phase, current_phase)

        parameters = operation.parameters
        if operation.operation_type is OperationType.FILTER:
            column = parameters["column"]
            operator = parameters["operator"]
            value = parameters["value"]
            if column not in schema_columns:
                errors.append(f"Filter column does not exist: {column}")
            if operator not in FILTER_OPERATORS:
                errors.append(f"Unsupported filter operator: {operator}")
            if operator in {"in", "not_in"} and not isinstance(value, list):
                errors.append(f"Filter operator {operator} requires a list value.")
            if operator not in {"in", "not_in"} and isinstance(value, list):
                errors.append(f"Filter operator {operator} requires a scalar value.")

        elif operation.operation_type is OperationType.GROUPBY:
            groupby_columns = list(parameters["columns"])
            for column in groupby_columns:
                if column not in schema_columns:
                    errors.append(f"Groupby column does not exist: {column}")

        elif operation.operation_type is OperationType.AGGREGATE:
            aliases: set[str] = set()
            for metric in parameters["metrics"]:
                column = metric["column"]
                function = metric["function"]
                alias = metric["alias"]
                if column not in schema_columns:
                    errors.append(f"Aggregate column does not exist: {column}")
                if function not in AGGREGATE_FUNCTIONS:
                    errors.append(f"Unsupported aggregate function: {function}")
                if alias in aliases:
                    errors.append(f"Duplicate aggregate alias: {alias}")
                if alias in groupby_columns:
                    errors.append(f"Aggregate alias conflicts with groupby column: {alias}")
                aliases.add(alias)
            available_columns = set(groupby_columns) | aliases

        elif operation.operation_type is OperationType.TOP_N:
            sort_by = parameters["sort_by"]
            n = parameters["n"]
            if sort_by not in available_columns:
                errors.append(f"top_n sort_by column does not exist: {sort_by}")
            if not 1 <= n <= MAX_TOP_N:
                errors.append(f"top_n n must be between 1 and {MAX_TOP_N}.")

    if operation_counts[OperationType.GROUPBY] > 1:
        errors.append("Plan may contain at most one groupby operation.")
    if operation_counts[OperationType.AGGREGATE] > 1:
        errors.append("Plan may contain at most one aggregate operation.")
    if operation_counts[OperationType.TOP_N] > 1:
        errors.append("Plan may contain at most one top_n operation.")
    if (
        operation_counts[OperationType.GROUPBY]
        and not operation_counts[OperationType.AGGREGATE]
    ):
        errors.append("A groupby operation must be followed by aggregate.")
    if operation_counts[OperationType.TOP_N] and plan.operations[-1].operation_type is not OperationType.TOP_N:
        errors.append("top_n must be the final operation.")

    return ValidationResult(valid=not errors, errors=tuple(errors))

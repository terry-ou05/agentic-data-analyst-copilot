import inspect

import pandas as pd
import pytest

import src.agents.code_generator as code_generator
import src.analysis.operations as operations_module
import src.runtime.executor as executor_module
from src.runtime.executor import execute_analysis_plan
from src.schemas.analysis_plan import (
    Operation,
    ValidatedAnalysisPlan,
    create_validated_analysis_plan,
    parse_analysis_plan,
)


def _validated_plan(
    dataframe: pd.DataFrame,
    operations: list[dict],
    schema_columns: list[str] | None = None,
) -> ValidatedAnalysisPlan:
    plan = parse_analysis_plan(
        {
            "version": "1.0",
            "goal": "Execute a trusted analysis plan",
            "operations": operations,
        }
    )
    return create_validated_analysis_plan(
        plan,
        {"column_names": schema_columns or list(dataframe.columns)},
    )


def test_executor_runs_filter_groupby_aggregate_and_top_n() -> None:
    dataframe = pd.DataFrame(
        {
            "category": ["A", "B", "A", "B"],
            "region": ["North", "North", "South", "South"],
            "revenue": [100, 200, 50, 25],
        }
    )
    validated_plan = _validated_plan(
        dataframe,
        [
            {
                "operation": "filter",
                "column": "region",
                "operator": "eq",
                "value": "North",
            },
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
            {
                "operation": "top_n",
                "sort_by": "total_revenue",
                "n": 1,
                "ascending": False,
            },
        ],
    )

    result = execute_analysis_plan(dataframe, validated_plan)
    expected = pd.DataFrame({"category": ["B"], "total_revenue": [200]})

    assert result.success is True
    pd.testing.assert_frame_equal(
        result.dataframe.reset_index(drop=True),
        expected,
    )
    assert result.executed_operations == (
        "filter",
        "groupby",
        "aggregate",
        "top_n",
    )
    assert result.input_rows == 4
    assert result.output_rows == 1


@pytest.mark.parametrize("raw_plan", [{}, "{}", "generated Python code"])
def test_executor_rejects_unvalidated_plan_inputs(raw_plan) -> None:
    result = execute_analysis_plan(pd.DataFrame({"value": [1]}), raw_plan)

    assert result.success is False
    assert result.error_code == "INVALID_PLAN_TYPE"
    assert result.dataframe is None


def test_executor_rejects_unsupported_operation() -> None:
    forged_plan = ValidatedAnalysisPlan(
        version="1.0",
        goal="Forged plan",
        operations=(Operation(operation_type="join", parameters={}),),
        schema_columns=("value",),
    )

    result = execute_analysis_plan(pd.DataFrame({"value": [1]}), forged_plan)

    assert result.success is False
    assert result.error_code == "UNSUPPORTED_OPERATION"


def test_executor_returns_structured_missing_column_error() -> None:
    dataframe = pd.DataFrame({"value": [1, 2]})
    validated_plan = _validated_plan(
        dataframe,
        [
            {
                "operation": "filter",
                "column": "missing_column",
                "operator": "eq",
                "value": 1,
            }
        ],
        schema_columns=["value", "missing_column"],
    )

    result = execute_analysis_plan(dataframe, validated_plan)

    assert result.success is False
    assert result.error_code == "MISSING_COLUMN"
    assert "missing_column" in result.message


def test_executor_does_not_call_code_generator(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("code_generator must not participate in execution")

    monkeypatch.setattr(code_generator, "generate_analysis_code", fail_if_called)
    dataframe = pd.DataFrame({"value": [1, 2]})
    validated_plan = _validated_plan(
        dataframe,
        [
            {
                "operation": "top_n",
                "sort_by": "value",
                "n": 1,
                "ascending": False,
            }
        ],
    )

    result = execute_analysis_plan(dataframe, validated_plan)

    assert result.success is True
    assert result.dataframe["value"].tolist() == [2]


def test_executor_source_contains_no_dynamic_execution() -> None:
    source = inspect.getsource(executor_module) + inspect.getsource(operations_module)

    for forbidden in ("eval(", "exec(", "compile(", ".query(", "getattr("):
        assert forbidden not in source
    assert "lambda" not in source


def test_empty_dataframe_returns_success_with_warning() -> None:
    dataframe = pd.DataFrame({"category": pd.Series(dtype="object")})
    validated_plan = _validated_plan(
        dataframe,
        [
            {
                "operation": "filter",
                "column": "category",
                "operator": "eq",
                "value": "A",
            }
        ],
    )

    result = execute_analysis_plan(dataframe, validated_plan)

    assert result.success is True
    assert result.output_rows == 0
    assert "Input DataFrame is empty." in result.warnings


def test_filter_that_removes_all_rows_returns_empty_success() -> None:
    dataframe = pd.DataFrame({"category": ["A", "B"]})
    validated_plan = _validated_plan(
        dataframe,
        [
            {
                "operation": "filter",
                "column": "category",
                "operator": "eq",
                "value": "missing",
            }
        ],
    )

    result = execute_analysis_plan(dataframe, validated_plan)

    assert result.success is True
    assert result.dataframe.empty
    assert "Operation filter produced an empty result." in result.warnings


def test_executor_does_not_modify_input_dataframe() -> None:
    dataframe = pd.DataFrame({"value": [3, 1, 2]})
    original = dataframe.copy(deep=True)
    validated_plan = _validated_plan(
        dataframe,
        [
            {
                "operation": "top_n",
                "sort_by": "value",
                "n": 2,
                "ascending": True,
            }
        ],
    )

    result = execute_analysis_plan(dataframe, validated_plan)
    result.dataframe.loc[:, "value"] = 0

    pd.testing.assert_frame_equal(dataframe, original)

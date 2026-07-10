import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

import pandas as pd

from src.agents import insight_generator, plan_generator
from src.analysis.chart_generator import ChartGenerationError, generate_chart
from src.analysis.profiler import (
    build_aggregated_result_summary,
    profile_analysis_result,
)
from src.analysis.visualization import plan_visualization
from src.connectors.sqlite_connector import SQLiteConnector, SQLiteConnectorError
from src.evaluation.cases import (
    EvaluationCase,
    SemanticEvaluationCase,
    default_evaluation_cases,
    default_semantic_evaluation_cases,
)
from src.evaluation.synthetic_data import (
    SYNTHETIC_TABLE_NAME,
    create_synthetic_sqlite_database,
)
from src.llm.client import LLMResult
from src.runtime.executor import execute_analysis_plan
from src.schemas.analysis_plan import (
    AnalysisPlan,
    OperationType,
    create_validated_analysis_plan,
    validate_analysis_plan,
)


@dataclass(frozen=True)
class EvaluationCaseResult:
    name: str
    question: str
    success: bool
    error: str
    plan_success: bool
    validation_success: bool
    execution_success: bool
    profile_success: bool
    visualization_success: bool
    chart_success: bool
    insight_success: bool
    input_rows: int
    output_rows: int
    executed_operations: tuple[str, ...]
    chart_type: str | None
    plan_duration_ms: float
    execution_duration_ms: float
    report_duration_ms: float
    total_duration_ms: float
    generated_plan: AnalysisPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "question": self.question,
            "success": self.success,
            "error": self.error,
            "phases": {
                "plan": self.plan_success,
                "validation": self.validation_success,
                "execution": self.execution_success,
                "profile": self.profile_success,
                "visualization": self.visualization_success,
                "chart": self.chart_success,
                "insight": self.insight_success,
            },
            "metrics": {
                "input_rows": self.input_rows,
                "output_rows": self.output_rows,
                "executed_operations": list(self.executed_operations),
                "chart_type": self.chart_type,
                "plan_duration_ms": self.plan_duration_ms,
                "execution_duration_ms": self.execution_duration_ms,
                "report_duration_ms": self.report_duration_ms,
                "total_duration_ms": self.total_duration_ms,
            },
            "generated_operations": (
                [operation.operation_type.value for operation in self.generated_plan.operations]
                if self.generated_plan is not None
                else []
            ),
        }


@dataclass(frozen=True)
class EvaluationReport:
    database_path: str
    table_name: str
    data_load_duration_ms: float
    results: tuple[EvaluationCaseResult, ...]
    setup_error: str = ""

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def successful_cases(self) -> int:
        return sum(result.success for result in self.results)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.successful_cases / self.total_cases

    @property
    def average_execution_duration_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.execution_duration_ms for result in self.results) / len(
            self.results
        )

    @property
    def average_total_duration_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(result.total_duration_ms for result in self.results) / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "database_path": self.database_path,
            "table_name": self.table_name,
            "setup_error": self.setup_error,
            "metrics": {
                "total_cases": self.total_cases,
                "successful_cases": self.successful_cases,
                "success_rate": self.success_rate,
                "data_load_duration_ms": self.data_load_duration_ms,
                "average_execution_duration_ms": self.average_execution_duration_ms,
                "average_total_duration_ms": self.average_total_duration_ms,
            },
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class SemanticVariationResult:
    """One natural-language variation evaluated against its business capability."""

    question: str
    workflow_result: EvaluationCaseResult
    capability_success: bool
    missing_operations: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    missing_metrics: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return self.workflow_result.success and self.capability_success

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "success": self.success,
            "capability_success": self.capability_success,
            "missing_operations": list(self.missing_operations),
            "missing_columns": list(self.missing_columns),
            "missing_metrics": list(self.missing_metrics),
            "workflow": self.workflow_result.to_dict(),
        }


@dataclass(frozen=True)
class SemanticIntentResult:
    """Aggregated robustness result for one business intent."""

    intent_name: str
    description: str
    expected_operations: tuple[str, ...]
    expected_columns: tuple[str, ...]
    expected_metrics: tuple[str, ...]
    variations: tuple[SemanticVariationResult, ...]

    @property
    def total_variations(self) -> int:
        return len(self.variations)

    @property
    def successful_variations(self) -> int:
        return sum(variation.success for variation in self.variations)

    @property
    def success_rate(self) -> float:
        if not self.variations:
            return 0.0
        return self.successful_variations / self.total_variations

    @property
    def robust(self) -> bool:
        return bool(self.variations) and self.successful_variations == self.total_variations

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_name": self.intent_name,
            "description": self.description,
            "expected_capability": {
                "operations": list(self.expected_operations),
                "columns": list(self.expected_columns),
                "metrics": list(self.expected_metrics),
            },
            "metrics": {
                "total_variations": self.total_variations,
                "successful_variations": self.successful_variations,
                "success_rate": self.success_rate,
                "robust": self.robust,
            },
            "variations": [variation.to_dict() for variation in self.variations],
        }


@dataclass(frozen=True)
class SemanticEvaluationReport:
    """Intent-level and variation-level semantic robustness metrics."""

    database_path: str
    table_name: str
    data_load_duration_ms: float
    intents: tuple[SemanticIntentResult, ...]
    setup_error: str = ""

    @property
    def total_intents(self) -> int:
        return len(self.intents)

    @property
    def robust_intents(self) -> int:
        return sum(intent.robust for intent in self.intents)

    @property
    def intent_robustness_rate(self) -> float:
        if not self.intents:
            return 0.0
        return self.robust_intents / self.total_intents

    @property
    def total_variations(self) -> int:
        return sum(intent.total_variations for intent in self.intents)

    @property
    def successful_variations(self) -> int:
        return sum(intent.successful_variations for intent in self.intents)

    @property
    def variation_success_rate(self) -> float:
        if not self.total_variations:
            return 0.0
        return self.successful_variations / self.total_variations

    def to_dict(self) -> dict[str, Any]:
        return {
            "database_path": self.database_path,
            "table_name": self.table_name,
            "setup_error": self.setup_error,
            "metrics": {
                "total_intents": self.total_intents,
                "robust_intents": self.robust_intents,
                "intent_robustness_rate": self.intent_robustness_rate,
                "total_variations": self.total_variations,
                "successful_variations": self.successful_variations,
                "variation_success_rate": self.variation_success_rate,
                "data_load_duration_ms": self.data_load_duration_ms,
            },
            "intents": [intent.to_dict() for intent in self.intents],
        }


class EvaluationRunner:
    """Run deterministic offline evaluation cases through the existing V5/V6 modules."""

    def __init__(
        self,
        database_path: str | Path,
        table_name: str = SYNTHETIC_TABLE_NAME,
    ) -> None:
        self.database_path = Path(database_path)
        self.table_name = table_name

    @staticmethod
    def _milliseconds(start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000, 3)

    def _failure_result(
        self,
        case: EvaluationCase,
        error: str,
        total_start: float,
        plan_success: bool = False,
        validation_success: bool = False,
        execution_success: bool = False,
        profile_success: bool = False,
        visualization_success: bool = False,
        chart_success: bool = False,
        insight_success: bool = False,
        input_rows: int = 0,
        output_rows: int = 0,
        executed_operations: tuple[str, ...] = (),
        chart_type: str | None = None,
        plan_duration_ms: float = 0.0,
        execution_duration_ms: float = 0.0,
        report_duration_ms: float = 0.0,
        generated_plan: AnalysisPlan | None = None,
    ) -> EvaluationCaseResult:
        return EvaluationCaseResult(
            name=case.name,
            question=case.question,
            success=False,
            error=error,
            plan_success=plan_success,
            validation_success=validation_success,
            execution_success=execution_success,
            profile_success=profile_success,
            visualization_success=visualization_success,
            chart_success=chart_success,
            insight_success=insight_success,
            input_rows=input_rows,
            output_rows=output_rows,
            executed_operations=executed_operations,
            chart_type=chart_type,
            plan_duration_ms=plan_duration_ms,
            execution_duration_ms=execution_duration_ms,
            report_duration_ms=report_duration_ms,
            total_duration_ms=self._milliseconds(total_start),
            generated_plan=generated_plan,
        )

    def _run_case(
        self,
        dataframe: pd.DataFrame,
        schema_summary: dict[str, Any],
        case: EvaluationCase,
    ) -> EvaluationCaseResult:
        total_start = time.perf_counter()
        plan_start = time.perf_counter()
        mocked_plan_response = LLMResult(
            success=True,
            content=json.dumps(case.plan_payload),
        )
        with patch.object(
            plan_generator,
            "generate_chat_completion",
            return_value=mocked_plan_response,
        ):
            plan_result = plan_generator.generate_structured_plan(
                schema_summary,
                case.question,
            )
        plan_duration_ms = self._milliseconds(plan_start)
        if not plan_result.success or plan_result.plan is None:
            return self._failure_result(
                case,
                plan_result.error or "Structured plan generation failed.",
                total_start,
                plan_duration_ms=plan_duration_ms,
            )

        validation_result = validate_analysis_plan(plan_result.plan, schema_summary)
        if not validation_result.valid:
            return self._failure_result(
                case,
                "Plan validation failed: " + "; ".join(validation_result.errors),
                total_start,
                plan_success=True,
                plan_duration_ms=plan_duration_ms,
                generated_plan=plan_result.plan,
            )

        try:
            validated_plan = create_validated_analysis_plan(
                plan_result.plan,
                schema_summary,
            )
        except ValueError as exc:
            return self._failure_result(
                case,
                "Validated plan creation failed.",
                total_start,
                plan_success=True,
                validation_success=True,
                plan_duration_ms=plan_duration_ms,
                generated_plan=plan_result.plan,
            )

        execution_start = time.perf_counter()
        execution_result = execute_analysis_plan(dataframe, validated_plan)
        execution_duration_ms = self._milliseconds(execution_start)
        if not execution_result.success or execution_result.dataframe is None:
            return self._failure_result(
                case,
                execution_result.message,
                total_start,
                plan_success=True,
                validation_success=True,
                input_rows=execution_result.input_rows,
                output_rows=execution_result.output_rows,
                executed_operations=execution_result.executed_operations,
                plan_duration_ms=plan_duration_ms,
                execution_duration_ms=execution_duration_ms,
                generated_plan=plan_result.plan,
            )

        report_start = time.perf_counter()
        try:
            profile = profile_analysis_result(execution_result.dataframe)
            visualization_plan = plan_visualization(profile, validated_plan)
            if visualization_plan is None:
                return self._failure_result(
                    case,
                    "No visualization plan was produced.",
                    total_start,
                    plan_success=True,
                    validation_success=True,
                    execution_success=True,
                    profile_success=True,
                    input_rows=execution_result.input_rows,
                    output_rows=execution_result.output_rows,
                    executed_operations=execution_result.executed_operations,
                    plan_duration_ms=plan_duration_ms,
                    execution_duration_ms=execution_duration_ms,
                    report_duration_ms=self._milliseconds(report_start),
                    generated_plan=plan_result.plan,
                )
            if (
                case.expected_chart_type is not None
                and visualization_plan.chart_type is not case.expected_chart_type
            ):
                return self._failure_result(
                    case,
                    "Visualization chart type did not match the evaluation case.",
                    total_start,
                    plan_success=True,
                    validation_success=True,
                    execution_success=True,
                    profile_success=True,
                    visualization_success=True,
                    input_rows=execution_result.input_rows,
                    output_rows=execution_result.output_rows,
                    executed_operations=execution_result.executed_operations,
                    chart_type=visualization_plan.chart_type.value,
                    plan_duration_ms=plan_duration_ms,
                    execution_duration_ms=execution_duration_ms,
                    report_duration_ms=self._milliseconds(report_start),
                    generated_plan=plan_result.plan,
                )
            generate_chart(visualization_plan, execution_result.dataframe)
            result_summary = build_aggregated_result_summary(
                execution_result.dataframe,
                validated_plan,
            )
            with patch.object(
                insight_generator,
                "generate_chat_completion",
                return_value=LLMResult(
                    success=True,
                    content="The evaluation workflow completed successfully.",
                ),
            ):
                insight_result = insight_generator.generate_insight(
                    profile,
                    result_summary,
                    visualization_plan,
                )
        except (ChartGenerationError, TypeError, ValueError) as exc:
            return self._failure_result(
                case,
                "Report generation failed.",
                total_start,
                plan_success=True,
                validation_success=True,
                execution_success=True,
                profile_success=True,
                visualization_success=True,
                input_rows=execution_result.input_rows,
                output_rows=execution_result.output_rows,
                executed_operations=execution_result.executed_operations,
                plan_duration_ms=plan_duration_ms,
                execution_duration_ms=execution_duration_ms,
                report_duration_ms=self._milliseconds(report_start),
                generated_plan=plan_result.plan,
            )

        report_duration_ms = self._milliseconds(report_start)
        if not insight_result.success:
            return self._failure_result(
                case,
                insight_result.error,
                total_start,
                plan_success=True,
                validation_success=True,
                execution_success=True,
                profile_success=True,
                visualization_success=True,
                chart_success=True,
                input_rows=execution_result.input_rows,
                output_rows=execution_result.output_rows,
                executed_operations=execution_result.executed_operations,
                chart_type=visualization_plan.chart_type.value,
                plan_duration_ms=plan_duration_ms,
                execution_duration_ms=execution_duration_ms,
                report_duration_ms=report_duration_ms,
                generated_plan=plan_result.plan,
            )

        actual_top_value = None
        if case.expected_top_column is not None and case.expected_top_value is not None:
            actual_top_value = str(
                execution_result.dataframe.iloc[0][case.expected_top_column]
            )
        if actual_top_value is not None and actual_top_value != case.expected_top_value:
            return self._failure_result(
                case,
                "Result did not match the expected business outcome.",
                total_start,
                plan_success=True,
                validation_success=True,
                execution_success=True,
                profile_success=True,
                visualization_success=True,
                chart_success=True,
                insight_success=True,
                input_rows=execution_result.input_rows,
                output_rows=execution_result.output_rows,
                executed_operations=execution_result.executed_operations,
                chart_type=visualization_plan.chart_type.value,
                plan_duration_ms=plan_duration_ms,
                execution_duration_ms=execution_duration_ms,
                report_duration_ms=report_duration_ms,
                generated_plan=plan_result.plan,
            )

        return EvaluationCaseResult(
            name=case.name,
            question=case.question,
            success=True,
            error="",
            plan_success=True,
            validation_success=True,
            execution_success=True,
            profile_success=True,
            visualization_success=True,
            chart_success=True,
            insight_success=True,
            input_rows=execution_result.input_rows,
            output_rows=execution_result.output_rows,
            executed_operations=execution_result.executed_operations,
            chart_type=visualization_plan.chart_type.value,
            plan_duration_ms=plan_duration_ms,
            execution_duration_ms=execution_duration_ms,
            report_duration_ms=report_duration_ms,
            total_duration_ms=self._milliseconds(total_start),
            generated_plan=plan_result.plan,
        )

    @staticmethod
    def _capability_gaps(
        plan: AnalysisPlan | None,
        case: SemanticEvaluationCase,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        if plan is None:
            return (
                case.expected_operations,
                case.expected_columns,
                tuple(
                    f"{metric.column}:{metric.function}"
                    for metric in case.expected_metrics
                ),
            )

        generated_operations = {
            operation.operation_type.value for operation in plan.operations
        }
        referenced_columns: set[str] = set()
        generated_metrics: set[tuple[str, str]] = set()
        for operation in plan.operations:
            if operation.operation_type is OperationType.FILTER:
                referenced_columns.add(operation.parameters["column"])
            elif operation.operation_type is OperationType.GROUPBY:
                referenced_columns.update(operation.parameters["columns"])
            elif operation.operation_type is OperationType.AGGREGATE:
                for metric in operation.parameters["metrics"]:
                    referenced_columns.add(metric["column"])
                    generated_metrics.add((metric["column"], metric["function"]))

        missing_operations = tuple(
            operation
            for operation in case.expected_operations
            if operation not in generated_operations
        )
        missing_columns = tuple(
            column for column in case.expected_columns if column not in referenced_columns
        )
        missing_metrics = tuple(
            f"{metric.column}:{metric.function}"
            for metric in case.expected_metrics
            if (metric.column, metric.function) not in generated_metrics
        )
        return missing_operations, missing_columns, missing_metrics

    def _run_semantic_intent(
        self,
        dataframe: pd.DataFrame,
        schema_summary: dict[str, Any],
        case: SemanticEvaluationCase,
    ) -> SemanticIntentResult:
        variations: list[SemanticVariationResult] = []
        for index, (question, plan_payload) in enumerate(
            zip(case.question_variations, case.plan_payloads, strict=True),
            start=1,
        ):
            workflow_case = EvaluationCase(
                name=f"{case.intent_name}_variation_{index}",
                question=question,
                plan_payload=plan_payload,
            )
            workflow_result = self._run_case(
                dataframe,
                schema_summary,
                workflow_case,
            )
            missing_operations, missing_columns, missing_metrics = self._capability_gaps(
                workflow_result.generated_plan,
                case,
            )
            variations.append(
                SemanticVariationResult(
                    question=question,
                    workflow_result=workflow_result,
                    capability_success=not (
                        missing_operations or missing_columns or missing_metrics
                    ),
                    missing_operations=missing_operations,
                    missing_columns=missing_columns,
                    missing_metrics=missing_metrics,
                )
            )

        return SemanticIntentResult(
            intent_name=case.intent_name,
            description=case.description,
            expected_operations=case.expected_operations,
            expected_columns=case.expected_columns,
            expected_metrics=tuple(
                f"{metric.column}:{metric.function}"
                for metric in case.expected_metrics
            ),
            variations=tuple(variations),
        )

    def run(self, cases: Iterable[EvaluationCase] | None = None) -> EvaluationReport:
        evaluation_cases = (
            default_evaluation_cases() if cases is None else tuple(cases)
        )
        load_start = time.perf_counter()
        try:
            connector = SQLiteConnector(self.database_path, self.table_name)
            dataframe = connector.load()
            schema_summary = connector.get_schema()
        except SQLiteConnectorError:
            return EvaluationReport(
                database_path=str(self.database_path),
                table_name=self.table_name,
                data_load_duration_ms=self._milliseconds(load_start),
                results=(),
                setup_error="Evaluation database could not be loaded.",
            )

        results = tuple(
            self._run_case(dataframe, schema_summary, case)
            for case in evaluation_cases
        )
        return EvaluationReport(
            database_path=str(self.database_path),
            table_name=self.table_name,
            data_load_duration_ms=self._milliseconds(load_start),
            results=results,
        )

    def run_semantic_evaluation(
        self,
        cases: Iterable[SemanticEvaluationCase] | None = None,
    ) -> SemanticEvaluationReport:
        """Evaluate every question variation against intent-level capabilities."""
        semantic_cases = (
            default_semantic_evaluation_cases() if cases is None else tuple(cases)
        )
        load_start = time.perf_counter()
        try:
            connector = SQLiteConnector(self.database_path, self.table_name)
            dataframe = connector.load()
            schema_summary = connector.get_schema()
        except SQLiteConnectorError:
            return SemanticEvaluationReport(
                database_path=str(self.database_path),
                table_name=self.table_name,
                data_load_duration_ms=self._milliseconds(load_start),
                intents=(),
                setup_error="Evaluation database could not be loaded.",
            )

        intents = tuple(
            self._run_semantic_intent(dataframe, schema_summary, case)
            for case in semantic_cases
        )
        return SemanticEvaluationReport(
            database_path=str(self.database_path),
            table_name=self.table_name,
            data_load_duration_ms=self._milliseconds(load_start),
            intents=intents,
        )


def run_synthetic_evaluation(database_path: str | Path) -> EvaluationReport:
    """Create a synthetic SQLite dataset and evaluate every default business case."""
    created_database = create_synthetic_sqlite_database(database_path)
    return EvaluationRunner(created_database).run()


def run_semantic_evaluation(
    database_path: str | Path,
    cases: Iterable[SemanticEvaluationCase] | None = None,
) -> SemanticEvaluationReport:
    """Run semantic evaluation against an existing SQLite evaluation database."""
    return EvaluationRunner(database_path).run_semantic_evaluation(cases)


def run_synthetic_semantic_evaluation(
    database_path: str | Path,
) -> SemanticEvaluationReport:
    """Create synthetic data and run every default semantic evaluation intent."""
    created_database = create_synthetic_sqlite_database(database_path)
    return EvaluationRunner(created_database).run_semantic_evaluation()

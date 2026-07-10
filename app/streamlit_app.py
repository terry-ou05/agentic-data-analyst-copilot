import hashlib
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from plotly.graph_objects import Figure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import CsvLoadError, load_csv
from src.data.schema import build_schema_summary
from src.agents.code_generator import generate_analysis_code
from src.agents.insight_generator import (
    InsightGenerationResult,
    generate_insight,
)
from src.agents.plan_generator import generate_structured_plan
from src.agents.planner import generate_analysis_plan
from src.analysis.chart_generator import ChartGenerationError, generate_chart
from src.analysis.profiler import (
    AnalysisProfile,
    build_aggregated_result_summary,
    profile_analysis_result,
)
from src.analysis.visualization import VisualizationPlan, plan_visualization
from src.llm.client import get_llm_config
from src.runtime.code_guard import review_code_safety
from src.runtime.executor import AnalysisResult, execute_analysis_plan
from src.schemas.analysis_plan import (
    AnalysisPlan,
    OperationType,
    PlanValidationError,
    ValidatedAnalysisPlan,
    ValidationResult,
    create_validated_analysis_plan,
    validate_analysis_plan,
)
from src.utils.schema_signature import build_schema_signature


SAMPLE_DATASET = PROJECT_ROOT / "data" / "samples" / "sales_demo.csv"

V5_STATE_DEFAULTS = {
    "v5_plan": None,
    "v5_validation_result": None,
    "v5_execution_result": None,
    "v5_schema_signature": None,
    "v5_validated_plan": None,
    "v5_generation_error": "",
    "v5_dataset_identity": None,
    "v5_analysis_profile": None,
    "v5_visualization_plan": None,
    "v5_chart": None,
    "v5_chart_error": "",
    "v5_insight_result": None,
}


@dataclass(frozen=True)
class V5PlanPreparation:
    success: bool
    plan: AnalysisPlan | None
    validation_result: ValidationResult | None
    validated_plan: ValidatedAnalysisPlan | None
    schema_signature: str
    error: str = ""


@dataclass(frozen=True)
class V52AnalysisReport:
    profile: AnalysisProfile
    visualization_plan: VisualizationPlan | None
    chart: Figure | None
    chart_error: str
    insight_result: InsightGenerationResult


def initialize_v5_session_state(state: MutableMapping) -> None:
    for key, default_value in V5_STATE_DEFAULTS.items():
        if key not in state:
            state[key] = default_value


def clear_v5_analysis_state(state: MutableMapping) -> None:
    for key in (
        "v5_plan",
        "v5_validation_result",
        "v5_execution_result",
        "v5_schema_signature",
        "v5_validated_plan",
        "v5_analysis_profile",
        "v5_visualization_plan",
        "v5_chart",
        "v5_insight_result",
    ):
        state[key] = None
    state["v5_generation_error"] = ""
    state["v5_chart_error"] = ""


def synchronize_v5_dataset_state(
    state: MutableMapping,
    dataset_identity: str,
    current_schema_signature: str,
) -> bool:
    """Clear prior V5 state when the file or its bound schema changes."""
    initialize_v5_session_state(state)
    existing_identity = state.get("v5_dataset_identity")
    bound_signature = state.get("v5_schema_signature")
    dataset_changed = (
        existing_identity is not None and existing_identity != dataset_identity
    )
    schema_changed = (
        bound_signature is not None
        and bound_signature != current_schema_signature
    )

    if dataset_changed or schema_changed:
        clear_v5_analysis_state(state)

    state["v5_dataset_identity"] = dataset_identity
    return dataset_changed or schema_changed


def prepare_v5_plan(
    schema_summary: dict,
    user_question: str,
) -> V5PlanPreparation:
    """Generate and independently validate a V5 structured analysis plan."""
    schema_signature = build_schema_signature(schema_summary)
    generation_result = generate_structured_plan(schema_summary, user_question)
    if not generation_result.success:
        validation_result = (
            ValidationResult(False, generation_result.validation_errors)
            if generation_result.validation_errors
            else None
        )
        return V5PlanPreparation(
            success=False,
            plan=None,
            validation_result=validation_result,
            validated_plan=None,
            schema_signature=schema_signature,
            error=generation_result.error,
        )
    if generation_result.plan is None:
        return V5PlanPreparation(
            success=False,
            plan=None,
            validation_result=None,
            validated_plan=None,
            schema_signature=schema_signature,
            error="Structured plan generation returned no plan.",
        )

    plan = generation_result.plan
    validation_result = validate_analysis_plan(plan, schema_summary)
    if not validation_result.valid:
        return V5PlanPreparation(
            success=False,
            plan=plan,
            validation_result=validation_result,
            validated_plan=None,
            schema_signature=schema_signature,
            error="Structured analysis plan validation failed.",
        )

    try:
        validated_plan = create_validated_analysis_plan(plan, schema_summary)
    except PlanValidationError as exc:
        return V5PlanPreparation(
            success=False,
            plan=plan,
            validation_result=ValidationResult(False, exc.errors),
            validated_plan=None,
            schema_signature=schema_signature,
            error="Structured analysis plan validation failed.",
        )

    return V5PlanPreparation(
        success=True,
        plan=plan,
        validation_result=validation_result,
        validated_plan=validated_plan,
        schema_signature=schema_signature,
    )


def execute_v5_plan(
    dataframe: pd.DataFrame,
    validated_plan: ValidatedAnalysisPlan | None,
    expected_schema_signature: str | None,
) -> AnalysisResult:
    """Reject stale plans before calling the trusted V5 executor."""
    input_rows = len(dataframe) if isinstance(dataframe, pd.DataFrame) else 0
    if not isinstance(validated_plan, ValidatedAnalysisPlan):
        return AnalysisResult(
            success=False,
            dataframe=None,
            error_code="INVALID_PLAN_TYPE",
            message="Generate and validate a structured plan before running analysis.",
            executed_operations=(),
            input_rows=input_rows,
            output_rows=0,
            warnings=(),
        )

    current_signature = build_schema_signature(build_schema_summary(dataframe))
    if not expected_schema_signature or current_signature != expected_schema_signature:
        return AnalysisResult(
            success=False,
            dataframe=None,
            error_code="SCHEMA_CHANGED",
            message="Dataset changed. Please regenerate analysis plan.",
            executed_operations=(),
            input_rows=input_rows,
            output_rows=0,
            warnings=(),
        )

    return execute_analysis_plan(dataframe, validated_plan)


def analysis_plan_to_dict(plan: AnalysisPlan) -> dict:
    return {
        "version": plan.version,
        "goal": plan.goal,
        "operations": [
            {
                "operation": operation.operation_type.value,
                **dict(operation.parameters),
            }
            for operation in plan.operations
        ],
    }


def build_v52_report(
    dataframe: pd.DataFrame,
    validated_plan: ValidatedAnalysisPlan,
) -> V52AnalysisReport:
    """Build profile, chart, and metadata-only insight after safe execution."""
    profile = profile_analysis_result(dataframe)
    visualization_plan = plan_visualization(profile, validated_plan)
    chart = None
    chart_error = ""
    if visualization_plan is not None:
        try:
            chart = generate_chart(visualization_plan, dataframe)
        except ChartGenerationError as exc:
            chart_error = str(exc)

    result_summary = build_aggregated_result_summary(dataframe, validated_plan)
    insight_result = generate_insight(
        profile,
        result_summary,
        visualization_plan,
    )
    return V52AnalysisReport(
        profile=profile,
        visualization_plan=visualization_plan,
        chart=chart,
        chart_error=chart_error,
        insight_result=insight_result,
    )


def render_sidebar(summary: dict, dataset_source: str) -> None:
    st.sidebar.header("Project Stage")
    st.sidebar.write("V5.2 Visual Analysis Report")

    st.sidebar.header("Dataset Info")
    st.sidebar.write(f"Dataset Source: {dataset_source}")
    st.sidebar.metric("Rows", summary["number_of_rows"])
    st.sidebar.metric("Columns", summary["number_of_columns"])

    llm_config = get_llm_config()
    api_key_status = "Configured" if llm_config["api_key_configured"] else "Missing"
    st.sidebar.header("LLM Configuration")
    st.sidebar.write(f"Model: {llm_config['model']}")
    st.sidebar.write(f"Base URL: {llm_config['base_url']}")
    st.sidebar.write(f"API Key: {api_key_status}")

    st.sidebar.header("Next Step")
    st.sidebar.write("Current: safe execution, visualization, and insight")


def render_schema_summary(summary: dict) -> None:
    st.dataframe(summary["schema_table"], use_container_width=True)


def render_structured_plan(plan: AnalysisPlan) -> None:
    st.subheader("Structured Plan Preview")
    st.write("Analysis Goal:")
    st.write(plan.goal)
    st.write("Operations:")

    for index, operation in enumerate(plan.operations, start=1):
        parameters = operation.parameters
        with st.container(border=True):
            if operation.operation_type is OperationType.FILTER:
                st.write(f"{index}. Filter")
                st.write(
                    f"{parameters['column']} {parameters['operator']} "
                    f"{parameters['value']!r}"
                )
            elif operation.operation_type is OperationType.GROUPBY:
                st.write(f"{index}. Group By")
                st.write(", ".join(parameters["columns"]))
            elif operation.operation_type is OperationType.AGGREGATE:
                st.write(f"{index}. Aggregate")
                for metric in parameters["metrics"]:
                    st.write(
                        f"{metric['column']} ({metric['function']}) "
                        f"as {metric['alias']}"
                    )
            elif operation.operation_type is OperationType.TOP_N:
                st.write(f"{index}. Top N")
                direction = "ascending" if parameters["ascending"] else "descending"
                st.write(
                    f"Top {parameters['n']} by {parameters['sort_by']} ({direction})"
                )

    with st.expander("View structured plan JSON"):
        st.json(analysis_plan_to_dict(plan))


def render_v5_validation(validation_result: ValidationResult | None) -> None:
    if validation_result is None:
        return

    st.subheader("Validation Status")
    if validation_result.valid:
        st.success("✓ Plan validated")
        st.success("✓ Schema verified")
        st.success("✓ Operations allowed")
        st.info("Ready to execute")
        return

    st.error("Plan validation failed.")
    for error in validation_result.errors:
        st.warning(error)


def render_v5_result(result: AnalysisResult | None) -> None:
    if result is None:
        return

    st.subheader("Analysis Result")
    if not result.success or result.dataframe is None:
        st.error(result.message)
        return

    st.dataframe(result.dataframe, use_container_width=True)
    st.write("Result Metadata")
    input_col, output_col = st.columns(2)
    input_col.metric("Input rows", result.input_rows)
    output_col.metric("Output rows", result.output_rows)
    st.write("Executed operations:")
    for operation_name in result.executed_operations:
        st.write(f"- {operation_name}")
    for warning in result.warnings:
        st.warning(warning)


def render_v52_report(
    profile: AnalysisProfile | None,
    visualization_plan: VisualizationPlan | None,
    chart: Figure | None,
    chart_error: str,
    insight_result: InsightGenerationResult | None,
) -> None:
    if profile is None:
        return

    with st.expander("View result profile metadata"):
        st.json(profile.to_dict())

    st.subheader("Visualization")
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)
    elif chart_error:
        st.warning(f"Visualization unavailable: {chart_error}")
    elif visualization_plan is None:
        st.info("No supported chart is suitable for this analysis result.")

    st.subheader("AI Insight")
    if insight_result is None:
        st.info("No insight was generated.")
    elif insight_result.success:
        st.write(insight_result.insight)
    else:
        st.warning(f"AI insight unavailable: {insight_result.error}")


def render_v5_workflow(dataframe: pd.DataFrame, summary: dict) -> None:
    initialize_v5_session_state(st.session_state)
    st.subheader("V5 Structured Analysis")
    st.write(
        "Generate a schema-bound structured plan and run only trusted pandas "
        "operations. No generated Python is executed."
    )

    if summary["number_of_rows"] == 0:
        st.warning(
            "This CSV contains column headers but no data rows. "
            "Add data rows before generating a structured plan."
        )
        return

    user_question = st.text_area(
        "V5 business question",
        placeholder="Example: Which category has the highest online revenue?",
        height=100,
        key="v5_business_question",
    )

    if st.button("Generate Structured Plan", type="primary"):
        clear_v5_analysis_state(st.session_state)
        if not user_question.strip():
            st.session_state["v5_generation_error"] = (
                "Enter a business question before generating a structured plan."
            )
        else:
            with st.spinner("Generating structured plan..."):
                preparation = prepare_v5_plan(summary, user_question.strip())
            st.session_state["v5_plan"] = preparation.plan
            st.session_state["v5_validation_result"] = (
                preparation.validation_result
            )
            st.session_state["v5_validated_plan"] = preparation.validated_plan
            st.session_state["v5_schema_signature"] = (
                preparation.schema_signature if preparation.plan is not None else None
            )
            st.session_state["v5_generation_error"] = preparation.error

    generation_error = st.session_state["v5_generation_error"]
    if generation_error:
        st.error(generation_error)

    plan = st.session_state["v5_plan"]
    if isinstance(plan, AnalysisPlan):
        render_structured_plan(plan)

    validation_result = st.session_state["v5_validation_result"]
    render_v5_validation(validation_result)

    validated_plan = st.session_state["v5_validated_plan"]
    ready_to_execute = isinstance(validated_plan, ValidatedAnalysisPlan)
    if st.button("Run Analysis", disabled=not ready_to_execute):
        for key in (
            "v5_analysis_profile",
            "v5_visualization_plan",
            "v5_chart",
            "v5_insight_result",
        ):
            st.session_state[key] = None
        st.session_state["v5_chart_error"] = ""
        execution_result = execute_v5_plan(
            dataframe,
            validated_plan,
            st.session_state["v5_schema_signature"],
        )
        st.session_state["v5_execution_result"] = execution_result
        if execution_result.error_code == "SCHEMA_CHANGED":
            st.session_state["v5_validated_plan"] = None
            st.session_state["v5_validation_result"] = ValidationResult(
                False,
                (execution_result.message,),
            )
        elif execution_result.success and execution_result.dataframe is not None:
            with st.spinner("Building visualization and insight..."):
                report = build_v52_report(
                    execution_result.dataframe,
                    validated_plan,
                )
            st.session_state["v5_analysis_profile"] = report.profile
            st.session_state["v5_visualization_plan"] = report.visualization_plan
            st.session_state["v5_chart"] = report.chart
            st.session_state["v5_chart_error"] = report.chart_error
            st.session_state["v5_insight_result"] = report.insight_result

    if not ready_to_execute and plan is None:
        st.caption("Generate and validate a plan to enable Run Analysis.")

    render_v5_result(st.session_state["v5_execution_result"])
    render_v52_report(
        st.session_state["v5_analysis_profile"],
        st.session_state["v5_visualization_plan"],
        st.session_state["v5_chart"],
        st.session_state["v5_chart_error"],
        st.session_state["v5_insight_result"],
    )


def render_code_safety_review(code: str) -> None:
    safety_result = review_code_safety(code)
    static_check_status = (
        "blocked patterns detected"
        if safety_result["issues"]
        else "passed"
    )

    with st.container(border=True):
        st.subheader("Code Safety Review")
        st.write(f"Static check: {static_check_status}")
        st.write(f"Pattern severity: {safety_result['risk_level']}")
        st.caption(
            "Preview only: this code has not been executed. Static checks do not "
            "provide a complete security guarantee."
        )

        if safety_result["issues"]:
            st.write("Issues:")
            for issue in safety_result["issues"]:
                st.warning(issue)
        else:
            st.success("No known blocked string pattern was detected by this limited check.")


def render_analysis_planner(summary: dict) -> None:
    st.subheader("Ask a Business Question")
    st.write(
        "Generate an analysis plan and pandas/Plotly code preview from the "
        "current dataset schema."
    )
    st.info(
        "Code is generated for preview only and is not executed in V4."
    )

    if summary["number_of_rows"] == 0:
        st.warning(
            "This CSV contains column headers but no data rows. "
            "Add data rows before generating an analysis plan."
        )
        return

    user_question = st.text_area(
        "Business question",
        placeholder="Example: Which region and product category should we investigate for revenue decline?",
        height=100,
    )

    if st.button("Generate Analysis Plan and Code Preview", type="primary"):
        if not user_question.strip():
            st.warning("Enter a business question before generating an analysis plan.")
            return

        with st.spinner("Generating analysis plan..."):
            plan_result = generate_analysis_plan(summary, user_question.strip())

        if not plan_result.success:
            st.error(plan_result.error)
            st.info("Create a local .env file from .env.example and configure your LLM settings.")
            return

        st.subheader("Analysis Plan")
        st.markdown(plan_result.content)

        with st.spinner("Generating code preview..."):
            code_result = generate_analysis_code(
                summary,
                user_question.strip(),
                plan_result.content,
            )

        if not code_result["success"]:
            st.error(code_result["error"])
            st.info("The analysis plan was generated, but the code preview request failed.")
            return

        st.subheader("Generated Code Preview")
        st.caption("Code is generated for preview only and is not executed in V4.")
        st.write(f"Code Source: {code_result['source']}")
        if code_result["target_group_column"]:
            st.write(f"Target Group Column: {code_result['target_group_column']}")
        if code_result["metric_column"]:
            st.write(f"Metric Column: {code_result['metric_column']}")
        st.code(code_result["code"], language="python")
        render_code_safety_review(code_result["code"])


def main() -> None:
    st.set_page_config(
        page_title="Agentic Data Analyst Copilot",
        page_icon="A",
        layout="wide",
    )

    st.title("Agentic Data Analyst Copilot")
    st.caption(
        "Upload a CSV dataset, inspect its schema, generate a structured plan, "
        "run allowlisted operations, and review a visual analysis report."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    try:
        if uploaded_file is None:
            st.info("No file uploaded. Loading the sample dataset.")
            dataframe = load_csv(SAMPLE_DATASET)
            dataset_name = SAMPLE_DATASET.name
            dataset_source = "Sample CSV"
            dataset_identity = "sample-dataset"
        else:
            dataframe = load_csv(uploaded_file)
            dataset_name = uploaded_file.name
            dataset_source = "Uploaded CSV"
            raw_upload = uploaded_file.getvalue()
            if isinstance(raw_upload, memoryview):
                raw_upload = raw_upload.tobytes()
            dataset_identity = "uploaded:" + hashlib.sha256(raw_upload).hexdigest()
    except CsvLoadError as exc:
        st.error(str(exc))
        st.stop()

    summary = build_schema_summary(dataframe)
    current_schema_signature = build_schema_signature(summary)
    dataset_state_reset = synchronize_v5_dataset_state(
        st.session_state,
        dataset_identity,
        current_schema_signature,
    )
    render_sidebar(summary, dataset_source)

    st.subheader(f"Dataset: {dataset_name}")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Rows", summary["number_of_rows"])
    metric_col2.metric("Columns", summary["number_of_columns"])
    metric_col3.metric("Missing Values", summary["total_missing_values"])

    st.subheader("Data Preview")
    st.dataframe(dataframe.head(20), use_container_width=True)

    st.subheader("Schema Summary")
    render_schema_summary(summary)

    if dataset_state_reset:
        st.info("Dataset changed. Generate a new structured analysis plan.")

    render_v5_workflow(dataframe, summary)

    with st.expander("V4 Legacy Code Preview", expanded=False):
        render_analysis_planner(summary)


if __name__ == "__main__":
    main()

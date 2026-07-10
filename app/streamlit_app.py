from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import CsvLoadError, load_csv
from src.data.schema import build_schema_summary
from src.agents.code_generator import generate_analysis_code
from src.agents.planner import generate_analysis_plan
from src.llm.client import get_llm_config
from src.runtime.code_guard import review_code_safety


SAMPLE_DATASET = PROJECT_ROOT / "data" / "samples" / "sales_demo.csv"


def render_sidebar(summary: dict, dataset_source: str) -> None:
    st.sidebar.header("Project Stage")
    st.sidebar.write("V4 Code Safety Guard")

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
    st.sidebar.write("Future: reviewed execution path")


def render_schema_summary(summary: dict) -> None:
    st.dataframe(summary["schema_table"], use_container_width=True)


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
        "Upload a CSV dataset, inspect its schema, and prepare it for future "
        "agentic analysis workflows."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    try:
        if uploaded_file is None:
            st.info("No file uploaded. Loading the sample dataset.")
            dataframe = load_csv(SAMPLE_DATASET)
            dataset_name = SAMPLE_DATASET.name
            dataset_source = "Sample CSV"
        else:
            dataframe = load_csv(uploaded_file)
            dataset_name = uploaded_file.name
            dataset_source = "Uploaded CSV"
    except CsvLoadError as exc:
        st.error(str(exc))
        st.stop()

    summary = build_schema_summary(dataframe)
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

    render_analysis_planner(summary)


if __name__ == "__main__":
    main()

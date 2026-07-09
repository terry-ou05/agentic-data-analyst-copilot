from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_csv
from src.data.schema import build_schema_summary
from src.agents.planner import generate_analysis_plan
from src.llm.client import get_llm_config


SAMPLE_DATASET = PROJECT_ROOT / "data" / "samples" / "sales_demo.csv"


def render_sidebar(summary: dict, dataset_source: str) -> None:
    st.sidebar.header("Project Stage")
    st.sidebar.write("V2 LLM Analysis Planner")

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
    st.sidebar.write("Safe Code Generation (future)")


def render_schema_summary(summary: dict) -> None:
    st.dataframe(summary["schema_table"], use_container_width=True)


def render_analysis_planner(summary: dict) -> None:
    st.subheader("Ask a Business Question")
    st.write(
        "Generate a structured analysis plan from the current dataset schema. "
        "V2 does not generate or execute code."
    )

    user_question = st.text_area(
        "Business question",
        placeholder="Example: Which region and product category should we investigate for revenue decline?",
        height=100,
    )

    if st.button("Generate Analysis Plan", type="primary"):
        if not user_question.strip():
            st.warning("Enter a business question before generating an analysis plan.")
            return

        with st.spinner("Generating analysis plan..."):
            result = generate_analysis_plan(summary, user_question.strip())

        if result.success:
            st.markdown(result.content)
        else:
            st.error(result.error)
            st.info("Create a local .env file from .env.example and configure your LLM settings.")


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

    if uploaded_file is None:
        st.info("No file uploaded. Loading the sample dataset.")
        dataframe = load_csv(SAMPLE_DATASET)
        dataset_name = SAMPLE_DATASET.name
        dataset_source = "Sample CSV"
    else:
        dataframe = load_csv(uploaded_file)
        dataset_name = uploaded_file.name
        dataset_source = "Uploaded CSV"

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

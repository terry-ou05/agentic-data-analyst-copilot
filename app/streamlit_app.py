from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_csv
from src.data.schema import build_schema_summary


SAMPLE_DATASET = PROJECT_ROOT / "data" / "samples" / "sales_demo.csv"


def render_sidebar(summary: dict, dataset_source: str) -> None:
    st.sidebar.header("Project Stage")
    st.sidebar.write("V1.1 Data Upload & Schema Inspection")

    st.sidebar.header("Dataset Info")
    st.sidebar.write(f"Dataset Source: {dataset_source}")
    st.sidebar.metric("Rows", summary["number_of_rows"])
    st.sidebar.metric("Columns", summary["number_of_columns"])

    st.sidebar.header("Next Step")
    st.sidebar.write("LLM Analysis Planner")


def render_schema_summary(summary: dict) -> None:
    st.dataframe(summary["schema_table"], use_container_width=True)


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


if __name__ == "__main__":
    main()

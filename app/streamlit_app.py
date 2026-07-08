from pathlib import Path
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_csv
from src.data.schema import build_schema_summary


SAMPLE_DATASET = PROJECT_ROOT / "data" / "samples" / "sales_demo.csv"


def render_sidebar(summary: dict) -> None:
    st.sidebar.header("Project Stage")
    st.sidebar.write("V1 Data Upload")

    st.sidebar.header("Dataset Info")
    st.sidebar.metric("Rows", summary["number_of_rows"])
    st.sidebar.metric("Columns", summary["number_of_columns"])

    st.sidebar.header("Next Step")
    st.sidebar.write("LLM Analysis Planner")


def render_schema_summary(summary: dict) -> None:
    col1, col2 = st.columns(2)
    col1.metric("Number of rows", summary["number_of_rows"])
    col2.metric("Number of columns", summary["number_of_columns"])

    st.subheader("Column Names")
    st.write(summary["column_names"])

    st.subheader("Data Types")
    st.dataframe(summary["data_types"], use_container_width=True)

    st.subheader("Missing Values Per Column")
    st.dataframe(summary["missing_values"], use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Agentic Data Analyst Copilot",
        page_icon="A",
        layout="wide",
    )

    st.title("Agentic Data Analyst Copilot")
    st.caption("V1 local data upload foundation for a future LLM agent workflow.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        st.info("No file uploaded. Loading the sample dataset.")
        dataframe = load_csv(SAMPLE_DATASET)
        dataset_name = SAMPLE_DATASET.name
    else:
        dataframe = load_csv(uploaded_file)
        dataset_name = uploaded_file.name

    summary = build_schema_summary(dataframe)
    render_sidebar(summary)

    st.subheader(f"Dataset: {dataset_name}")

    st.subheader("Data Preview")
    st.dataframe(dataframe.head(20), use_container_width=True)

    st.subheader("Data Overview")
    render_schema_summary(summary)


if __name__ == "__main__":
    main()

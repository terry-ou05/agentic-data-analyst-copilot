# Agentic Data Analyst Copilot

Agentic Data Analyst Copilot is a local Streamlit application for exploring tabular datasets before adding an LLM agent workflow. The V1.1 version focuses on reliable CSV upload, sample-data loading, data preview, and schema inspection.

This project is intended as a job-search and resume showcase AI project. Future versions will add an LLM analysis planner, safe code generation, execution review, and evaluation metrics. V1.1 intentionally does not include LangGraph, DeepSeek, or a code executor.

## Features

- Upload a CSV file in the Streamlit UI
- Load `data/samples/sales_demo.csv` when no file is uploaded
- Preview the first 20 rows
- Show metric cards for rows, columns, and total missing values
- Show a schema summary table with column names, data types, missing values, and missing percentages
- Display project stage, dataset source, and next step in the sidebar

## Tech Stack

- Python
- Streamlit
- pandas

## Project Structure

```text
agentic-data-analyst-copilot/
+-- app/
|   +-- streamlit_app.py
+-- src/
|   +-- data/
|       +-- loader.py
|       +-- schema.py
+-- data/
|   +-- samples/
|       +-- sales_demo.csv
+-- .env.example
+-- .gitignore
+-- README.md
+-- requirements.txt
```

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

From the project root:

```powershell
streamlit run app/streamlit_app.py
```

## Current Stage

V1.1 Data Upload & Schema Inspection

## Next Step

V2 LLM Analysis Planner

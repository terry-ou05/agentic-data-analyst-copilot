# Agentic Data Analyst Copilot

Agentic Data Analyst Copilot is a local Streamlit application for exploring tabular datasets before adding an LLM agent workflow. The V1 version focuses on reliable CSV upload, sample-data loading, data preview, and schema summary.

This project is intended as a job-search and resume showcase AI project. Future versions will add an LLM analysis planner, safe code generation, execution review, and evaluation metrics. V1 intentionally does not include LangGraph, DeepSeek, or a code executor.

## Features

- Upload a CSV file in the Streamlit UI
- Load `data/samples/sales_demo.csv` when no file is uploaded
- Preview the first 20 rows
- Show number of rows and columns
- Show column names
- Show data types
- Show missing values per column
- Display current project stage and next step in the sidebar

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

V1 Data Upload

## Next Step

LLM Analysis Planner

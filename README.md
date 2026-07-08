# Agentic Data Analyst Copilot

Agentic Data Analyst Copilot is a local Streamlit application for exploring tabular datasets and preparing structured analysis plans. The V2 version adds an LLM Analysis Planner that turns a natural-language business question and dataset schema into a structured analysis plan.

This project is intended as a job-search and resume showcase AI project. V2 intentionally does not include LangGraph, generated pandas or Plotly code, code execution, code review, or a code executor.

## Features

- Upload a CSV file in the Streamlit UI
- Load `data/samples/sales_demo.csv` when no file is uploaded
- Preview the first 20 rows
- Show metric cards for rows, columns, and total missing values
- Show a schema summary table with column names, data types, missing values, and missing percentages
- Display project stage, dataset source, and next step in the sidebar
- Ask a natural-language business question
- Generate a structured LLM analysis plan from the dataset schema
- Keep V2 limited to planning only, with no generated code and no execution

## Tech Stack

- Python
- Streamlit
- pandas
- OpenAI-compatible API client

## Project Structure

```text
agentic-data-analyst-copilot/
+-- app/
|   +-- streamlit_app.py
+-- src/
|   +-- agents/
|       +-- planner.py
|   +-- data/
|       +-- loader.py
|       +-- schema.py
|   +-- llm/
|       +-- client.py
|       +-- prompts.py
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

Create a local `.env` file from `.env.example`:

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4.1-mini
```

For OpenAI-compatible providers, set `LLM_BASE_URL` and `LLM_MODEL` to the provider values. Do not commit `.env`.

## Run

From the project root:

```powershell
streamlit run app/streamlit_app.py
```

## Current Stage

V2 LLM Analysis Planner

## Next Step

Safe Code Generation (future)

## V2 Scope

V2 supports natural-language questions and returns an analysis plan with:

- Analysis Goal
- Relevant Columns
- Data Quality Checks
- Calculation Steps
- Suggested Visualizations
- Expected Output
- Business Interpretation Notes

V2 does not generate Python code and does not execute code.

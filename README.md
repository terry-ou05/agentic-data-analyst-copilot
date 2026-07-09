# Agentic Data Analyst Copilot

Agentic Data Analyst Copilot is a local Streamlit application for exploring tabular datasets, preparing structured analysis plans, and previewing pandas/Plotly analysis code. The V3 version generates an analysis plan first, then generates code preview text from the plan and dataset schema.

This project is intended as a job-search and resume showcase AI project. V3 intentionally does not include LangGraph, code execution, code review, sandboxing, or a code executor.

## Features

- Upload a CSV file in the Streamlit UI
- Load `data/samples/sales_demo.csv` when no file is uploaded
- Preview the first 20 rows
- Show metric cards for rows, columns, and total missing values
- Show a schema summary table with column names, data types, missing values, and missing percentages
- Display project stage, dataset source, and next step in the sidebar
- Ask a natural-language business question
- Generate a structured LLM analysis plan from the dataset schema
- Keep planner outputs grounded in real schema columns, with missing fields called out explicitly
- Generate pandas/Plotly code preview from the analysis plan and dataset schema
- Keep V3 limited to preview only, with no generated code execution

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
|       +-- code_generator.py
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
LLM_API_KEY=your_deepseek_api_key_here
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-pro
```

`.env.example` is only a template. Put real local credentials in `.env`, keep `.env` in the project root, and do not commit it. The app reads `.env` on startup and shows the active model and base URL in the sidebar without displaying the API key.

For other OpenAI-compatible providers, set `LLM_BASE_URL` and `LLM_MODEL` to the provider values.

## Run

From the project root:

```powershell
streamlit run app/streamlit_app.py
```

## Current Stage

V3 Code Generation Preview

## Next Step

V4 Code Review / Safety Guard

## Roadmap

- V1: CSV upload
- V1.1: UI and schema inspection
- V2: LLM Analysis Planner
- V2.1: Prompt Quality Polish
- V3: Code Generation Preview
- V4: Code Review / Safety Guard

## V2.1 Prompt Quality Improvements

V2.1 improves the planner prompt so that generated plans use exact dataset column names, avoid invented fields, and separate unavailable fields into a dedicated section. The planner remains analysis-plan only: it does not calculate final answers, generate pandas or Plotly code, execute code, or add LangGraph.

When a question mentions `category`, the prompt directs the planner to prefer a real `category` field if it exists and not substitute `product` unless the user explicitly asks about product.

## V3 Code Generation Preview

V3 adds a code preview step after analysis planning. The app asks the LLM to generate pandas and Plotly code that assumes the current dataset is already available as a DataFrame named `df`.

V3 also includes deterministic field selection so category-level questions use the `category` field instead of concrete product-name fields, including mapping "product category" to the existing `category` column.

V3 uses deterministic templates for simple groupby revenue questions to reduce LLM field-selection errors.

V3 routes simple highest/top revenue aggregation questions to deterministic templates before LLM fallback.

V3 displays the code generation source so users can distinguish deterministic templates from LLM-generated previews.

The generated code is displayed with syntax highlighting for review only. V3 does not execute generated code, does not add a code executor, does not read or save files from generated code, and does not add LangGraph or a complex agent workflow.

## V2 Scope

V2 supports natural-language questions and returns an analysis plan with:

- Analysis Goal
- Relevant Columns
- Missing or Unavailable Columns
- Data Quality Checks
- Calculation Steps
- Suggested Visualizations
- Expected Output
- Assumptions / Limitations
- Business Interpretation Notes

V2 does not generate Python code and does not execute code.

## V3 Scope

V3 supports natural-language questions and returns:

- Analysis Plan
- Generated Code Preview

Generated code is constrained to pandas and `plotly.express`, must use real schema column names, and is displayed with `st.code(..., language="python")`. The app does not execute the generated code in V3.

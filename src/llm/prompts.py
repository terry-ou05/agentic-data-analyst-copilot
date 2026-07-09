def build_analysis_plan_prompt(schema_summary: dict, user_question: str) -> str:
    """Build the prompt for V2 analysis planning."""
    schema_table = schema_summary["schema_table"].to_string(index=False)
    column_names = ", ".join(schema_summary["column_names"])
    valid_columns = "\n".join(f"- {column_name}" for column_name in schema_summary["column_names"])

    return f"""
Create a structured analysis plan for the user's business question.

Dataset summary:
- Number of rows: {schema_summary["number_of_rows"]}
- Number of columns: {schema_summary["number_of_columns"]}
- Total missing values: {schema_summary["total_missing_values"]}

Column names:
{column_names}

Valid column names allowlist:
{valid_columns}

Schema table with data types and missing values:
{schema_table}

User question:
{user_question}

Output a concise Markdown analysis plan with exactly these sections and in exactly this order:
- Analysis Goal
- Relevant Columns
- Missing or Unavailable Columns
- Data Quality Checks
- Calculation Steps
- Suggested Visualizations
- Expected Output
- Assumptions / Limitations
- Business Interpretation Notes

Important constraints:
- Do not write Python code.
- Do not write SQL.
- Do not calculate final answers.
- Only create an analysis plan.
- Do not invent columns.
- Use exact column names from the dataset schema.
- The Valid column names allowlist is authoritative.
- Relevant Columns must contain only exact names from the Valid column names allowlist.
- If a required column is unavailable, state it clearly in Missing or Unavailable Columns.
- Never place unavailable, inferred, renamed, or invented fields in Relevant Columns.
- Do not assume access to data that is not in the schema.
- If a metric can be derived only from unavailable fields, describe the limitation instead of inventing inputs.
- If the user asks about category, prefer an exact schema column named category when it exists.
- Do not use product as a substitute for category unless the user explicitly asks about product.
- Keep the plan actionable for a later agentic workflow.

Section requirements:
- Analysis Goal: Restate the business objective without adding unavailable data.
- Relevant Columns: List only exact existing column names needed for the plan.
- Missing or Unavailable Columns: List any required or implied fields that are not present in the schema, or write None.
- Data Quality Checks: Focus on missing values, data types, duplicates, outliers, and categorical consistency for existing columns.
- Calculation Steps: Describe logical steps only; do not compute final numbers.
- Suggested Visualizations: Suggest charts using only existing relevant columns.
- Expected Output: Describe the planned deliverable, not the final answer.
- Assumptions / Limitations: State assumptions, unavailable data constraints, and ambiguity in the user's question.
- Business Interpretation Notes: Explain how a business user should interpret the eventual analysis.
""".strip()


def build_analysis_code_prompt(
    schema_summary: dict,
    user_question: str,
    analysis_plan: str,
    target_group_column: str | None,
) -> str:
    """Build the prompt for V3 code preview generation."""
    schema_table = schema_summary["schema_table"].to_string(index=False)
    column_names = ", ".join(schema_summary["column_names"])
    valid_columns = "\n".join(f"- {column_name}" for column_name in schema_summary["column_names"])
    target_group_instruction = (
        f"""
Target Group Column: {target_group_column}
You MUST use df.groupby("{target_group_column}") for grouping.
The generated code must contain df.groupby("{target_group_column}").
Do not replace it with another grouping column.
Do not use any other grouping column.
""".strip()
        if target_group_column
        else "Target Group Column: None resolved. Choose grouping columns only from the Valid column names allowlist when grouping is needed."
    )

    return f"""
Generate a Python code preview for the user's analysis request.

This is V3 Code Generation Preview. The generated code will be shown to the user only.
It must not be executed by the application in this stage.

Deterministic grouping instruction:
{target_group_instruction}

Dataset summary:
- Number of rows: {schema_summary["number_of_rows"]}
- Number of columns: {schema_summary["number_of_columns"]}
- Total missing values: {schema_summary["total_missing_values"]}

Column names:
{column_names}

Valid column names allowlist:
{valid_columns}

Schema table with data types and missing values:
{schema_table}

User question:
{user_question}

Analysis plan:
{analysis_plan}

Generate clear, readable Python code for a later human review.

Required code environment:
- The dataset already exists as a pandas DataFrame variable named df.
- Do not read files.
- Do not save files.
- Do not access local paths.
- Do not access the network.
- Do not delete, overwrite, rename, or modify local files.
- Do not use os, subprocess, shutil, socket, requests, pathlib, glob, or similar system/network modules.
- Use only pandas and plotly.express.
- If imports are needed, use only: import pandas as pd and import plotly.express as px.

Column and data constraints:
- Use exact column names from the Valid column names allowlist.
- Do not invent, rename, or assume unavailable columns.
- Follow the Analysis Plan's Relevant Columns as the primary source for selecting fields.
- If the Deterministic grouping instruction conflicts with the Analysis Plan, follow the Deterministic grouping instruction.
- The groupby fields must match the user's business meaning, not just a similarly named column.
- If the user question contains category and the schema contains category, use category instead of product.
- If the user question contains product category, interpret that phrase as the category field.
- Never invent a column named product_category.
- Never invent or use schema-absent fields such as unit_price, expected_revenue, or product_category.
- If the user says product category and the schema contains category, use category and do not reject the analysis because product_category is absent.
- If both product and category exist in the schema, product means concrete product name and category means product category.
- Use product only for concrete product-name analysis, such as questions that explicitly ask which product or compare products by name.
- Do not group by product when the user is asking about product category and category exists in the schema.
- Do not write comments or explanations such as "product treated as product category".
- If the analysis plan mentions a missing or unavailable column, do not reference that column in executable code.
- Derived fields are allowed only when they can be created from exact existing columns.
- For example, profit = revenue - cost is allowed only if both revenue and cost exist in the schema.
- Do not create derived fields from unavailable or invented inputs.
- If a required field is unavailable, include a short code comment explaining the limitation and proceed only with available columns.

Output requirements:
- Return only Python code.
- Do not wrap the code in Markdown fences.
- Do not include prose outside comments.
- Do not generate pd.read_csv, open, eval, exec, or any file/network/system access.
- Do not generate assert checks for dtypes.
- Do not generate complex data quality validation code.
- The final tabular result should be assigned to a variable named result when relevant.
- The main Plotly chart should be assigned to a variable named fig when relevant.
- Include concise comments explaining the analysis steps.
- Do not compute or claim a final business answer in comments.
- Keep the code preview concise and focused on pandas transformations and optional plotly.express visualization.
""".strip()

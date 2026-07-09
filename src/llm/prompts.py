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

def build_analysis_plan_prompt(schema_summary: dict, user_question: str) -> str:
    """Build the prompt for V2 analysis planning."""
    schema_table = schema_summary["schema_table"].to_string(index=False)
    column_names = ", ".join(schema_summary["column_names"])

    return f"""
Create a structured analysis plan for the user's business question.

Dataset summary:
- Number of rows: {schema_summary["number_of_rows"]}
- Number of columns: {schema_summary["number_of_columns"]}
- Total missing values: {schema_summary["total_missing_values"]}

Column names:
{column_names}

Schema table with data types and missing values:
{schema_table}

User question:
{user_question}

Output a concise Markdown analysis plan with exactly these sections:
- Analysis Goal
- Relevant Columns
- Data Quality Checks
- Calculation Steps
- Suggested Visualizations
- Expected Output
- Business Interpretation Notes

Important constraints:
- Do not write Python code.
- Do not write SQL.
- Do not assume access to data that is not in the schema.
- Keep the plan actionable for a later agentic workflow.
""".strip()

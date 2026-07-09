from src.llm.client import generate_chat_completion
from src.llm.prompts import build_analysis_code_prompt


def get_schema_column_names(schema_summary: dict) -> list[str]:
    """Extract column names from the current schema summary structure."""
    column_names = schema_summary.get("column_names")

    if isinstance(column_names, list):
        return [str(column) for column in column_names]

    schema_table = schema_summary.get("schema_table")

    if hasattr(schema_table, "columns") and "column name" in schema_table.columns:
        return [str(column) for column in schema_table["column name"].tolist()]

    if isinstance(schema_table, list):
        columns = []
        for row in schema_table:
            if isinstance(row, dict) and "column name" in row:
                columns.append(str(row["column name"]))
        return columns

    if isinstance(schema_table, dict) and "column name" in schema_table:
        values = schema_table["column name"]
        if isinstance(values, list):
            return [str(column) for column in values]

    return []


def resolve_target_group_column(schema_summary: dict, user_question: str) -> str | None:
    """Resolve the group-by column using deterministic rules."""
    q = user_question.lower()
    columns = get_schema_column_names(schema_summary)
    lower_to_original = {column.lower(): column for column in columns}

    if "product category" in q and "category" in lower_to_original:
        return lower_to_original["category"]

    if "category" in q and "category" in lower_to_original:
        return lower_to_original["category"]

    if "region" in q and "region" in lower_to_original:
        return lower_to_original["region"]

    if "product" in q and "category" not in q and "product" in lower_to_original:
        return lower_to_original["product"]

    return None


def resolve_metric_column(schema_summary: dict, user_question: str) -> str | None:
    """Resolve the metric column using deterministic rules."""
    q = user_question.lower()
    columns = get_schema_column_names(schema_summary)
    lower_to_original = {column.lower(): column for column in columns}

    if "revenue" in q and "revenue" in lower_to_original:
        return lower_to_original["revenue"]

    if "sales" in q and "revenue" in lower_to_original:
        return lower_to_original["revenue"]

    return None


def has_ranking_intent(user_question: str) -> bool:
    """Detect highest/top ranking intent."""
    q = user_question.lower()
    return any(word in q for word in ["highest", "top", "largest", "most"])


def is_simple_revenue_aggregation_question(
    schema_summary: dict,
    user_question: str,
) -> bool:
    """Return True for simple highest/top revenue by group questions."""
    target_group_column = resolve_target_group_column(schema_summary, user_question)
    metric_column = resolve_metric_column(schema_summary, user_question)

    return bool(
        target_group_column
        and metric_column
        and has_ranking_intent(user_question)
    )


def build_groupby_sum_code(target_group_column: str, metric_column: str) -> str:
    """Build deterministic pandas/Plotly code for simple group-by sum analysis."""
    variable_name = f"{metric_column}_by_{target_group_column}"
    group_label = target_group_column.replace("_", " ").title()
    metric_label = metric_column.replace("_", " ").title()
    chart_title = f"Total {metric_label} by {group_label}"

    return f'''import pandas as pd
import plotly.express as px

# Aggregate total {metric_column} by {target_group_column}
{variable_name} = df.groupby("{target_group_column}", as_index=False)["{metric_column}"].sum()
{variable_name} = {variable_name}.sort_values("{metric_column}", ascending=False)

# Store the sorted result
result = {variable_name}

# Create a horizontal bar chart
fig = px.bar(
    {variable_name},
    x="{metric_column}",
    y="{target_group_column}",
    orientation="h",
    title="{chart_title}",
    labels={{"{metric_column}": "{metric_label}", "{target_group_column}": "{group_label}"}},
    text="{metric_column}",
)

fig.update_layout(yaxis={{"categoryorder": "total ascending"}})
'''


def generate_analysis_code(
    schema_summary: dict,
    user_question: str,
    analysis_plan: str,
) -> dict:
    """Generate pandas/Plotly code text for preview without executing it."""
    target_group_column = resolve_target_group_column(schema_summary, user_question)
    metric_column = resolve_metric_column(schema_summary, user_question)

    if is_simple_revenue_aggregation_question(schema_summary, user_question):
        return {
            "success": True,
            "code": build_groupby_sum_code(target_group_column, metric_column),
            "error": "",
            "source": "deterministic_template",
            "target_group_column": target_group_column,
            "metric_column": metric_column,
        }

    prompt = build_analysis_code_prompt(
        schema_summary,
        user_question,
        analysis_plan,
        target_group_column,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You generate Python code previews for data analysis. "
                "Return code text only. Do not execute code, read files, "
                "access networks, or use system modules."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    result = generate_chat_completion(messages)

    return {
        "success": result.success,
        "code": result.content,
        "error": result.error,
        "source": "llm_generated",
        "target_group_column": target_group_column,
        "metric_column": metric_column,
    }
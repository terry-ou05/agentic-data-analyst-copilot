import ast

from src.llm.client import generate_chat_completion
from src.llm.prompts import build_analysis_code_prompt


FENCE_MARKERS = {"```", "```python"}
PLOTLY_COLUMN_ARGUMENTS = {
    "x",
    "y",
    "color",
    "facet_row",
    "facet_col",
    "hover_name",
    "hover_data",
    "animation_frame",
    "animation_group",
    "text",
}


def normalize_generated_code(content: str) -> str:
    """Normalize a plain Python response or one explicit Markdown code block."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Generated code is empty.")

    stripped_content = content.strip()
    lines = stripped_content.splitlines()
    fence_lines = [
        (index, line.strip().lower())
        for index, line in enumerate(lines)
        if line.strip().startswith("```")
    ]

    if not fence_lines:
        return stripped_content

    if len(fence_lines) != 2:
        raise ValueError("Generated code must contain one complete Markdown code block.")

    (opening_index, opening_marker), (closing_index, closing_marker) = fence_lines
    if opening_marker not in FENCE_MARKERS or closing_marker != "```":
        raise ValueError("Generated code contains an unsupported Markdown fence.")

    if opening_index >= closing_index:
        raise ValueError("Generated code contains an invalid Markdown code block.")

    normalized_code = "\n".join(lines[opening_index + 1:closing_index]).strip()
    if not normalized_code:
        raise ValueError("Generated code block is empty.")

    return normalized_code


def _extract_literal_columns(node: ast.AST) -> tuple[list[str], bool]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value], False

    if isinstance(node, (ast.List, ast.Tuple)):
        columns = []
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                return [], True
            columns.append(element.value)
        return columns, False

    return [], True


def _get_call_argument(
    node: ast.Call,
    keyword_name: str,
    positional_index: int | None = None,
) -> ast.AST | None:
    for keyword in node.keywords:
        if keyword.arg == keyword_name:
            return keyword.value

    if positional_index is not None and len(node.args) > positional_index:
        return node.args[positional_index]

    return None


def _is_dataframe_column_subscript(node: ast.Subscript) -> bool:
    if isinstance(node.value, ast.Name) and node.value.id == "df":
        return True

    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
        return (
            node.value.func.attr == "groupby"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "df"
        )

    return False


def _is_boolean_row_filter(node: ast.AST) -> bool:
    if isinstance(node, (ast.Compare, ast.BoolOp)):
        return True

    if isinstance(node, ast.BinOp) and isinstance(
        node.op,
        (ast.BitAnd, ast.BitOr, ast.BitXor),
    ):
        return True

    return isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Invert, ast.Not))


class _DataFrameColumnVisitor(ast.NodeVisitor):
    def __init__(self, schema_columns: list[str]) -> None:
        self.known_columns = set(schema_columns)
        self.missing_columns: set[str] = set()
        self.has_unverifiable_reference = False

    def _record_columns(self, value: ast.AST | None) -> None:
        if value is None:
            return

        if isinstance(value, ast.Constant) and value.value is None:
            return

        columns, is_dynamic = _extract_literal_columns(value)
        if is_dynamic:
            self.has_unverifiable_reference = True
            return

        for column in columns:
            if column not in self.known_columns:
                self.missing_columns.add(column)

    def _record_assigned_columns(self, target: ast.AST) -> None:
        if not isinstance(target, ast.Subscript):
            return

        if not isinstance(target.value, ast.Name) or target.value.id != "df":
            return

        columns, is_dynamic = _extract_literal_columns(target.slice)
        if is_dynamic:
            self.has_unverifiable_reference = True
            return

        self.known_columns.update(columns)

    def _record_rename_mapping(self, value: ast.AST | None) -> None:
        if not isinstance(value, ast.Dict):
            self.has_unverifiable_reference = True
            return

        for key in value.keys:
            if key is None:
                self.has_unverifiable_reference = True
            else:
                self._record_columns(key)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._record_assigned_columns(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self._record_assigned_columns(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Subscript) and _is_dataframe_column_subscript(node.target):
            self._record_columns(node.target.slice)
        self.visit(node.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Load):
            if (
                isinstance(node.value, ast.Name)
                and node.value.id == "df"
                and _is_boolean_row_filter(node.slice)
            ):
                pass
            elif _is_dataframe_column_subscript(node):
                self._record_columns(node.slice)
            elif (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "df"
                and node.value.attr == "loc"
                and isinstance(node.slice, ast.Tuple)
                and len(node.slice.elts) >= 2
            ):
                column_selector = node.slice.elts[1]
                if not isinstance(column_selector, ast.Slice):
                    self._record_columns(column_selector)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in {"groupby", "sort_values"}:
                self._record_columns(_get_call_argument(node, "by", 0))
            elif method_name == "dropna":
                self._record_columns(_get_call_argument(node, "subset"))
            elif method_name == "pivot_table":
                self._record_columns(_get_call_argument(node, "index", 1))
                self._record_columns(_get_call_argument(node, "columns", 2))
                self._record_columns(_get_call_argument(node, "values", 0))
            elif method_name == "rename":
                rename_mapping = _get_call_argument(node, "columns")
                if rename_mapping is not None:
                    self._record_rename_mapping(rename_mapping)

            if isinstance(node.func.value, ast.Name) and node.func.value.id == "px":
                data_frame = _get_call_argument(node, "data_frame", 0)
                if data_frame is not None:
                    for keyword in node.keywords:
                        if keyword.arg in PLOTLY_COLUMN_ARGUMENTS:
                            self._record_columns(keyword.value)

        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "df"
        ):
            self.has_unverifiable_reference = True

        self.generic_visit(node)


def validate_generated_code(code: str, schema_summary: dict) -> dict:
    """Normalize code, validate Python syntax, and check common column references."""
    try:
        normalized_code = normalize_generated_code(code)
    except ValueError as exc:
        return {
            "success": False,
            "code": "",
            "error": str(exc),
            "missing_columns": [],
            "unverifiable": False,
        }

    try:
        syntax_tree = ast.parse(normalized_code)
    except SyntaxError as exc:
        line_number = exc.lineno or 1
        return {
            "success": False,
            "code": "",
            "error": f"Generated code is not valid Python (SyntaxError at line {line_number}).",
            "missing_columns": [],
            "unverifiable": False,
        }

    meaningful_statements = [
        statement
        for statement in syntax_tree.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]
    if not meaningful_statements:
        return {
            "success": False,
            "code": "",
            "error": "Generated content does not contain executable Python statements.",
            "missing_columns": [],
            "unverifiable": False,
        }

    visitor = _DataFrameColumnVisitor(get_schema_column_names(schema_summary))
    visitor.visit(syntax_tree)

    if visitor.missing_columns:
        missing_columns = sorted(visitor.missing_columns)
        return {
            "success": False,
            "code": "",
            "error": (
                "Generated code references columns that are not in the dataset schema: "
                + ", ".join(missing_columns)
            ),
            "missing_columns": missing_columns,
            "unverifiable": visitor.has_unverifiable_reference,
        }

    if visitor.has_unverifiable_reference:
        return {
            "success": False,
            "code": "",
            "error": (
                "Generated code contains dynamic DataFrame column references "
                "that cannot be statically verified."
            ),
            "missing_columns": [],
            "unverifiable": True,
        }

    return {
        "success": True,
        "code": normalized_code,
        "error": "",
        "missing_columns": [],
        "unverifiable": False,
    }


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


def _build_validated_code_result(
    code: str,
    schema_summary: dict,
    source: str,
    target_group_column: str | None,
    metric_column: str | None,
) -> dict:
    validation_result = validate_generated_code(code, schema_summary)
    return {
        "success": validation_result["success"],
        "code": validation_result["code"],
        "error": validation_result["error"],
        "source": source,
        "target_group_column": target_group_column,
        "metric_column": metric_column,
    }


def generate_analysis_code(
    schema_summary: dict,
    user_question: str,
    analysis_plan: str,
) -> dict:
    """Generate pandas/Plotly code text for preview without executing it."""
    target_group_column = resolve_target_group_column(schema_summary, user_question)
    metric_column = resolve_metric_column(schema_summary, user_question)

    if is_simple_revenue_aggregation_question(schema_summary, user_question):
        if target_group_column is None or metric_column is None:
            return {
                "success": False,
                "code": "",
                "error": "The deterministic code route could not resolve required columns.",
                "source": "deterministic_template",
                "target_group_column": target_group_column,
                "metric_column": metric_column,
            }

        return _build_validated_code_result(
            build_groupby_sum_code(target_group_column, metric_column),
            schema_summary,
            "deterministic_template",
            target_group_column,
            metric_column,
        )

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

    if not result.success:
        return {
            "success": False,
            "code": "",
            "error": result.error,
            "source": "llm_generated",
            "target_group_column": target_group_column,
            "metric_column": metric_column,
        }

    return _build_validated_code_result(
        result.content,
        schema_summary,
        "llm_generated",
        target_group_column,
        metric_column,
    )

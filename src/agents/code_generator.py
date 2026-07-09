from src.llm.client import LLMResult, generate_chat_completion
from src.llm.prompts import build_analysis_code_prompt


def _build_field_selection_hint(schema_summary: dict, user_question: str) -> str:
    column_names = set(schema_summary["column_names"])
    normalized_question = user_question.lower()

    if "category" in column_names:
        if "product category" in normalized_question:
            return (
                'Field Selection Hint: The user is asking about product category. '
                'Use the exact column "category" for grouping. Do not use "product".'
            )
        if "category" in normalized_question:
            return (
                'Field Selection Hint: The user is asking about category. '
                'Prefer the exact column "category" for grouping.'
            )

    if "product" in column_names and (
        "which product" in normalized_question or "top product" in normalized_question
    ):
        return (
            'Field Selection Hint: The user is asking about specific products. '
            'The exact column "product" may be used for product-name grouping.'
        )

    return "Field Selection Hint: Use the Analysis Plan's Relevant Columns and the schema semantics to choose grouping fields."


def generate_analysis_code(
    schema_summary: dict,
    user_question: str,
    analysis_plan: str,
) -> LLMResult:
    """Generate pandas/Plotly code text for preview without executing it."""
    field_selection_hint = _build_field_selection_hint(schema_summary, user_question)
    prompt = build_analysis_code_prompt(
        schema_summary,
        user_question,
        analysis_plan,
        field_selection_hint,
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
    return generate_chat_completion(messages)

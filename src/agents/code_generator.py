from src.llm.client import LLMResult, generate_chat_completion
from src.llm.prompts import build_analysis_code_prompt


def generate_analysis_code(
    schema_summary: dict,
    user_question: str,
    analysis_plan: str,
) -> LLMResult:
    """Generate pandas/Plotly code text for preview without executing it."""
    prompt = build_analysis_code_prompt(schema_summary, user_question, analysis_plan)
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

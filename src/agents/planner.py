from src.llm.client import LLMResult, generate_chat_completion
from src.llm.prompts import build_analysis_plan_prompt


def generate_analysis_plan(schema_summary: dict, user_question: str) -> LLMResult:
    """Generate a structured analysis plan without generating or executing code."""
    prompt = build_analysis_plan_prompt(schema_summary, user_question)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful data analyst. Create analysis plans only. "
                "Do not generate Python code, SQL, or executable instructions."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return generate_chat_completion(messages)

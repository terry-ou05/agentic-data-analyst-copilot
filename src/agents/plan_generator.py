import json
from dataclasses import dataclass

from src.llm.client import generate_chat_completion
from src.llm.prompts import build_structured_analysis_plan_prompt
from src.schemas.analysis_plan import (
    AnalysisPlan,
    PlanParseError,
    parse_analysis_plan,
)


JSON_FENCE_MARKERS = {"```", "```json"}


@dataclass(frozen=True)
class PlanGenerationResult:
    success: bool
    plan: AnalysisPlan | None = None
    error: str = ""
    validation_errors: tuple[str, ...] = ()


def _normalize_json_response(content: str) -> str:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Structured analysis plan response is empty.")

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
        raise ValueError("Structured analysis plan must contain one complete JSON block.")

    (opening_index, opening_marker), (closing_index, closing_marker) = fence_lines
    if opening_marker not in JSON_FENCE_MARKERS or closing_marker != "```":
        raise ValueError("Structured analysis plan contains an unsupported JSON fence.")
    if opening_index >= closing_index:
        raise ValueError("Structured analysis plan contains an invalid JSON block.")

    normalized = "\n".join(lines[opening_index + 1:closing_index]).strip()
    if not normalized:
        raise ValueError("Structured analysis plan JSON block is empty.")
    return normalized


def generate_structured_plan(
    schema_summary: dict,
    user_question: str,
) -> PlanGenerationResult:
    """Generate and parse a structured analysis plan without execution."""
    prompt = build_structured_analysis_plan_prompt(schema_summary, user_question)
    messages = [
        {
            "role": "system",
            "content": (
                "Return one JSON analysis plan only. Never return Python, pandas "
                "code, SQL, or operations outside the provided contract."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    llm_result = generate_chat_completion(messages)
    if not llm_result.success:
        return PlanGenerationResult(success=False, error=llm_result.error)

    try:
        normalized_content = _normalize_json_response(llm_result.content)
        payload = json.loads(normalized_content)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return PlanGenerationResult(
            success=False,
            error=f"Structured analysis plan JSON is invalid ({exc.__class__.__name__}).",
        )

    try:
        plan = parse_analysis_plan(payload)
    except PlanParseError as exc:
        return PlanGenerationResult(success=False, error=str(exc))

    return PlanGenerationResult(success=True, plan=plan)

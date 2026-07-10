import pandas as pd

from src.agents import code_generator
from src.data.schema import build_schema_summary
from src.llm.client import LLMResult


def _summary(**columns) -> dict:
    return build_schema_summary(pd.DataFrame(columns))


def test_missing_category_from_llm_fails(monkeypatch) -> None:
    schema = _summary(product=["Laptop"], revenue=[10])
    monkeypatch.setattr(
        code_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(success=True, content='result = df["category"]'),
    )

    result = code_generator.generate_analysis_code(
        schema,
        "Compare category revenue",
        "Use available columns.",
    )

    assert result["success"] is False
    assert "category" in result["error"]


def test_valid_category_groupby_passes() -> None:
    schema = _summary(category=["Computer"], revenue=[10])
    code = 'result = df.groupby("category")["revenue"].sum()'

    result = code_generator.validate_generated_code(code, schema)

    assert result["success"] is True


def test_nonexistent_category_variant_fails() -> None:
    schema = _summary(category=["Computer"], revenue=[10])

    result = code_generator.validate_generated_code(
        'result = df["category_does_not_exist"].sum()',
        schema,
    )

    assert result["success"] is False
    assert result["missing_columns"] == ["category_does_not_exist"]


def test_deterministic_template_is_validated(monkeypatch) -> None:
    schema = _summary(category=["Computer"], product=["Laptop"], revenue=[10])
    original_validator = code_generator.validate_generated_code
    calls = []

    def tracking_validator(code, schema_summary):
        calls.append(code)
        return original_validator(code, schema_summary)

    monkeypatch.setattr(code_generator, "validate_generated_code", tracking_validator)
    monkeypatch.setattr(
        code_generator,
        "generate_chat_completion",
        lambda messages: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )

    result = code_generator.generate_analysis_code(
        schema,
        "Which product category has the highest revenue?",
        "Aggregate revenue by category.",
    )

    assert result["success"] is True
    assert result["source"] == "deterministic_template"
    assert len(calls) == 1


def test_markdown_fenced_category_code_is_cleaned_and_validated(monkeypatch) -> None:
    schema = _summary(category=["Computer"], revenue=[10])
    monkeypatch.setattr(
        code_generator,
        "generate_chat_completion",
        lambda messages: LLMResult(
            success=True,
            content='```python\nresult = df.groupby("category")["revenue"].sum()\n```',
        ),
    )

    result = code_generator.generate_analysis_code(
        schema,
        "Compare category revenue",
        "Use category and revenue.",
    )

    assert result["success"] is True
    assert result["code"] == 'result = df.groupby("category")["revenue"].sum()'

import pytest

from src.agents.code_generator import (
    normalize_generated_code,
    validate_generated_code,
)


def _schema(*columns: str) -> dict:
    return {"column_names": list(columns)}


def test_plain_python_is_preserved() -> None:
    code = 'result = df["revenue"].sum()'

    assert normalize_generated_code(code) == code


@pytest.mark.parametrize(
    "content",
    [
        '```python\nresult = df["revenue"].sum()\n```',
        '```\nresult = df["revenue"].sum()\n```',
        'Here is the code:\n```python\nresult = df["revenue"].sum()\n```\nDone.',
    ],
)
def test_markdown_fenced_code_is_extracted(content) -> None:
    result = validate_generated_code(content, _schema("revenue"))

    assert result["success"] is True
    assert result["code"] == 'result = df["revenue"].sum()'


@pytest.mark.parametrize(
    "content",
    [
        "```python\n```",
        "```\n```",
        '```python\nresult = df["revenue"].sum()',
    ],
)
def test_empty_or_unclosed_fence_fails(content) -> None:
    result = validate_generated_code(content, _schema("revenue"))

    assert result["success"] is False


def test_explanation_only_fails() -> None:
    result = validate_generated_code(
        "Here is the analysis code you requested.",
        _schema("revenue"),
    )

    assert result["success"] is False
    assert "valid Python" in result["error"]


def test_string_literal_explanation_is_not_treated_as_code() -> None:
    result = validate_generated_code('"This is only an explanation."', _schema("revenue"))

    assert result["success"] is False
    assert "does not contain executable Python statements" in result["error"]


def test_invalid_python_syntax_fails() -> None:
    result = validate_generated_code("if True print('broken')", _schema("revenue"))

    assert result["success"] is False
    assert "SyntaxError" in result["error"]


def test_common_column_reference_forms_pass() -> None:
    code = '''selected = df[["category", "revenue"]]
grouped = df.groupby(["category", "region"])["revenue"].sum()
sorted_result = selected.sort_values(by="revenue")
clean = df.dropna(subset=["revenue"])
fig = px.bar(df, x="category", y="revenue")
'''

    result = validate_generated_code(
        code,
        _schema("category", "region", "revenue"),
    )

    assert result["success"] is True


def test_chinese_spaces_and_special_column_names_pass() -> None:
    code = 'result = df[["客户 类型", "收入(元)"]]\nfig = px.bar(df, x="客户 类型", y="收入(元)")'

    result = validate_generated_code(code, _schema("客户 类型", "收入(元)"))

    assert result["success"] is True


def test_missing_columns_are_reported() -> None:
    result = validate_generated_code(
        'result = df["category_does_not_exist"].sum()',
        _schema("category", "revenue"),
    )

    assert result["success"] is False
    assert result["missing_columns"] == ["category_does_not_exist"]
    assert "category_does_not_exist" in result["error"]


def test_dynamic_column_reference_is_unverifiable() -> None:
    code = "column_name = user_value\nresult = df[column_name]"

    result = validate_generated_code(code, _schema("category", "revenue"))

    assert result["success"] is False
    assert result["unverifiable"] is True
    assert "cannot be statically verified" in result["error"]


def test_derived_column_is_allowed_after_verified_assignment() -> None:
    code = '''df["profit"] = df["revenue"] - df["cost"]
result = df.sort_values("profit")
fig = px.bar(df, x="category", y="profit")
'''

    result = validate_generated_code(
        code,
        _schema("category", "revenue", "cost"),
    )

    assert result["success"] is True


COMPATIBILITY_SCHEMA = _schema(
    "category",
    "region",
    "revenue",
    "profit",
    "订单 金额",
    "产品-类型",
)


@pytest.mark.parametrize(
    "code",
    [
        'filtered = df[df["revenue"] > 100]',
        'filtered = df.loc[df["revenue"] > 100, ["category", "revenue"]]',
        'filtered = df.loc[:, "category"]',
        '''fig = px.scatter(
    df,
    x="revenue",
    y="profit",
    hover_data=["category", "region"],
)''',
        '''result = df.pivot_table(
    index="category",
    columns="region",
    values="revenue",
    aggfunc="sum",
)''',
        'result = df.rename(columns={"revenue": "total_revenue"})',
        'result = df.groupby("产品-类型")["订单 金额"].sum()',
    ],
)
def test_extended_column_contexts_pass(code) -> None:
    result = validate_generated_code(code, COMPATIBILITY_SCHEMA)

    assert result["success"] is True


@pytest.mark.parametrize(
    ("code", "missing_column"),
    [
        ('filtered = df.loc[:, ["category", "missing_column"]]', "missing_column"),
        (
            'fig = px.scatter(df, x="revenue", y="profit", '
            'hover_data=["missing_column"])',
            "missing_column",
        ),
        (
            'result = df.pivot_table(index="category", values="missing_column")',
            "missing_column",
        ),
        (
            'result = df.rename(columns={"missing_column": "new_name"})',
            "missing_column",
        ),
        ('result = df.groupby("missing_column")["revenue"].sum()', "missing_column"),
        ('fig = px.bar(df, x="missing_column", y="revenue")', "missing_column"),
        ('fig = px.bar(df, x="category", y="missing_column")', "missing_column"),
        (
            'fig = px.bar(df, x="category", y="revenue", color="missing_column")',
            "missing_column",
        ),
    ],
)
def test_extended_column_contexts_report_missing_columns(code, missing_column) -> None:
    result = validate_generated_code(code, COMPATIBILITY_SCHEMA)

    assert result["success"] is False
    assert result["missing_columns"] == [missing_column]
    assert missing_column in result["error"]


@pytest.mark.parametrize(
    "code",
    [
        'filtered = df.loc[:, dynamic_column]',
        'fig = px.scatter(df, x="revenue", y="profit", '
        'hover_data=[user_selected_column])',
        'result = df.pivot_table(index=dynamic_column, values="revenue")',
        'result = df.rename(columns=rename_mapping)',
    ],
)
def test_extended_dynamic_column_contexts_are_unverifiable(code) -> None:
    result = validate_generated_code(code, COMPATIBILITY_SCHEMA)

    assert result["success"] is False
    assert result["unverifiable"] is True
    assert "cannot be statically verified" in result["error"]


def test_static_review_ui_avoids_absolute_safety_claim(monkeypatch) -> None:
    from app import streamlit_app

    class FakeStreamlit:
        def __init__(self):
            self.messages = []

        def container(self, **kwargs):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def subheader(self, message):
            self.messages.append(message)

        def write(self, message):
            self.messages.append(message)

        def caption(self, message):
            self.messages.append(message)

        def warning(self, message):
            self.messages.append(message)

        def success(self, message):
            self.messages.append(message)

    fake_streamlit = FakeStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake_streamlit)

    streamlit_app.render_code_safety_review('result = df["revenue"].sum()')

    rendered_text = "\n".join(fake_streamlit.messages)
    assert "Safe:" not in rendered_text
    assert "Static check: passed" in rendered_text
    assert "has not been executed" in rendered_text
    assert "complete security guarantee" in rendered_text

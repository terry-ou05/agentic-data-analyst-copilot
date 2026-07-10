from dataclasses import replace

import pytest

from src.evaluation.cases import (
    MetricRequirement,
    SemanticEvaluationCase,
    default_semantic_evaluation_cases,
)
from src.evaluation.runner import (
    EvaluationRunner,
    run_semantic_evaluation,
    run_synthetic_semantic_evaluation,
)
from src.evaluation.synthetic_data import create_synthetic_sqlite_database


def test_default_semantic_cases_cover_multiple_intents_and_variations():
    cases = default_semantic_evaluation_cases()

    assert [case.intent_name for case in cases] == [
        "highest_revenue_category",
        "highest_online_revenue_region",
        "highest_average_quantity_category",
    ]
    assert all(len(case.question_variations) == 3 for case in cases)
    assert all(
        len(case.question_variations) == len(case.plan_payloads) for case in cases
    )
    assert all(case.expected_operations for case in cases)
    assert all(case.expected_columns for case in cases)
    assert all(case.expected_metrics for case in cases)


def test_semantic_evaluation_runs_all_variations_through_the_offline_workflow(
    tmp_path,
    monkeypatch,
):
    def unexpected_llm_call(*_args, **_kwargs):
        raise AssertionError("Semantic evaluation must not call the real LLM client.")

    monkeypatch.setattr("src.llm.client.generate_chat_completion", unexpected_llm_call)
    report = run_synthetic_semantic_evaluation(tmp_path / "semantic.db")

    assert report.setup_error == ""
    assert report.total_intents == 3
    assert report.robust_intents == 3
    assert report.intent_robustness_rate == 1.0
    assert report.total_variations == 9
    assert report.successful_variations == 9
    assert report.variation_success_rate == 1.0
    for intent in report.intents:
        assert intent.robust is True
        assert intent.success_rate == 1.0
        for variation in intent.variations:
            assert variation.success is True
            assert variation.capability_success is True
            assert variation.missing_operations == ()
            assert variation.missing_columns == ()
            assert variation.missing_metrics == ()
            assert variation.workflow_result.validation_success is True
            assert variation.workflow_result.execution_success is True
            assert variation.workflow_result.generated_plan is not None
            assert variation.workflow_result.to_dict()["generated_operations"]


def test_semantic_evaluation_compares_capability_not_exact_json(tmp_path):
    database_path = create_synthetic_sqlite_database(tmp_path / "semantic.db")
    base_case = default_semantic_evaluation_cases()[0]
    equivalent_case = replace(
        base_case,
        question_variations=("Which category wins?", "Where is the most money?"),
        plan_payloads=base_case.plan_payloads[:2],
    )

    report = EvaluationRunner(database_path).run_semantic_evaluation([equivalent_case])

    assert report.total_variations == 2
    assert report.successful_variations == 2
    aliases = [
        variation.workflow_result.generated_plan.operations[1].parameters["metrics"][0][
            "alias"
        ]
        for variation in report.intents[0].variations
    ]
    assert aliases == ["total_revenue", "category_sales"]
    assert report.intents[0].robust is True


def test_semantic_evaluation_marks_workflow_success_as_non_robust_when_capability_is_missing(
    tmp_path,
):
    database_path = create_synthetic_sqlite_database(tmp_path / "semantic.db")
    base_case = default_semantic_evaluation_cases()[0]
    incomplete_capability_case = replace(
        base_case,
        expected_operations=("filter", "groupby", "aggregate", "top_n"),
        expected_columns=("channel", "category", "revenue"),
        expected_metrics=(MetricRequirement("revenue", "mean"),),
    )

    report = EvaluationRunner(database_path).run_semantic_evaluation(
        [incomplete_capability_case]
    )
    variation = report.intents[0].variations[0]

    assert variation.workflow_result.success is True
    assert variation.capability_success is False
    assert variation.success is False
    assert variation.missing_operations == ("filter",)
    assert variation.missing_columns == ("channel",)
    assert variation.missing_metrics == ("revenue:mean",)
    assert report.intent_robustness_rate == 0.0


def test_semantic_evaluation_reports_missing_database_without_executing_variations(tmp_path):
    report = run_semantic_evaluation(tmp_path / "missing.db")

    assert report.setup_error == "Evaluation database could not be loaded."
    assert report.total_intents == 0
    assert report.total_variations == 0
    assert report.variation_success_rate == 0.0


def test_semantic_case_rejects_mismatched_variations_and_payloads():
    base_case = default_semantic_evaluation_cases()[0]

    with pytest.raises(ValueError, match="same length"):
        SemanticEvaluationCase(
            intent_name="invalid_case",
            description="Invalid fixture.",
            question_variations=("Only one question",),
            plan_payloads=base_case.plan_payloads[:2],
            expected_operations=base_case.expected_operations,
            expected_columns=base_case.expected_columns,
            expected_metrics=base_case.expected_metrics,
        )

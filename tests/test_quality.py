import pandas as pd

from ai_agent.models import (
    CleanupPlan,
    MissingValueSuggestion,
    TransformationSuggestion,
)
from ai_agent.services.quality import apply_cleanup_plan, profile, write_dataset
from ai_agent.services.recommender import (
    add_market_missingness_candidates,
    add_zero_placeholder_candidates,
)


def test_profile_counts_duplicates_missing_values_and_numeric_summary() -> None:
    frame = pd.DataFrame({"region": [" West ", " West ", None], "revenue": [10, 10, 20]})
    report = profile(frame)
    assert report.duplicate_rows == 1
    assert report.column_profiles[0].missing_count == 1
    assert report.column_profiles[1].numeric_summary == {
        "min": 10.0,
        "max": 20.0,
        "mean": 13.333333333333334,
        "median": 10.0,
    }


def test_profile_flags_negative_business_metrics_and_zero_quantity() -> None:
    report = profile(pd.DataFrame({"quantity": [1, 0, -2], "revenue": [10, 0, -20]}))

    assert "quantity: 1 negative values; verify whether they represent refunds/returns or invalid records" in report.quality_signals
    assert "quantity: 1 zero values; verify whether zero is a valid quantity" in report.quality_signals
    assert "revenue: 1 negative values; verify whether they represent refunds/returns or invalid records" in report.quality_signals
    assert report.column_profiles[0].zero_percentage == 33.33


def test_cleanup_replaces_zero_only_after_explicit_approval() -> None:
    plan = CleanupPlan(
        dataset_summary="test",
        suggestions=[
            TransformationSuggestion(
                id="zero-to-missing", operation="replace_zero_with_missing", column="bid_price",
                reason="test", impact_estimate="test", confidence="low",
            )
        ],
    )

    cleaned, validation = apply_cleanup_plan(pd.DataFrame({"bid_price": [0.0, 10.0]}), plan, {"zero-to-missing"})

    assert pd.isna(cleaned.iloc[0, 0])
    assert validation.applied_changes[0].affected_rows == 1


def test_high_zero_market_quote_fields_get_an_opt_in_review_candidate() -> None:
    dataset_profile = profile(pd.DataFrame({"bidPrice": [0.0, 0.0, 10.0], "lastPrice": [10.0, 11.0, 12.0]}))
    plan = add_zero_placeholder_candidates(CleanupPlan(dataset_summary="test"), dataset_profile)

    assert plan.suggestions[0].operation == "replace_zero_with_missing"
    assert plan.suggestions[0].column == "bidPrice"
    assert plan.suggestions[0].confidence == "low"


def test_substantial_missing_market_values_are_explicitly_left_missing() -> None:
    dataset_profile = profile(pd.DataFrame({"bidPrice": [None, None, 100.0], "lastPrice": [100.0, 101.0, 102.0]}))
    plan = add_market_missingness_candidates(CleanupPlan(dataset_summary="test"), dataset_profile)

    assert plan.missing_value_suggestions[0].strategy == "leave_missing"
    assert plan.missing_value_suggestions[0].column == "bidPrice"


def test_cleanup_applies_only_approved_actions_and_reports_results() -> None:
    frame = pd.DataFrame({"region": [" West ", " West ", None], "revenue": [10, 10, 20]})
    plan = CleanupPlan(
        dataset_summary="test",
        suggestions=[
            TransformationSuggestion(id="trim", operation="trim_text", reason="test", impact_estimate="test", confidence="high"),
            TransformationSuggestion(id="dedupe", operation="remove_duplicates", reason="test", impact_estimate="test", confidence="high"),
        ],
        missing_value_suggestions=[
            MissingValueSuggestion(
                id="fill", column="region", missing_count=1, missing_percentage=33.3,
                strategy="unknown_category", reason="test", confidence="medium",
            )
        ],
    )
    cleaned, validation = apply_cleanup_plan(frame, plan, {"trim", "dedupe"})
    assert cleaned.iloc[0].to_dict() == {"region": "West", "revenue": 10}
    assert pd.isna(cleaned.iloc[1]["region"])
    assert validation.rows_before == 3
    assert validation.rows_after == 2
    assert validation.missing_values_after["region"] == 1


def test_csv_export_neutralizes_spreadsheet_formulas(tmp_path) -> None:
    output = tmp_path / "safe.csv"
    write_dataset(pd.DataFrame({"value": ["=SUM(1,1)", "normal"]}), output)

    assert pd.read_csv(output).iloc[0, 0] == "'=SUM(1,1)"

"""Gemini-powered, structured cleanup-plan recommendations."""

from __future__ import annotations

from ai_agent.config import get_settings
from ai_agent.models import (
    CleanupPlan,
    DatasetContext,
    DatasetProfile,
    MissingValueSuggestion,
    TransformationSuggestion,
)
from utils.llm_pick import pick_llm


class CleanupRecommender:
    """Turn a privacy-aware dataset profile into an approval-ready cleanup plan."""

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required to generate cleanup suggestions.")
        self._model = pick_llm(get_settings().llm_thinking_level).with_structured_output(CleanupPlan)

    def recommend(self, profile: DatasetProfile) -> CleanupPlan:
        """Recommend strategies without receiving raw records or PII values."""

        prompt = f"""You are a careful junior data analyst.
Create an approval-ready data-cleaning plan from this dataset profile.

Rules:
- The profile covers the entire dataset. Do not request raw rows.
- Never suggest fabricated values or category mappings.
- Every missing-value action must require approval.
- Preserve identifiers and likely PII; never propose filling identifier fields.
- Suggest remove_duplicates only when duplicate_rows is greater than zero.
- Treat zero as ambiguous: it can be a valid value, a placeholder, or a business event. Only suggest
  replace_zero_with_missing when a numeric field has a substantial zero count and its field name or
  distribution makes a missing-value placeholder plausible. Never suggest it for identifiers, timestamps,
  counts, quantities, volume, prices, or price-change fields without explicitly listing that uncertainty in risks.
- Every replace_zero_with_missing action must require approval and explain why zero may be a placeholder.
- For market price, bid, ask, and price-change fields, do not invent missing values. When those fields have
  substantial missingness, recommend leave_missing and source investigation rather than statistical imputation.
- Use only the supported operation and strategy values in the output schema.
- Explain uncertainty in risks.
- Give compact, conservative recommendations.

Dataset profile:
{profile.model_dump_json(indent=2)}
"""
        plan = self._model.invoke(prompt)
        plan = add_zero_placeholder_candidates(plan, profile)
        return add_market_missingness_candidates(plan, profile)


def add_zero_placeholder_candidates(plan: CleanupPlan, profile: DatasetProfile) -> CleanupPlan:
    """Surface high-zero market quote fields for human review.

    A zero may be a real value, so this creates only a low-confidence, opt-in
    proposal. It intentionally targets quote fields commonly populated with an
    API placeholder and never changes the data itself.
    """

    quote_field = ("bid", "ask", "prevclose", "previousclose")
    existing_columns = {suggestion.column for suggestion in plan.suggestions}
    for column in profile.column_profiles:
        normalized_name = column.name.lower().replace("_", "")
        if (
            column.name in existing_columns
            or column.zero_percentage < 50
            or not any(term in normalized_name for term in quote_field)
        ):
            continue
        plan.suggestions.append(
            TransformationSuggestion(
                id=f"review-zero-placeholder-{column.name.lower()}",
                operation="replace_zero_with_missing",
                column=column.name,
                reason=(
                    f"{column.zero_count} of {profile.rows} values ({column.zero_percentage}%) are zero. In a market quote field, "
                    "this can be an API placeholder rather than an observed quote; confirm with the source."
                ),
                impact_estimate=f"Would replace {column.zero_count} zero values with missing values.",
                confidence="low",
            )
        )
    return plan


def add_market_missingness_candidates(plan: CleanupPlan, profile: DatasetProfile) -> CleanupPlan:
    """Make substantial missingness in market fields visible without fabricating values."""

    market_field = ("bid", "ask", "prevclose", "previousclose", "pricechange")
    existing_columns = {suggestion.column for suggestion in plan.missing_value_suggestions}
    for column in profile.column_profiles:
        normalized_name = column.name.lower().replace("_", "")
        if (
            column.name in existing_columns
            or column.missing_percentage < 20
            or not any(term in normalized_name for term in market_field)
        ):
            continue
        plan.missing_value_suggestions.append(
            MissingValueSuggestion(
                id=f"review-market-missingness-{column.name.lower()}",
                column=column.name,
                missing_count=column.missing_count,
                missing_percentage=column.missing_percentage,
                strategy="leave_missing",
                reason=(
                    f"{column.missing_count} of {profile.rows} values ({column.missing_percentage}%) are missing. "
                    "A bid, ask, previous-close, or price-change value cannot be safely imputed from other assets; investigate the source or market status."
                ),
                confidence="high",
            )
        )
    return plan


class DatasetGuide:
    """Explain a dataset and suggest grounded starting questions from its full profile."""

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required to generate dataset context.")
        self._model = pick_llm(get_settings().llm_thinking_level).with_structured_output(DatasetContext)

    def explain(self, profile: DatasetProfile) -> DatasetContext:
        """Create an honest orientation brief without transmitting raw records."""

        prompt = f"""You are a careful junior data analyst helping a non-technical person orient themselves to an unfamiliar dataset.
Use only this full-dataset, privacy-aware profile. You do not have raw records or business documentation.

Rules:
- State likely interpretations as likely, not certain. Do not invent business facts.
- Explain every column in `column_explanations` using only its name, type, summary, and non-sensitive distribution.
- Describe the likely row grain, such as one row per order, event, person, or record. State uncertainty when needed.
- Include caveats that make conclusions unreliable, especially quality signals and likely PII.
- Suggest 4 to 8 concrete, answerable starter questions using only columns that exist. Include at least one quality question when quality signals exist.
- Do not use PII columns in suggested questions. Do not request raw records.
- Use concise, plain language.

Dataset profile:
{profile.model_dump_json(indent=2)}
"""
        return self._model.invoke(prompt)

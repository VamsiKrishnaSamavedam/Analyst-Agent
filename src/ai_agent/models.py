"""Typed contracts for the dataset-quality and cleanup workflow."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    data_type: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    zero_count: int = 0
    zero_percentage: float = 0.0
    numeric_summary: dict[str, float] | None = None
    top_values: list[tuple[str, int]] = Field(default_factory=list)
    potential_pii: bool = False


class DatasetProfile(BaseModel):
    rows: int
    columns: int
    duplicate_rows: int
    column_profiles: list[ColumnProfile]
    quality_signals: list[str]


class DatasetContext(BaseModel):
    """A profile-grounded orientation brief for a person new to a dataset."""

    dataset_description: str
    likely_grain: str
    key_entities: list[str] = Field(default_factory=list)
    column_explanations: dict[str, str] = Field(default_factory=dict)
    caveats: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list, min_length=4, max_length=8)


class MissingValueSuggestion(BaseModel):
    id: str
    column: str
    missing_count: int
    missing_percentage: float
    strategy: Literal[
        "leave_missing",
        "drop_rows",
        "fill_constant",
        "median",
        "mode",
        "group_median",
        "unknown_category",
    ]
    proposed_value: str | float | None = None
    group_by_column: str | None = None
    reason: str
    confidence: Literal["low", "medium", "high"]
    approval_required: bool = True


class TransformationSuggestion(BaseModel):
    id: str
    operation: Literal[
        "remove_duplicates",
        "trim_text",
        "normalize_category",
        "convert_type",
        "replace_zero_with_missing",
    ]
    column: str | None = None
    category_mapping: dict[str, str] = Field(default_factory=dict)
    target_data_type: Literal["numeric", "datetime", "string"] | None = None
    reason: str
    impact_estimate: str
    confidence: Literal["low", "medium", "high"]
    approval_required: bool = True


class CleanupPlan(BaseModel):
    dataset_summary: str
    risks: list[str] = Field(default_factory=list)
    suggestions: list[TransformationSuggestion] = Field(default_factory=list)
    missing_value_suggestions: list[MissingValueSuggestion] = Field(default_factory=list)


class AppliedChange(BaseModel):
    id: str
    description: str
    affected_rows: int


class ValidationReport(BaseModel):
    rows_before: int
    rows_after: int
    duplicate_rows_before: int
    duplicate_rows_after: int
    missing_values_before: dict[str, int]
    missing_values_after: dict[str, int]
    applied_changes: list[AppliedChange] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TargetColumn(BaseModel):
    """A PostgreSQL target-table field available to an ETL mapping."""

    name: str
    data_type: str
    nullable: bool
    is_primary_key: bool = False


class ColumnMapping(BaseModel):
    """A deterministic source-to-target mapping proposal for review."""

    source_column: str | None = None
    target_column: str
    status: Literal["mapped", "review_required", "unmapped"]
    rationale: str


class ETLReadinessReport(BaseModel):
    """Pre-load mapping and quality checks; no target data is changed."""

    source_rows: int
    target_table: str
    mappings: list[ColumnMapping] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ReconciliationReport(BaseModel):
    """A source-versus-target row-count comparison for an ETL load."""

    source_rows: int
    target_rows: int
    difference: int
    status: Literal["matched", "mismatch"]
    note: str


class StagingResult(BaseModel):
    """Receipt for an immutable raw-dataset copy stored in PostgreSQL staging."""

    staging_dataset_id: str
    source_name: str
    source_format: str
    rows_staged: int
    staging_table: str = "etl_staging.dataset_rows"

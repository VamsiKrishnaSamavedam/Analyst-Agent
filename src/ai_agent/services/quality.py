"""Full-dataset profiling, approved cleanup, and validation reporting."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ai_agent.models import (
    AppliedChange,
    CleanupPlan,
    ColumnProfile,
    DatasetProfile,
    ValidationReport,
)

_PII_NAME = re.compile(r"(email|e-mail|phone|mobile|address|ssn|social|password|credit|card|note|comment|message|description)", re.IGNORECASE)
_FORMULA_PREFIX = re.compile(r"^[=+\-@]")
_NON_NEGATIVE_METRIC = re.compile(r"(quantity|qty|unit|price|revenue|sales|amount|total|cost|balance)", re.IGNORECASE)
_QUANTITY_METRIC = re.compile(r"(quantity|qty|units?)", re.IGNORECASE)


def read_dataset(path: Path) -> pd.DataFrame:
    """Read one supported tabular dataset into a dataframe."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError("Supported input formats are CSV, JSON, and Parquet.")


def profile(frame: pd.DataFrame) -> DatasetProfile:
    """Inspect every row and return a compact, privacy-aware quality profile."""

    columns: list[ColumnProfile] = []
    signals: list[str] = []
    for name in frame.columns:
        series = frame[name]
        missing_count = int(series.isna().sum())
        pii = bool(_PII_NAME.search(str(name)))
        zero_count = 0
        numeric_summary = None
        if pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            if not valid.empty:
                numeric_summary = {
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "mean": float(valid.mean()),
                    "median": float(valid.median()),
                }
            zero_count = int((series == 0).sum())
        # The LLM receives this profile. Do not include source values, because a
        # non-obvious text field can contain personal or confidential data.
        top_values: list[tuple[str, int]] = []
        columns.append(
            ColumnProfile(
                name=str(name),
                data_type=str(series.dtype),
                missing_count=missing_count,
                missing_percentage=round((missing_count / len(frame) * 100) if len(frame) else 0, 2),
                unique_count=int(series.nunique(dropna=True)),
                zero_count=zero_count,
                zero_percentage=round((zero_count / len(frame) * 100) if len(frame) else 0, 2),
                numeric_summary=numeric_summary,
                top_values=top_values,
                potential_pii=pii,
            )
        )
        if missing_count:
            signals.append(f"{name}: {missing_count} missing values")
        if pd.api.types.is_numeric_dtype(series) and _NON_NEGATIVE_METRIC.search(str(name)):
            negative_count = int((series < 0).sum())
            if negative_count:
                signals.append(f"{name}: {negative_count} negative values; verify whether they represent refunds/returns or invalid records")
            if zero_count:
                meaning = "verify whether zero is a valid quantity" if _QUANTITY_METRIC.search(str(name)) else "verify whether zero represents a real value or a missing-value placeholder"
                signals.append(f"{name}: {zero_count} zero values; {meaning}")
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        signals.append(f"{duplicates} exact duplicate rows")
    return DatasetProfile(
        rows=len(frame),
        columns=len(frame.columns),
        duplicate_rows=duplicates,
        column_profiles=columns,
        quality_signals=signals,
    )


def apply_cleanup_plan(frame: pd.DataFrame, plan: CleanupPlan, approved_ids: set[str]) -> tuple[pd.DataFrame, ValidationReport]:
    """Apply only user-approved, deterministic plan items and validate the result."""

    before = profile(frame)
    cleaned = frame.copy()
    changes: list[AppliedChange] = []
    warnings: list[str] = []

    for suggestion in plan.suggestions:
        if suggestion.id not in approved_ids:
            continue
        if suggestion.operation == "remove_duplicates":
            old_count = len(cleaned)
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)
            changes.append(AppliedChange(id=suggestion.id, description="Removed exact duplicate rows", affected_rows=old_count - len(cleaned)))
        elif suggestion.operation == "trim_text":
            target = [suggestion.column] if suggestion.column else list(cleaned.select_dtypes(include=["object", "string"]).columns)
            for column in target:
                if column in cleaned:
                    original = cleaned[column]
                    cleaned[column] = original.where(original.isna(), original.astype(str).str.strip())
                    changes.append(AppliedChange(id=suggestion.id, description=f"Trimmed whitespace in {column}", affected_rows=int((original != cleaned[column]).fillna(False).sum())))
        elif suggestion.operation == "normalize_category" and suggestion.column in cleaned:
            original = cleaned[suggestion.column]
            cleaned[suggestion.column] = cleaned[suggestion.column].replace(suggestion.category_mapping)
            changes.append(AppliedChange(id=suggestion.id, description=f"Normalized category values in {suggestion.column}", affected_rows=int((original != cleaned[suggestion.column]).fillna(False).sum())))
        elif suggestion.operation == "convert_type" and suggestion.column in cleaned:
            original = cleaned[suggestion.column]
            if suggestion.target_data_type == "numeric":
                cleaned[suggestion.column] = pd.to_numeric(cleaned[suggestion.column], errors="coerce")
            elif suggestion.target_data_type == "datetime":
                cleaned[suggestion.column] = pd.to_datetime(cleaned[suggestion.column], errors="coerce")
            elif suggestion.target_data_type == "string":
                cleaned[suggestion.column] = cleaned[suggestion.column].astype("string")
            newly_missing = int((original.notna() & cleaned[suggestion.column].isna()).sum())
            if newly_missing:
                warnings.append(f"{suggestion.column}: {newly_missing} values could not be converted and are now missing.")
            changes.append(AppliedChange(id=suggestion.id, description=f"Converted {suggestion.column} to {suggestion.target_data_type}", affected_rows=len(cleaned)))
        elif suggestion.operation == "replace_zero_with_missing" and suggestion.column in cleaned:
            if not pd.api.types.is_numeric_dtype(cleaned[suggestion.column]):
                warnings.append(f"Skipped zero replacement for non-numeric column: {suggestion.column}")
                continue
            affected = int((cleaned[suggestion.column] == 0).sum())
            cleaned[suggestion.column] = cleaned[suggestion.column].mask(cleaned[suggestion.column] == 0)
            changes.append(AppliedChange(id=suggestion.id, description=f"Replaced zero placeholders with missing values in {suggestion.column}", affected_rows=affected))
        else:
            warnings.append(f"Skipped unsupported or invalid transformation: {suggestion.id}")

    for suggestion in plan.missing_value_suggestions:
        if suggestion.id not in approved_ids or suggestion.column not in cleaned:
            continue
        column = cleaned[suggestion.column]
        affected = int(column.isna().sum())
        if suggestion.strategy == "leave_missing":
            continue
        if suggestion.strategy == "drop_rows":
            cleaned = cleaned.dropna(subset=[suggestion.column]).reset_index(drop=True)
        elif suggestion.strategy == "median":
            cleaned[suggestion.column] = column.fillna(column.median())
        elif suggestion.strategy == "mode" and not column.mode().empty:
            cleaned[suggestion.column] = column.fillna(column.mode().iloc[0])
        elif suggestion.strategy == "fill_constant":
            cleaned[suggestion.column] = column.fillna(suggestion.proposed_value)
        elif suggestion.strategy == "unknown_category":
            cleaned[suggestion.column] = column.fillna(suggestion.proposed_value or "Unknown")
        elif suggestion.strategy == "group_median" and suggestion.group_by_column in cleaned:
            cleaned[suggestion.column] = column.fillna(cleaned.groupby(suggestion.group_by_column)[suggestion.column].transform("median"))
        else:
            warnings.append(f"Skipped invalid missing-value strategy: {suggestion.id}")
            continue
        changes.append(AppliedChange(id=suggestion.id, description=f"Applied {suggestion.strategy} to {suggestion.column}", affected_rows=affected))

    after = profile(cleaned)
    return cleaned, ValidationReport(
        rows_before=before.rows,
        rows_after=after.rows,
        duplicate_rows_before=before.duplicate_rows,
        duplicate_rows_after=after.duplicate_rows,
        missing_values_before={column.name: column.missing_count for column in before.column_profiles},
        missing_values_after={column.name: column.missing_count for column in after.column_profiles},
        applied_changes=changes,
        warnings=warnings,
    )


def write_dataset(frame: pd.DataFrame, output: Path) -> None:
    """Write cleaned data in the output file's declared format."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        safe_frame = frame.copy()
        for column in safe_frame.select_dtypes(include=["object", "string"]).columns:
            safe_frame[column] = safe_frame[column].map(
                lambda value: f"'{value}" if isinstance(value, str) and _FORMULA_PREFIX.match(value) else value
            )
        safe_frame.to_csv(output, index=False)
    elif output.suffix.lower() == ".json":
        frame.to_json(output, orient="records", lines=True)
    elif output.suffix.lower() == ".parquet":
        frame.to_parquet(output, index=False)
    else:
        raise ValueError("Output must end in .csv, .json, or .parquet.")

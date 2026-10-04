"""Deterministic ETL mapping and reconciliation for local PostgreSQL targets."""

from __future__ import annotations

import re

import pandas as pd

from ai_agent.models import (
    ColumnMapping,
    ETLReadinessReport,
    ReconciliationReport,
    TargetColumn,
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def propose_mapping(frame: pd.DataFrame, target_table: str, target_columns: list[TargetColumn]) -> ETLReadinessReport:
    """Create a conservative, name-based mapping report without changing data."""

    source_by_normalized_name = {_normalize(name): str(name) for name in frame.columns}
    mappings: list[ColumnMapping] = []
    blocking_issues: list[str] = []
    warnings: list[str] = []
    mapped_sources: set[str] = set()

    for target in target_columns:
        source = source_by_normalized_name.get(_normalize(target.name))
        if source:
            mapped_sources.add(source)
            mappings.append(
                ColumnMapping(
                    source_column=source,
                    target_column=target.name,
                    status="mapped",
                    rationale="Exact normalized column-name match.",
                )
            )
        elif not target.nullable:
            blocking_issues.append(f"Required target column {target.name} has no source mapping.")
            mappings.append(
                ColumnMapping(
                    target_column=target.name,
                    status="unmapped",
                    rationale="No exact source-column match; a human mapping decision is required.",
                )
            )
        else:
            warnings.append(f"Optional target column {target.name} has no source mapping.")
            mappings.append(
                ColumnMapping(
                    target_column=target.name,
                    status="unmapped",
                    rationale="No exact source-column match; it may remain NULL after approval.",
                )
            )

    for source in frame.columns:
        if str(source) not in mapped_sources:
            warnings.append(f"Source column {source} is not mapped to the target table.")

    return ETLReadinessReport(
        source_rows=len(frame),
        target_table=target_table,
        mappings=mappings,
        blocking_issues=blocking_issues,
        warnings=warnings,
    )


def reconcile_row_counts(source_rows: int, target_rows: int, target_table: str) -> ReconciliationReport:
    """Compare source and target row counts without inferring business rules."""

    difference = target_rows - source_rows
    matched = difference == 0
    return ReconciliationReport(
        source_rows=source_rows,
        target_rows=target_rows,
        difference=difference,
        status="matched" if matched else "mismatch",
        note=(
            "Source and target row counts match."
            if matched
            else f"{target_table} differs from the source by {abs(difference)} row(s); investigate approved filters, duplicates, and load errors."
        ),
    )


def validate_target_table_name(table_name: str) -> tuple[str, str]:
    """Allow only a simple PostgreSQL schema.table identifier."""

    parts = table_name.split(".")
    if len(parts) == 1:
        parts.insert(0, "public")
    if len(parts) != 2 or not all(_IDENTIFIER.fullmatch(part) for part in parts):
        raise ValueError("Target table must be a simple name such as public.rides.")
    return parts[0], parts[1]


def _normalize(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

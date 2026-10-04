import pandas as pd
import pytest

from ai_agent.models import TargetColumn
from ai_agent.services.postgres_etl import (
    propose_mapping,
    reconcile_row_counts,
    validate_target_table_name,
)


def test_mapping_requires_human_decision_for_missing_required_target_column() -> None:
    report = propose_mapping(
        pd.DataFrame({"ride_id": [1], "fare": [12.5], "unexpected": ["x"]}),
        "public.rides",
        [
            TargetColumn(name="ride_id", data_type="integer", nullable=False, is_primary_key=True),
            TargetColumn(name="fare", data_type="numeric", nullable=True),
            TargetColumn(name="rider_id", data_type="integer", nullable=False),
        ],
    )

    assert [item.status for item in report.mappings] == ["mapped", "mapped", "unmapped"]
    assert report.blocking_issues == ["Required target column rider_id has no source mapping."]
    assert "Source column unexpected is not mapped to the target table." in report.warnings


def test_reconciliation_reports_count_mismatches_without_guessing_cause() -> None:
    report = reconcile_row_counts(10, 8, "public.rides")

    assert report.status == "mismatch"
    assert report.difference == -2


@pytest.mark.parametrize("table", ["public.rides;DROP", "public.rides.more", "public.123rides"])
def test_target_table_validation_rejects_unsafe_names(table: str) -> None:
    with pytest.raises(ValueError):
        validate_target_table_name(table)

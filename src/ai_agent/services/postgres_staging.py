"""Generic, append-only raw-data staging for local PostgreSQL ETL work."""

from __future__ import annotations

import json
import uuid

import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values

from ai_agent.models import StagingResult


def stage_dataframe(frame: pd.DataFrame, source_name: str, source_format: str, database_config: dict[str, object]) -> StagingResult:
    """Store heterogeneous raw records as JSONB in a fixed staging schema."""

    dataset_id = str(uuid.uuid4())
    records = dataframe_records(frame)
    columns = [{"name": str(name), "data_type": str(data_type)} for name, data_type in frame.dtypes.items()]
    with psycopg2.connect(**database_config) as connection, connection.cursor() as cursor:
        _ensure_staging_schema(cursor)
        cursor.execute(
            "INSERT INTO etl_staging.datasets (dataset_id, source_name, source_format, row_count, columns) VALUES (%s, %s, %s, %s, %s)",
            (dataset_id, source_name, source_format, len(records), Json(columns)),
        )
        if records:
            execute_values(
                cursor,
                "INSERT INTO etl_staging.dataset_rows (dataset_id, row_number, payload) VALUES %s",
                [(dataset_id, row_number, Json(record)) for row_number, record in enumerate(records, start=1)],
                page_size=1_000,
            )
    return StagingResult(staging_dataset_id=dataset_id, source_name=source_name, source_format=source_format, rows_staged=len(records))


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert pandas values to JSON-safe records while preserving missing values."""

    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _ensure_staging_schema(cursor) -> None:
    cursor.execute("CREATE SCHEMA IF NOT EXISTS etl_staging")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS etl_staging.datasets (
            dataset_id UUID PRIMARY KEY, source_name TEXT NOT NULL, source_format TEXT NOT NULL,
            staged_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), row_count INTEGER NOT NULL CHECK (row_count >= 0), columns JSONB NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS etl_staging.dataset_rows (
            dataset_id UUID NOT NULL REFERENCES etl_staging.datasets(dataset_id), row_number INTEGER NOT NULL CHECK (row_number > 0),
            payload JSONB NOT NULL, PRIMARY KEY (dataset_id, row_number)
        )
    """)

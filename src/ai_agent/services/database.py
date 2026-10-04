"""Read-only PostgreSQL access and deterministic SQL validation."""

from __future__ import annotations

import re
from collections.abc import Sequence

import psycopg2

from ai_agent.models import TargetColumn

_FORBIDDEN = re.compile(
    r"\b(ALTER|CALL|COPY|CREATE|DELETE|DROP|GRANT|INSERT|MERGE|REVOKE|TRUNCATE|UPDATE|VACUUM)\b",
    re.IGNORECASE,
)
_COMMENTS = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.DOTALL)


def validate_read_only_sql(query: str) -> str:
    """Return a normalized single SELECT/CTE query or raise a useful error."""

    normalized = _COMMENTS.sub("", query).strip()
    if not normalized:
        raise ValueError("SQL query cannot be empty.")
    if normalized.count(";") > 1 or (";" in normalized and not normalized.endswith(";")):
        raise ValueError("Only one SQL statement is allowed.")
    normalized = normalized.rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", normalized, re.IGNORECASE):
        raise ValueError("Only SELECT queries and read-only CTEs are allowed.")
    if _FORBIDDEN.search(normalized):
        raise ValueError("The query contains a database-changing or administrative command.")
    return normalized


class ReadOnlyDatabase:
    """Small repository layer that never commits application queries."""

    def __init__(self, config: dict[str, object]) -> None:
        self._config = config

    def execute(self, query: str) -> tuple[list[str], list[tuple[object, ...]]]:
        safe_query = validate_read_only_sql(query)
        with psycopg2.connect(**self._config) as connection:
            connection.set_session(readonly=True, autocommit=False)
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout = '10s'")
                cursor.execute(safe_query)
                rows = cursor.fetchmany(100)
                columns = [item.name for item in cursor.description]
        return columns, rows

    def describe_public_schema(self) -> str:
        """Return compact schema context suitable for a model prompt or CLI output."""

        statement = """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position;
        """
        _, rows = self.execute(statement)
        return "\n".join(f"public.{table}: {column} ({data_type})" for table, column, data_type in rows)

    def describe_table(self, schema_name: str, table_name: str) -> list[TargetColumn]:
        """Read target metadata for ETL mapping without exposing table data."""

        statement = """
            SELECT
                columns.column_name,
                columns.data_type,
                columns.is_nullable = 'YES' AS nullable,
                EXISTS (
                    SELECT 1
                    FROM information_schema.table_constraints constraints
                    JOIN information_schema.key_column_usage keys
                      ON constraints.constraint_name = keys.constraint_name
                     AND constraints.table_schema = keys.table_schema
                    WHERE constraints.constraint_type = 'PRIMARY KEY'
                      AND constraints.table_schema = columns.table_schema
                      AND constraints.table_name = columns.table_name
                      AND keys.column_name = columns.column_name
                ) AS is_primary_key
            FROM information_schema.columns columns
            WHERE columns.table_schema = %s AND columns.table_name = %s
            ORDER BY columns.ordinal_position
        """
        with psycopg2.connect(**self._config) as connection:
            connection.set_session(readonly=True, autocommit=False)
            with connection.cursor() as cursor:
                cursor.execute("SET LOCAL statement_timeout = '10s'")
                cursor.execute(statement, (schema_name, table_name))
                rows = cursor.fetchall()
        if not rows:
            raise ValueError(f"Target table {schema_name}.{table_name} was not found.")
        return [
            TargetColumn(name=name, data_type=data_type, nullable=nullable, is_primary_key=is_primary_key)
            for name, data_type, nullable, is_primary_key in rows
        ]

    def count_rows(self, schema_name: str, table_name: str) -> int:
        """Return a target-table row count through a readonly session."""

        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema_name) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            raise ValueError("Invalid schema or table name.")
        statement = f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"'
        _, rows = self.execute(statement)
        return int(rows[0][0])


def format_rows(columns: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render query results as a compact plain-text table without extra dependencies."""

    if not rows:
        return "No matching rows found."
    rendered = [[str(value) if value is not None else "" for value in row] for row in rows]
    widths = [max(len(column), *(len(row[index]) for row in rendered)) for index, column in enumerate(columns)]
    header = " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns))
    divider = "-+-".join("-" * width for width in widths)
    body = [" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rendered]
    return "\n".join([header, divider, *body])

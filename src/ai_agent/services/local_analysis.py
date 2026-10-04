"""Safe local SQL analysis for an uploaded dataset."""

from __future__ import annotations

import re

import duckdb
import pandas as pd
from langchain_core.output_parsers import StrOutputParser

from ai_agent.config import get_settings
from ai_agent.models import DatasetProfile
from ai_agent.services.database import validate_read_only_sql
from utils.llm_pick import pick_llm

_DISALLOWED_LOCAL_SQL = re.compile(
    r"\b(ATTACH|COPY|EXPORT|IMPORT|INSTALL|LOAD|PRAGMA|READ_(?:CSV|JSON|PARQUET|TEXT)(?:_AUTO)?|"
    r"GLOB|HTTPFS|SQLITE_SCAN|POSTGRES_SCAN|DUCKDB_TABLES|INFORMATION_SCHEMA)\b",
    re.IGNORECASE,
)
_FROM_DATASET = re.compile(r"\b(FROM|JOIN)\s+(?:main\.)?dataset\b", re.IGNORECASE)
_REQUESTED_ROW_LIMIT = re.compile(r"\b(?:first|top|last)\s+(\d+)\s+(?:records?|rows?|items?|entries?|results?)\b", re.IGNORECASE)


def execute_local_query(frame: pd.DataFrame, query: str) -> tuple[str, list[str], list[list[object]]]:
    """Run one validated analytical query against the in-memory `dataset` table."""

    safe_query = validate_read_only_sql(query)
    if _DISALLOWED_LOCAL_SQL.search(safe_query) or not _FROM_DATASET.search(safe_query):
        raise ValueError("Queries may read only the local table named dataset.")
    connection = duckdb.connect(":memory:", config={"enable_external_access": "false", "memory_limit": "512MB", "threads": "2"})
    try:
        connection.register("dataset", frame)
        result = connection.execute(safe_query).fetchdf()
    finally:
        connection.close()
    return safe_query, list(result.columns), result.where(pd.notna(result), None).values.tolist()


def apply_requested_row_limit(question: str, query: str) -> str:
    """Make a generated detail query honor an explicit user-requested row count."""

    match = _REQUESTED_ROW_LIMIT.search(question)
    if not match:
        return query
    limit = match.group(1)
    if re.search(r"\bLIMIT\s+\d+\b", query, re.IGNORECASE):
        return re.sub(r"\bLIMIT\s+\d+\b", f"LIMIT {limit}", query, count=1, flags=re.IGNORECASE)
    return f"{query.rstrip().rstrip(';')} LIMIT {limit}"


class LocalSQLAssistant:
    """Generate DuckDB SQL using only a privacy-aware profile and column names."""

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for natural-language analysis.")
        self._model = pick_llm(get_settings().llm_thinking_level) | StrOutputParser()

    def generate(self, question: str, dataset_profile: DatasetProfile) -> str:
        prompt = f'''Write exactly one DuckDB SELECT query for local analytics.
The only available table is named dataset. Use only columns shown in the profile.
Return SQL only: no prose, no markdown, no comments, no mutations, and no external files.
If the question asks for a specific number of records, use that exact LIMIT. Otherwise use LIMIT 100 for detail results.

Profile (no raw rows):
{dataset_profile.model_dump_json(indent=2)}

Question: {question}'''
        return self._model.invoke(prompt).replace("```sql", "").replace("```", "").strip()

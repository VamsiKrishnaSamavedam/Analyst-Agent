"""Gemini-backed, schema-grounded natural-language SQL assistant."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from ai_agent.config import get_settings
from ai_agent.services.database import ReadOnlyDatabase, validate_read_only_sql
from utils.llm_pick import pick_llm


class SQLAssistant:
    """Generate a single query from database metadata and execute it read-only."""

    def __init__(self, database: ReadOnlyDatabase, api_key: str | None) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required for natural-language questions.")
        self._database = database
        self._model = pick_llm(get_settings().llm_thinking_level) | StrOutputParser()

    def answer(self, question: str) -> tuple[str, list[str], list[tuple[object, ...]]]:
        schema = self._database.describe_public_schema()
        prompt = f'''You write PostgreSQL for a read-only analytics assistant.
Return only one executable SELECT query (a read-only WITH query is also allowed).
Use only the exact tables and columns listed below. Never use comments, DDL, DML,
or multiple statements. Add LIMIT 100 to non-aggregate result sets unless the user
requests another limit.

Schema:
{schema}

Question: {question}
'''
        raw_query = self._model.invoke(prompt).replace("```sql", "").replace("```", "")
        query = validate_read_only_sql(raw_query)
        columns, rows = self._database.execute(query)
        return query, columns, rows

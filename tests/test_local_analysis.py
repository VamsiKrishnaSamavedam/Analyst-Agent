import pandas as pd
import pytest

from ai_agent.api import _result_interpretation_notes
from ai_agent.services.local_analysis import (
    apply_requested_row_limit,
    execute_local_query,
)


def test_local_query_reads_uploaded_dataset() -> None:
    query, columns, rows = execute_local_query(pd.DataFrame({"region": ["east", "west"], "sales": [10, 20]}), "SELECT region, sales FROM dataset ORDER BY sales")

    assert query == "SELECT region, sales FROM dataset ORDER BY sales"
    assert columns == ["region", "sales"]
    assert rows == [["east", 10], ["west", 20]]


@pytest.mark.parametrize("query", ["DELETE FROM dataset", "SELECT * FROM other_table", "SELECT * FROM read_csv_auto('secret.csv') JOIN dataset ON true"])
def test_local_query_rejects_unsafe_or_unknown_source(query: str) -> None:
    with pytest.raises(ValueError):
        execute_local_query(pd.DataFrame({"value": [1]}), query)


def test_negative_revenue_result_is_explained_not_silently_rewritten() -> None:
    notes = _result_interpretation_notes(["product", "total_revenue"], [["Headset", -239.98]])

    assert len(notes) == 1
    assert "refunds or returns" in notes[0]


def test_generated_sql_honors_an_explicit_requested_row_count() -> None:
    query = apply_requested_row_limit("Show me the first 10 records of this dataset", "SELECT * FROM dataset LIMIT 100")

    assert query == "SELECT * FROM dataset LIMIT 10"

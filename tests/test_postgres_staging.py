import pandas as pd

from ai_agent.services.postgres_staging import dataframe_records


def test_staging_records_preserve_columns_and_convert_missing_values_to_json_null() -> None:
    records = dataframe_records(pd.DataFrame({"symbol": ["BTCUSD", "ETHUSD"], "price": [100.0, None]}))

    assert records == [{"symbol": "BTCUSD", "price": 100.0}, {"symbol": "ETHUSD", "price": None}]

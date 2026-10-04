import pandas as pd

from ai_agent.models import ValidationReport
from ai_agent.services.dataset_store import DatasetStore


def test_raw_frame_remains_available_after_cleaned_output(tmp_path) -> None:
    store = DatasetStore(tmp_path)
    dataset_id, _ = store.create_from_frame(pd.DataFrame({"value": [1, 2]}), "source.csv")
    store.save_cleaned(
        dataset_id,
        pd.DataFrame({"value": [10]}),
        ValidationReport(
            rows_before=2,
            rows_after=1,
            duplicate_rows_before=0,
            duplicate_rows_after=0,
            missing_values_before={"value": 0},
            missing_values_after={"value": 0},
        ),
    )

    assert store.raw_frame(dataset_id).to_dict(orient="list") == {"value": [1, 2]}

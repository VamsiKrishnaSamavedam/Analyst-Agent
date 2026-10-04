"""Deterministic ETL operations used by the command-line interface."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

SUPPORTED_FORMATS = {"csv", "json", "parquet"}


def extract_api_data(url: str, output_directory: Path, output_format: str, filename: str) -> Path:
    """Fetch a JSON API endpoint and persist its first-level records."""

    normalized_format = output_format.lower().lstrip(".")
    if normalized_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {output_format}. Choose from {sorted(SUPPORTED_FORMATS)}.")

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    records = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    frame = pd.json_normalize(records if isinstance(records, list) else [records])

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{Path(filename).stem}.{normalized_format}"
    if normalized_format == "csv":
        frame.to_csv(output_path, index=False)
    elif normalized_format == "json":
        frame.to_json(output_path, orient="records", lines=True)
    else:
        frame.to_parquet(output_path, index=False)
    return output_path

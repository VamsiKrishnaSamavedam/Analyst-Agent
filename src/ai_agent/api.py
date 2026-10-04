"""Browser-facing application for safe, approval-based dataset analysis."""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from secrets import compare_digest
from typing import Annotated
from urllib.parse import urlparse

import pandas as pd
import psycopg2
import requests
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.aiagent import ApplicationDataAgent
from ai_agent.config import get_settings
from ai_agent.models import CleanupPlan
from ai_agent.services.dataset_store import DatasetStore

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_API_RESPONSE_BYTES = int(os.getenv("MAX_API_RESPONSE_MB", "25")) * 1024 * 1024
DATA_RETENTION_HOURS = int(os.getenv("DATA_RETENTION_HOURS", "24"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
APP_API_KEY = os.getenv("APP_API_KEY")
SUPPORTED_SUFFIXES = {".csv", ".json", ".parquet"}
logger = logging.getLogger(__name__)
request_times: dict[str, deque[float]] = defaultdict(deque)
ROOT = Path(__file__).resolve().parents[2]
store = DatasetStore(ROOT / "data" / "runtime")
workflow_agent = ApplicationDataAgent(get_settings().gemini_api_key)
app = FastAPI(title="Analyst Desk", version="1.0.0")


class ApiImportRequest(BaseModel):
    url: str


class ApplyPlanRequest(BaseModel):
    plan: CleanupPlan
    approved_ids: list[str] = Field(default_factory=list)


class AnalysisRequest(BaseModel):
    question: str | None = None
    sql: str | None = None


def _result_interpretation_notes(columns: list[str], rows: list[list[object]]) -> list[str]:
    """Flag potentially surprising metric results without changing the answer."""

    metric_columns = ("revenue", "sales", "amount", "total", "price", "cost", "balance", "quantity", "qty")
    notes: list[str] = []
    for index, column in enumerate(columns):
        if not any(term in column.lower() for term in metric_columns):
            continue
        negative_count = sum(
            isinstance(row[index], (int, float)) and not isinstance(row[index], bool) and row[index] < 0
            for row in rows
            if len(row) > index
        )
        if negative_count:
            notes.append(
                f"{column} has {negative_count} negative result value(s). This may be valid net activity "
                "such as refunds or returns, or it may reflect source-data errors. Verify the related status and quantity fields before treating it as revenue."
            )
    return notes


def _http_error(error: Exception) -> HTTPException:
    status_code = 404 if "not found" in str(error).lower() else 400
    return HTTPException(status_code=status_code, detail="The request could not be completed. Check the dataset or request and try again.")


def _safe_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API URL must be a complete http or https address.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise ValueError("API URL must resolve to a public address.")
    except socket.gaierror as error:
        raise ValueError("API host could not be resolved.") from error
    return url


def _fetch_json_frame(url: str) -> pd.DataFrame:
    response = requests.get(_safe_public_url(url), timeout=20, allow_redirects=False)
    response.raise_for_status()
    declared_size = int(response.headers.get("content-length", "0"))
    if declared_size > MAX_API_RESPONSE_BYTES:
        raise ValueError("API response exceeds the allowed size.")
    chunks: list[bytes] = []
    received = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        received += len(chunk)
        if received > MAX_API_RESPONSE_BYTES:
            raise ValueError("API response exceeds the allowed size.")
        chunks.append(chunk)
    payload = requests.models.complexjson.loads(b"".join(chunks))
    records = payload.get("results", payload) if isinstance(payload, dict) else payload
    return pd.json_normalize(records if isinstance(records, list) else [records])


@app.middleware("http")
async def protect_api(request: Request, call_next):
    """Apply optional API-key protection, per-IP rate limits, and retention."""

    if request.url.path.startswith("/api/"):
        if APP_API_KEY and not compare_digest(request.headers.get("X-API-Key", ""), APP_API_KEY):
            return JSONResponse(status_code=401, content={"detail": "Authentication is required."})
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        attempts = request_times[client]
        while attempts and now - attempts[0] > 60:
            attempts.popleft()
        if len(attempts) >= RATE_LIMIT_PER_MINUTE:
            return JSONResponse(status_code=429, content={"detail": "Too many requests. Try again shortly."})
        attempts.append(now)
    store.purge_older_than(DATA_RETENTION_HOURS * 3600)
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/datasets/upload")
async def upload_dataset(file: Annotated[UploadFile, File(...)]) -> dict[str, object]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a CSV, JSON, or Parquet file.")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        source = Path(temporary.name)
        size = 0
        while chunk := await file.read(64 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                source.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"Dataset must be {MAX_UPLOAD_MB} MB or smaller.")
            temporary.write(chunk)
    try:
        dataset_id, frame = store.create_from_file(source, file.filename or "dataset")
        dataset_profile = workflow_agent.profile_dataset(frame)
        store.save_profile(dataset_id, dataset_profile)
        return {"dataset_id": dataset_id, "profile": dataset_profile.model_dump(mode="json")}
    except (ValueError, pd.errors.ParserError) as error:
        logger.info("Dataset upload rejected: %s", type(error).__name__)
        raise HTTPException(status_code=400, detail="The file could not be read as a supported dataset.") from error
    finally:
        source.unlink(missing_ok=True)


@app.post("/api/datasets/import-api")
def import_api_dataset(request: ApiImportRequest) -> dict[str, object]:
    try:
        frame = _fetch_json_frame(request.url)
        dataset_id, _ = store.create_from_frame(frame, "api-data.csv")
        dataset_profile = workflow_agent.profile_dataset(frame)
        store.save_profile(dataset_id, dataset_profile)
        return {"dataset_id": dataset_id, "profile": dataset_profile.model_dump(mode="json")}
    except (ValueError, requests.RequestException) as error:
        logger.info("API import rejected: %s", type(error).__name__)
        raise HTTPException(status_code=400, detail="The API data could not be imported.") from error


@app.get("/api/datasets/{dataset_id}/profile")
def get_profile(dataset_id: str) -> dict[str, object]:
    try:
        frame = store.active_frame(dataset_id)
        dataset_profile = workflow_agent.profile_dataset(frame)
        store.save_profile(dataset_id, dataset_profile)
        return dataset_profile.model_dump(mode="json")
    except ValueError as error:
        raise _http_error(error) from error


@app.post("/api/datasets/{dataset_id}/suggest-cleanup")
def suggest_cleanup(dataset_id: str) -> dict[str, object]:
    try:
        plan = workflow_agent.suggest_cleanup(store.active_frame(dataset_id))
        return plan.model_dump(mode="json")
    except (RuntimeError, ValueError) as error:
        raise _http_error(error) from error


@app.post("/api/datasets/{dataset_id}/understand")
def understand_dataset(dataset_id: str) -> dict[str, object]:
    """Give a first-time user a profile-grounded explanation and starter questions."""

    try:
        context = workflow_agent.understand_dataset(store.active_frame(dataset_id))
        return context.model_dump(mode="json")
    except (RuntimeError, ValueError) as error:
        raise _http_error(error) from error


@app.post("/api/datasets/{dataset_id}/apply-plan")
def apply_plan(dataset_id: str, request: ApplyPlanRequest) -> dict[str, object]:
    try:
        cleaned, report = workflow_agent.apply_cleanup(store.active_frame(dataset_id), request.plan, request.approved_ids)
        store.save_cleaned(dataset_id, cleaned, report)
        return {"validation": report.model_dump(mode="json"), "profile": workflow_agent.profile_dataset(cleaned).model_dump(mode="json")}
    except ValueError as error:
        raise _http_error(error) from error


@app.post("/api/datasets/{dataset_id}/stage-postgres")
def stage_postgres(dataset_id: str) -> dict[str, object]:
    """Copy the active dataset to generic PostgreSQL staging on explicit request."""

    try:
        metadata = store.metadata(dataset_id)
        source_name = str(metadata.get("original_name", "dataset"))
        source_format = Path(str(metadata.get("raw", metadata.get("active", "")))).suffix.removeprefix(".") or "unknown"
        receipt = workflow_agent.stage_raw_dataset(
            store.raw_frame(dataset_id), source_name, source_format, get_settings().write_database_config()
        )
        return receipt.model_dump(mode="json")
    except (RuntimeError, ValueError, psycopg2.Error) as error:
        logger.info("PostgreSQL staging failed: %s", type(error).__name__)
        raise HTTPException(status_code=400, detail="The dataset could not be staged in PostgreSQL. Check writer configuration and try again.") from error


@app.post("/api/datasets/{dataset_id}/analyze")
def analyze(dataset_id: str, request: AnalysisRequest) -> dict[str, object]:
    if not request.sql and not request.question:
        raise HTTPException(status_code=400, detail="Provide a question or a SQL query.")
    try:
        frame = store.active_frame(dataset_id)
        dataset_profile = workflow_agent.profile_dataset(frame)
        query, columns, rows = workflow_agent.analyze_dataset(frame, dataset_profile, request.question, request.sql)
        displayed_rows = rows[:100]
        return {
            "sql": query,
            "columns": columns,
            "rows": displayed_rows,
            "interpretation_notes": _result_interpretation_notes(columns, displayed_rows),
        }
    except Exception as error:
        logger.info("Analysis request failed: %s", type(error).__name__)
        raise HTTPException(status_code=400, detail="The analysis could not be completed. Check the question or SQL query.") from error


@app.get("/api/datasets/{dataset_id}/download")
def download_dataset(dataset_id: str) -> FileResponse:
    try:
        path = store.download_path(dataset_id)
        return FileResponse(path, filename="cleaned-dataset" + path.suffix)
    except ValueError as error:
        raise _http_error(error) from error


WEB_ROOT = Path(__file__).resolve().parent / "web"
app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")

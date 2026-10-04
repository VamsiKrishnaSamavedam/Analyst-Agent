# AI Agent — Data Analytics Copilot

A portfolio-ready Python application for practical analytics workflows:

- **API extraction:** download a JSON API response into CSV, JSON, or Parquet.
- **Database analysis:** run guarded, read-only PostgreSQL queries or ask a natural-language question that Gemini translates into schema-grounded SQL.
- **Dataset workbench:** upload a raw CSV, JSON, or Parquet dataset (or import a public JSON API), profile every row, review Gemini cleanup suggestions, approve deterministic changes, validate results, and run safe local SQL analysis.
- **Dataset orientation:** ask the agent to describe the likely purpose and row-level meaning of an unfamiliar dataset, explain its fields, flag caveats, and provide clickable starter questions grounded in the profile.

The included ride-sharing dataset models users, vehicles, rides, payments, and ratings. The project demonstrates agent-assisted analytics with explicit security boundaries rather than an unrestricted autonomous agent.

## Architecture

```text
Browser application
 └─ FastAPI delivery layer ──> original ApplicationDataAgent
     ├─ ETL analyst ──> profile / understand / cleanup plan / approved cleanup
     └─ SQL analyst ──> Gemini DuckDB SQL (or explicit SQL) ──> validator ──> in-memory analysis

CLI
 ├─ extract ──> requests ──> pandas ──> local dataset
 ├─ query   ──> deterministic read-only SQL validator ──> PostgreSQL
 └─ ask     ──> Gemini + live schema context ──> validator ──> PostgreSQL
```

Every database session is opened in PostgreSQL read-only mode. The application also rejects multiple statements and database-changing/admin keywords before execution. Use a PostgreSQL role with `SELECT` permissions only for defence in depth.

## Run the full application

Requires Python 3.12+. Gemini is optional until you ask for AI cleanup suggestions or ask a natural-language analysis question.

```bash
uv sync --group dev
copy .env.example .env
# Add GEMINI_API_KEY to .env for AI suggestions and natural-language questions.
# Choose LLM_THINKING_LEVEL=low, medium, or high (medium is the default).
uv run uvicorn ai_agent.api:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The browser interface supports complete dataset profiling, reviewable cleanup, validation, download, generated SQL analysis, and explicit local SQL against a table named `dataset`.

After profiling an unfamiliar dataset, select **Understand this dataset**. Gemini receives the profile rather than raw records and returns a clearly qualified orientation brief, field guide, caveats, and suggested questions. You can select a suggested question to place it directly into the analysis box.

To run the same app in a container:

```bash
docker compose up --build
```

## Command-line quick start

Requires Python 3.12+ and PostgreSQL.

```bash
uv sync --group dev
copy .env.example .env
# Set the values in .env; do not commit it.
uv run python feed_db.py
uv run ai-agent schema
```

To intentionally replace existing loaded records, prefix the loader command with
`RESET_DATA=true`. It will otherwise preserve existing rows.

Load a public JSON API:

```bash
uv run ai-agent extract https://pokeapi.co/api/v2/pokemon --output-dir data/extract --format csv
```

Run explicit SQL:

```bash
uv run ai-agent query "SELECT DISTINCT payment_method FROM public.payments ORDER BY payment_method"
```

Ask a natural-language question (requires `GEMINI_API_KEY`):

```bash
uv run ai-agent ask "Which five drivers completed the most rides?"
```

Profile a raw dataset before making changes:

```bash
uv run ai-agent profile data/users.csv
```

Apply only explicit, deterministic cleanup actions and receive a before/after
quality report. Ambiguous rules—such as what a missing discount means—remain a
human approval decision:

```bash
uv run ai-agent clean data/users.csv data/cleaned/users.csv --remove-duplicates --trim-text
```

Generate Gemini-powered suggestions without sending raw rows to the model. The
result is an approval-ready JSON plan based on full-dataset profile statistics:

```bash
uv run ai-agent suggest-cleanup data/users.csv data/plans/users-cleanup.json
```

Review the plan, then apply only the suggestion IDs you approve. Every run
returns a before/after validation report:

```bash
uv run ai-agent apply-plan data/users.csv data/plans/users-cleanup.json data/cleaned/users.csv --approve remove-duplicates --approve trim-text
```

## Local PostgreSQL ETL workflow

The original ETL agent now supports a non-destructive first step before any
database load. It compares source columns with an existing PostgreSQL target
table and reports exact mappings, unmapped source fields, and required target
fields that need a human decision:

```bash
uv run ai-agent map data/rides.csv public.rides
```

After a separately approved load, reconcile the source and target row counts:

```bash
uv run ai-agent reconcile data/rides.csv public.rides
```

These commands use the existing `DB_*` settings and the read-only PostgreSQL
layer. They do not insert, update, or delete database records.

To retain an unchanged copy of any uploaded dataset in the generic PostgreSQL
staging area, configure a separate writer role using `DB_WRITE_*` values. The
browser then exposes **Stage raw data in PostgreSQL**. This creates the fixed
`etl_staging.datasets` and `etl_staging.dataset_rows` tables as needed and
stores heterogeneous source rows as JSONB; it never forces raw data into the
portfolio's `users`, `rides`, or `payments` tables.

## Project layout

```text
agents/                # original LangGraph prototype, extended with the production agent layer
models/schema.py        # shared agent state and structured output contracts
utils/                  # shared LLM selector and operational tools
src/ai_agent/           # FastAPI delivery layer, CLI, and deterministic safety services
data/                  # portfolio ride-sharing source data
feed_db.py             # local PostgreSQL schema/data bootstrap
tests/                 # focused regression tests
```

## Security notes

- Keep secrets exclusively in `.env` or a secret manager; `.env.example` contains placeholders only.
- The natural-language SQL route supplies live schema metadata to the model and validates the returned statement before execution.
- Generated Pandas code from the original experiment is not part of the production CLI, because executing model-generated code is not an acceptable default trust boundary.

## Development

```bash
uv run pytest
uv run ruff check src tests
```

## Deployment boundary

The app is container-ready and includes streamed upload limits, bounded API imports, optional `APP_API_KEY` protection, per-IP request limiting, local dataset expiry, disabled DuckDB external access, and spreadsheet-safe CSV exports. For public recruiter deployment, set `GEMINI_API_KEY` and `APP_API_KEY` only in the host secret manager; use encrypted persistent storage; put the service behind an HTTPS reverse proxy/WAF; and configure egress controls that prevent private-network access. The existing static portfolio demo remains useful for discovery, but it cannot securely host this Python backend or the Gemini secret by itself.

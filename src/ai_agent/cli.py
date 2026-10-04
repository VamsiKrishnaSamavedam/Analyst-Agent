"""Command-line interface for AI Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The original prototype keeps its shared LLM selector in the top-level
# ``utils`` package. Local CLI execution keeps that package on the import path
# instead of copying the selector into the delivery layer.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.aiagent import ApplicationDataAgent
from ai_agent.config import get_settings
from ai_agent.models import CleanupPlan, TransformationSuggestion
from ai_agent.services.database import ReadOnlyDatabase, format_rows
from ai_agent.services.etl import extract_api_data
from ai_agent.services.postgres_etl import validate_target_table_name
from ai_agent.services.quality import (
    apply_cleanup_plan,
    profile,
    read_dataset,
    write_dataset,
)
from ai_agent.services.recommender import CleanupRecommender
from ai_agent.services.sql_assistant import SQLAssistant


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-agent", description="Safe API ETL and PostgreSQL analytics.")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="Download a JSON API response into a local dataset.")
    extract.add_argument("url", help="JSON API endpoint")
    extract.add_argument("--output-dir", default="data/extract", type=Path)
    extract.add_argument("--format", default="csv", choices=("csv", "json", "parquet"))
    extract.add_argument("--filename", default="extracted_data")

    query = commands.add_parser("query", help="Execute an explicit read-only PostgreSQL query.")
    query.add_argument("sql", help="Single SELECT or read-only WITH query")

    ask = commands.add_parser("ask", help="Ask a natural-language question about the PostgreSQL database.")
    ask.add_argument("question")
    commands.add_parser("schema", help="Print the available public database schema.")

    profile_command = commands.add_parser("profile", help="Profile dataset quality before transformations.")
    profile_command.add_argument("path", type=Path)

    clean = commands.add_parser("clean", help="Apply only selected, deterministic cleanup actions.")
    clean.add_argument("input", type=Path)
    clean.add_argument("output", type=Path)
    clean.add_argument("--remove-duplicates", action="store_true")
    clean.add_argument("--trim-text", action="store_true")

    suggest = commands.add_parser("suggest-cleanup", help="Ask Gemini for an approval-ready cleanup plan.")
    suggest.add_argument("input", type=Path)
    suggest.add_argument("output", type=Path)

    apply_plan = commands.add_parser("apply-plan", help="Apply approved suggestions from a cleanup plan.")
    apply_plan.add_argument("input", type=Path)
    apply_plan.add_argument("plan", type=Path)
    apply_plan.add_argument("output", type=Path)
    apply_plan.add_argument("--approve", action="append", default=[], help="Suggestion ID to approve; repeat for each.")

    mapping = commands.add_parser("map", help="Compare a local dataset with a PostgreSQL target table before loading.")
    mapping.add_argument("input", type=Path)
    mapping.add_argument("target_table", help="Target table, for example public.rides")

    reconcile = commands.add_parser("reconcile", help="Compare local source rows with an existing PostgreSQL target table.")
    reconcile.add_argument("input", type=Path)
    reconcile.add_argument("target_table", help="Target table, for example public.rides")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()

    if args.command == "extract":
        result = extract_api_data(args.url, args.output_dir, args.format, args.filename)
        print(f"Saved extracted data to {result.resolve()}")
        return

    if args.command == "profile":
        print(profile(read_dataset(args.path)).model_dump_json(indent=2))
        return

    if args.command == "clean":
        suggestions = []
        if args.remove_duplicates:
            suggestions.append(TransformationSuggestion(
                id="remove-duplicates",
                operation="remove_duplicates",
                reason="Explicit CLI selection",
                impact_estimate="Remove exact duplicate rows",
                confidence="high",
            ))
        if args.trim_text:
            suggestions.append(TransformationSuggestion(
                id="trim-text",
                operation="trim_text",
                reason="Explicit CLI selection",
                impact_estimate="Trim surrounding whitespace in text columns",
                confidence="high",
            ))
        plan = CleanupPlan(dataset_summary="Explicit local cleanup request", suggestions=suggestions)
        cleaned, report = apply_cleanup_plan(
            read_dataset(args.input),
            plan,
            {item.id for item in suggestions},
        )
        write_dataset(cleaned, args.output)
        print(json.dumps({"validation": report.model_dump(), "output": str(args.output)}, indent=2))
        return

    if args.command == "suggest-cleanup":
        dataset_profile = profile(read_dataset(args.input))
        plan = CleanupRecommender(settings.gemini_api_key).recommend(dataset_profile)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved approval-ready cleanup plan to {args.output.resolve()}")
        return

    if args.command == "apply-plan":
        plan = CleanupPlan.model_validate_json(args.plan.read_text(encoding="utf-8"))
        cleaned, report = apply_cleanup_plan(read_dataset(args.input), plan, set(args.approve))
        write_dataset(cleaned, args.output)
        print(json.dumps({"validation": report.model_dump(), "output": str(args.output)}, indent=2))
        return

    if args.command in {"map", "reconcile"}:
        source = read_dataset(args.input)
        schema_name, table_name = validate_target_table_name(args.target_table)
        database = ReadOnlyDatabase(settings.database_config())
        workflow_agent = ApplicationDataAgent(settings.gemini_api_key)
        target_name = f"{schema_name}.{table_name}"
        if args.command == "map":
            report = workflow_agent.assess_load_readiness(source, target_name, database.describe_table(schema_name, table_name))
        else:
            report = workflow_agent.reconcile_load(len(source), database.count_rows(schema_name, table_name), target_name)
        print(report.model_dump_json(indent=2))
        return

    database = ReadOnlyDatabase(settings.database_config())
    if args.command == "schema":
        print(database.describe_public_schema())
    elif args.command == "query":
        columns, rows = database.execute(args.sql)
        print(format_rows(columns, rows))
    else:
        sql, columns, rows = SQLAssistant(database, settings.gemini_api_key).answer(args.question)
        print(f"Generated SQL:\n{sql}\n\nResults:\n{format_rows(columns, rows)}")

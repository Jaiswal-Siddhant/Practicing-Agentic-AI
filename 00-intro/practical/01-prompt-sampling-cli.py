#!/usr/bin/env python
"""
Prompt & Sampling Playground

CLI playground based on 01-prompt-sampling-cli.md.

Examples:
  python 00-intro/practical/01-prompt-sampling-cli.py --dry-run
  python 00-intro/practical/01-prompt-sampling-cli.py --prompt "Explain top-p" --repeat 3
  python 00-intro/practical/01-prompt-sampling-cli.py --structured --induce-schema-failure --dry-run
  python 00-intro/practical/01-prompt-sampling-cli.py compare --db runs.sqlite

Real model calls use LangChain with Google Gemini. Set GOOGLE_API_KEY in .env
or in your shell before running without --dry-run.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import time
import textwrap
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel, Field
except Exception:
    BaseModel = None
    Field = None


DEFAULT_PROMPT = "Explain temperature and top-p sampling to a curious beginner."
DEFAULT_SYSTEMS = [
    "You are concise, precise, and practical.",
    "You are playful, vivid, and analogy-driven.",
]
DEFAULT_TEMPERATURES = [0.0, 0.7, 1.1]
DEFAULT_TOP_PS = [1.0, 0.9]
DEFAULT_MAX_TOKENS = [180]
DEFAULT_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.6-flash")


if BaseModel is not None:

    class StructuredAnswer(BaseModel):
        answer: str = Field(min_length=20)
        bullets: list[str] = Field(min_length=2, max_length=5)
        confidence: float = Field(ge=0.0, le=1.0)
else:
    StructuredAnswer = None


@dataclass(frozen=True)
class RunConfig:
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    system: str
    prompt: str
    structured: bool


@dataclass
class RunRecord:
    id: str
    created_at: str
    config: RunConfig
    output: str
    latency_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    error: str | None
    repaired: bool

    @property
    def output_hash(self) -> str:
        return hashlib.sha256(self.output.encode("utf-8")).hexdigest()[:16]

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["config"] = asdict(self.config)
        data["output_hash"] = self.output_hash
        return data


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_systems(values: list[str] | None) -> list[str]:
    if not values:
        return DEFAULT_SYSTEMS
    systems: list[str] = []
    for value in values:
        systems.extend(part.strip() for part in value.split("||") if part.strip())
    return systems or DEFAULT_SYSTEMS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_configs(args: argparse.Namespace) -> list[RunConfig]:
    configs: list[RunConfig] = []
    for _ in range(args.repeat):
        for model in args.models:
            for temperature in parse_csv_floats(args.temperatures):
                for top_p in parse_csv_floats(args.top_ps):
                    for max_tokens in parse_csv_ints(args.max_tokens):
                        for system in parse_systems(args.system):
                            configs.append(
                                RunConfig(
                                    model=model,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_tokens=max_tokens,
                                    system=system,
                                    prompt=args.prompt,
                                    structured=args.structured,
                                )
                            )
    return configs


def structured_instruction(induce_failure: bool) -> str:
    instruction = (
        "Return only JSON with keys answer, bullets, and confidence. "
        "answer is a string, bullets is an array of 2 to 5 strings, "
        "confidence is a number from 0 to 1."
    )
    if induce_failure:
        return instruction + " Intentionally make confidence a phrase on this first attempt."
    return instruction


def dry_run_output(config: RunConfig, induce_schema_failure: bool) -> tuple[str, dict[str, int]]:
    seed = f"{config.prompt}|{config.system}|{config.temperature}|{config.top_p}|{config.max_tokens}"
    rng = random.Random(seed)
    if config.structured:
        confidence: Any = "very high" if induce_schema_failure else round(rng.uniform(0.55, 0.95), 2)
        output = {
            "answer": (
                f"Sampling changes which likely next token gets picked. "
                f"Temperature={config.temperature} broadens or narrows surprises; "
                f"top_p={config.top_p} trims the candidate pool."
            ),
            "bullets": [
                "Low temperature is steadier and more repeatable.",
                "Higher temperature permits more unusual word choices.",
                "Lower top-p ignores the long tail of unlikely tokens.",
            ],
            "confidence": confidence,
        }
        text = json.dumps(output, indent=2)
    else:
        adjectives = ["steady", "lively", "focused", "surprising", "compact"]
        chosen = ", ".join(rng.sample(adjectives, 3))
        text = (
            f"[dry run] With temperature {config.temperature} and top-p {config.top_p}, "
            f"this answer should feel {chosen}. The model '{config.model}' would receive "
            f"a max token budget of {config.max_tokens}."
        )
    prompt_tokens = len(config.prompt.split()) + len(config.system.split())
    completion_tokens = len(text.split())
    return text, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def usage_from_response(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {})

    input_tokens = usage.get("input_tokens", token_usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", token_usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens", token_usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def gemini_chat_model(config: RunConfig, repair_payload: str | None = None) -> Any:
    load_dotenv()
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError("Install langchain-google-genai or run with --dry-run.") from exc
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY in .env or run with --dry-run.")

    return ChatGoogleGenerativeAI(
        model=config.model,
        temperature=config.temperature if repair_payload is None else 0,
        top_p=config.top_p,
        max_output_tokens=config.max_tokens,
    )


def call_gemini(
    config: RunConfig,
    args: argparse.Namespace,
    repair_payload: str | None = None,
) -> tuple[str, dict[str, int | None]]:
    chat_model = gemini_chat_model(config, repair_payload)
    system = config.system
    prompt = config.prompt

    if config.structured:
        system = f"{system}\n\n{structured_instruction(args.induce_schema_failure and repair_payload is None)}"
    if repair_payload is not None:
        system = "You repair JSON so it exactly matches the requested schema. Return only JSON."
        prompt = repair_payload

    response = chat_model.invoke(
        [
            ("system", system),
            ("human", prompt),
        ]
    )
    text = response.content if isinstance(response.content, str) else json.dumps(response.content)
    return text or "", usage_from_response(response)


def extract_json(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def validate_structured(text: str) -> None:
    if StructuredAnswer is None:
        raise RuntimeError("Pydantic is required for --structured validation.")
    StructuredAnswer.model_validate(extract_json(text))


def terminal_width() -> int:
    return min(shutil.get_terminal_size((100, 24)).columns, 120)


def rule(title: str = "") -> str:
    width = terminal_width()
    if not title:
        return "-" * width
    label = f" {title} "
    return label.center(width, "-")


def wrap_block(text: str, indent: str = "  ") -> str:
    width = max(50, terminal_width() - len(indent))
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width, replace_whitespace=False))
    return "\n".join(f"{indent}{line}" if line else "" for line in lines)


def format_output(text: str) -> str:
    try:
        parsed = extract_json(text)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        return text.strip()


def print_record_output(index: int, total: int, record: RunRecord) -> None:
    cfg = record.config
    status = "error" if record.error else "ok"
    if record.repaired:
        status += " + repaired"

    print()
    print(rule(f"Run {index}/{total}"))
    print(f"  status      : {status}")
    print(f"  model       : {cfg.model}")
    print(f"  sampling    : temperature={cfg.temperature}, top_p={cfg.top_p}, max_tokens={cfg.max_tokens}")
    print(f"  latency     : {record.latency_ms}ms")
    print(f"  token usage : prompt={record.prompt_tokens}, completion={record.completion_tokens}, total={record.total_tokens}")
    print(f"  output hash : {record.output_hash}")
    if record.error:
        print("  error       :")
        print(wrap_block(record.error, indent="    "))
    if record.output:
        print("  output      :")
        print(wrap_block(format_output(record.output), indent="    "))
    print(rule())


def repair_structured_output(
    text: str,
    error: Exception,
    config: RunConfig,
    args: argparse.Namespace,
) -> tuple[str, bool, str | None]:
    repair_prompt = (
        "Repair this response to match the schema exactly.\n\n"
        "Schema: {\"answer\": string, \"bullets\": array of 2-5 strings, \"confidence\": number 0..1}\n\n"
        f"Validation error: {error}\n\n"
        f"Original response:\n{text}"
    )
    try:
        if args.dry_run:
            repaired, _usage = dry_run_output(config, False)
        else:
            repaired, _usage = call_gemini(config, args, repair_payload=repair_prompt)
        validate_structured(repaired)
        return repaired, True, None
    except Exception as repair_error:
        return text, False, f"repair failed: {repair_error}"


def execute_one(config: RunConfig, args: argparse.Namespace) -> RunRecord:
    started = time.perf_counter()
    error: str | None = None
    repaired = False
    usage: dict[str, int | None] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    try:
        if args.dry_run:
            output, usage = dry_run_output(config, args.induce_schema_failure)
        else:
            output, usage = call_gemini(config, args)
        if config.structured:
            try:
                validate_structured(output)
            except Exception as validation_error:
                output, repaired, error = repair_structured_output(output, validation_error, config, args)
    except Exception as exc:
        output = ""
        error = str(exc)

    return RunRecord(
        id=str(uuid.uuid4()),
        created_at=utc_now(),
        config=config,
        output=output,
        latency_ms=int((time.perf_counter() - started) * 1000),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        error=error,
        repaired=repaired,
    )


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            temperature REAL NOT NULL,
            top_p REAL NOT NULL,
            max_tokens INTEGER NOT NULL,
            system TEXT NOT NULL,
            prompt TEXT NOT NULL,
            structured INTEGER NOT NULL,
            output TEXT NOT NULL,
            output_hash TEXT NOT NULL,
            latency_ms INTEGER NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            error TEXT,
            repaired INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_record(conn: sqlite3.Connection, record: RunRecord, jsonl_path: Path) -> None:
    cfg = record.config
    conn.execute(
        """
        INSERT INTO runs (
            id, created_at, model, temperature, top_p, max_tokens, system, prompt,
            structured, output, output_hash, latency_ms, prompt_tokens,
            completion_tokens, total_tokens, error, repaired
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.id,
            record.created_at,
            cfg.model,
            cfg.temperature,
            cfg.top_p,
            cfg.max_tokens,
            cfg.system,
            cfg.prompt,
            int(cfg.structured),
            record.output,
            record.output_hash,
            record.latency_ms,
            record.prompt_tokens,
            record.completion_tokens,
            record.total_tokens,
            record.error,
            int(record.repaired),
        ),
    )
    conn.commit()
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_jsonable(), ensure_ascii=False) + "\n")


def summarize_records(records: list[RunRecord]) -> None:
    if not records:
        print("No runs were executed.")
        return
    print()
    print(rule("Run Summary"))
    headers = ("#", "status", "model", "temp", "top_p", "max", "latency", "tokens", "hash")
    widths = (4, 13, 18, 7, 7, 6, 9, 8, 16)
    header_line = " ".join(value.ljust(width) for value, width in zip(headers, widths))
    print(header_line)
    print(" ".join("-" * width for width in widths))
    for index, record in enumerate(records, 1):
        cfg = record.config
        status = "error" if record.error else "ok"
        if record.repaired:
            status += "+repair"
        values = (
            f"{index:02d}",
            status,
            cfg.model[:18],
            str(cfg.temperature),
            str(cfg.top_p),
            str(cfg.max_tokens),
            f"{record.latency_ms}ms",
            str(record.total_tokens),
            record.output_hash,
        )
        print(" ".join(value.ljust(width) for value, width in zip(values, widths)))
        if record.error:
            print(wrap_block(f"error: {record.error}", indent="  "))
    print(rule())


def compare_outputs(rows: list[tuple[Any, ...]]) -> None:
    if len(rows) < 2:
        print("Need at least two saved runs to compare.")
        return
    print("\nLatest saved outputs")
    print(rule("Latest Saved Outputs"))
    for index, row in enumerate(rows, 1):
        run_id, created_at, model, temperature, top_p, max_tokens, output_hash, output = row
        preview = " ".join(output.split())[: min(terminal_width() - 6, 110)]
        print(
            f"{index:02d}. {run_id[:8]} {created_at} {model} "
            f"temp={temperature} top_p={top_p} max={max_tokens} hash={output_hash}"
        )
        print(wrap_block(preview, indent="    "))

    first = rows[0][-1].splitlines()
    second = rows[1][-1].splitlines()
    print("\nDiff between latest two")
    print(rule("Diff Between Latest Two"))
    for line in difflib.unified_diff(first, second, fromfile=rows[0][0][:8], tofile=rows[1][0][:8], lineterm=""):
        print(line)


def run_compare(args: argparse.Namespace) -> None:
    conn = init_db(Path(args.db))
    rows = conn.execute(
        """
        SELECT id, created_at, model, temperature, top_p, max_tokens, output_hash, output
        FROM runs
        WHERE output <> ''
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (args.limit,),
    ).fetchall()
    compare_outputs(rows)


def run_playground(args: argparse.Namespace) -> None:
    configs = build_configs(args)
    conn = init_db(Path(args.db))
    records: list[RunRecord] = []

    for index, config in enumerate(configs, 1):
        print(
            f"[{index}/{len(configs)}] model={config.model} temp={config.temperature} "
            f"top_p={config.top_p} max={config.max_tokens}"
        )
        record = execute_one(config, args)
        save_record(conn, record, Path(args.jsonl))
        records.append(record)
        if not args.quiet:
            print_record_output(index, len(configs), record)
        if args.sleep:
            time.sleep(args.sleep)

    summarize_records(records)
    print(f"\nSaved JSONL: {Path(args.jsonl).resolve()}")
    print(f"Saved SQLite: {Path(args.db).resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt and sampling playground.")
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt to send repeatedly.")
    parser.add_argument("--system", action="append", help="System instruction. Use multiple flags or separate values with ||.")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL], help="Model names to test.")
    parser.add_argument("--temperatures", default=",".join(map(str, DEFAULT_TEMPERATURES)))
    parser.add_argument("--top-ps", default=",".join(map(str, DEFAULT_TOP_PS)))
    parser.add_argument("--max-tokens", default=",".join(map(str, DEFAULT_MAX_TOKENS)))
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the full parameter grid this many times.")
    parser.add_argument("--structured", action="store_true", help="Ask for JSON and validate it with Pydantic.")
    parser.add_argument("--induce-schema-failure", action="store_true", help="Trigger the structured-output repair path.")
    parser.add_argument("--dry-run", action="store_true", help="Run without calling an API.")
    parser.add_argument("--db", default="runs.sqlite", help="SQLite output path.")
    parser.add_argument("--jsonl", default="runs.jsonl", help="JSONL output path.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Pause between API calls to respect rate limits.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary.")
    parser.set_defaults(func=run_playground)

    compare = subparsers.add_parser("compare", help="Compare latest saved outputs.")
    compare.add_argument("--db", default="runs.sqlite", help="SQLite database path.")
    compare.add_argument("--limit", type=int, default=6, help="Number of latest runs to display.")
    compare.set_defaults(func=run_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

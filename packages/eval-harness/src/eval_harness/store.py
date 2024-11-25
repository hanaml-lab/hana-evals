"""Run folders: scores beside the exact settings that produced them.

    runs/2026-09-30T14-02-11/
        settings.json     every setting of the run
        stages.jsonl      one row per stage per run
        runs.jsonl        one row per run
        results.duckdb    the same rows as tables, when DuckDB is installed

Each stage row carries an OpenTelemetry-compatible trace id (32 hex chars, one per run), so a
row can be joined to traces exported by the agent.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime
from pathlib import Path


def trace_id() -> str:
    return secrets.token_hex(16)


class Store:
    def __init__(self, root: str | Path, name: str | None = None):
        name = name or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=False)

    def save(self, settings: dict, stages: list[dict], runs: list[dict]) -> Path:
        (self.dir / "settings.json").write_text(json.dumps(settings, indent=2, sort_keys=True, default=str))
        for name, rows in (("stages", stages), ("runs", runs)):
            with (self.dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, default=str) + "\n")
        try:
            import duckdb
        except ImportError:
            return self.dir
        con = duckdb.connect(str(self.dir / "results.duckdb"))
        for name in ("stages", "runs"):
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM read_json_auto(?)", [str(self.dir / f"{name}.jsonl")])
        con.close()
        return self.dir


def load(run_dir: str | Path) -> tuple[dict, list[dict], list[dict]]:
    d = Path(run_dir)
    settings = json.loads((d / "settings.json").read_text())

    def rows(name: str) -> list[dict]:
        return [json.loads(line) for line in (d / f"{name}.jsonl").read_text().splitlines() if line]

    return settings, rows("stages"), rows("runs")

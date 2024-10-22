"""Jobs as graphs of stages.

A stage names a tool, the stages it reads from, and how costly a mistake there is to undo.
Tools are pure functions of their inputs, so the reference answer for every stage of a job can
be computed, and a wrong output early on visibly corrupts the stages that depend on it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TOOLS: dict[str, Callable[..., Any]] = {}


def tool(name: str):
    def register(fn):
        TOOLS[name] = fn
        return fn
    return register


@dataclass(frozen=True)
class Stage:
    id: str
    tool: str
    deps: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    check: str = "code"  # code | judge
    undo_cost: float = 1.0  # relative cost of undoing a wrong result here
    critical: bool = False  # a wrong final value here harms the business
    instructions: str = ""


@dataclass(frozen=True)
class Job:
    id: str
    stages: tuple[Stage, ...]
    inputs: dict = field(default_factory=dict)

    def __post_init__(self):
        ids = [s.id for s in self.stages]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.id}: duplicate stage ids")
        for s in self.stages:
            for d in s.deps:
                if d not in ids:
                    raise ValueError(f"{self.id}: stage {s.id} depends on unknown {d}")
        self.order()  # raises on a cycle

    def stage(self, sid: str) -> Stage:
        return next(s for s in self.stages if s.id == sid)

    def order(self) -> list[Stage]:
        done: list[str] = []
        pending = {s.id: set(s.deps) for s in self.stages}
        while pending:
            ready = sorted(k for k, deps in pending.items() if deps <= set(done))
            if not ready:
                raise ValueError(f"{self.id}: stages form a cycle")
            for k in ready:
                done.append(k)
                del pending[k]
        return [self.stage(k) for k in done]

    def dependants(self, sid: str) -> set[str]:
        """Every stage that reads, directly or not, from ``sid``."""
        out: set[str] = set()
        frontier = {sid}
        while frontier:
            nxt = {s.id for s in self.stages if set(s.deps) & frontier} - out
            out |= nxt
            frontier = nxt
        return out

    def run_tool(self, stage: Stage, upstream: dict[str, Any]) -> Any:
        args = {d: upstream[d] for d in stage.deps}
        return TOOLS[stage.tool](inputs=self.inputs, params=stage.params, **args)

    def reference(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for s in self.order():
            out[s.id] = self.run_tool(s, out)
        return out


@dataclass
class Suite:
    name: str
    jobs: list[Job]
    adapter: str = "native"
    settings: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        path = Path(path)
        if path.is_dir():
            files = sorted(path.glob("*.yaml"))
            suites = [cls.load(f) for f in files]
            return cls(path.name, [j for s in suites for j in s.jobs])
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        adapter = raw.get("adapter", "native")
        if adapter == "inspect":
            from eval_harness.inspect_adapter import load_inspect

            return load_inspect(raw)
        import eval_harness.tools  # noqa: F401  registers the built-in tools

        jobs = []
        for j in raw["jobs"]:
            stages = tuple(Stage(id=s["id"], tool=s["tool"], deps=tuple(s.get("deps", [])),
                                 params=s.get("params", {}), check=s.get("check", "code"),
                                 undo_cost=float(s.get("undo_cost", 1.0)), critical=bool(s.get("critical", False)),
                                 instructions=s.get("instructions", "")) for s in raw["stages"])
            jobs.append(Job(j["id"], stages, j.get("inputs", {})))
        return cls(raw.get("name", path.stem), jobs, adapter, raw.get("settings", {}))

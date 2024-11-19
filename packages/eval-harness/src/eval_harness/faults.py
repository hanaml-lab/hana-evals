"""Fault injection: replace a tool result with a wrong one at a chosen stage."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from eval_harness.agents import corrupt


@dataclass(frozen=True)
class Fault:
    stage: str
    seed: int = 0

    def apply(self, value: Any) -> Any:
        wrong = corrupt(value, random.Random(f"fault:{self.stage}:{self.seed}"))
        return wrong if wrong != value else ["INJECTED"]


def recovery_steps(trace: list[dict], fault_stage: str) -> int | None:
    """Agent steps from the injected fault until every stage output is correct again.

    ``None`` when the run never recovered.
    """
    start = next((i for i, row in enumerate(trace) if row["stage"] == fault_stage), None)
    if start is None:
        return None
    wrong: set[str] = set()
    for i, row in enumerate(trace[start:]):
        if row["correct"]:
            wrong.discard(row["stage"])
        else:
            wrong.add(row["stage"])
        if i > 0 and not wrong:
            return i
    return None

"""Where a limited review budget should go.

A check on a stage compares its output with what the stage's tool gives on the inputs it actually
received, so it catches the errors made at that stage and nothing upstream. The value of checking a
stage is therefore the loss its own errors cause when they go unchecked: the undo cost of every
stage they leave wrong, with a critical stage counted twice.

With a budget of, say, 20% of a job's stages, ``dependency`` placement checks first the stages with
the largest such loss. It measures the loss rather than guessing it from the graph: it corrupts the
stage's reference output the way agents err (``agents.corrupt``), replays the job's tools, which
are pure functions, through every dependant, and averages over a few seeds. A stage with many
dependants whose errors rarely change them ranks low; a narrow stage whose errors flip a critical
decision ranks high. ``uniform`` spaces checks evenly through the job; ``last`` checks only the
final stages, which is how most benchmarks score.
"""

from __future__ import annotations

import math
import random

from eval_harness.agents import corrupt
from eval_harness.checks import equal
from eval_harness.stages import Job

SAMPLES = 8  # corrupted replays per stage; the tools are pure, so the estimate is exact for a seed set


def budget_count(job: Job, budget: float) -> int:
    if not 0 <= budget <= 1:
        raise ValueError("review budget is a fraction of stages between 0 and 1")
    return max(1, math.ceil(budget * len(job.stages))) if budget > 0 else 0


def propagated_loss(job: Job, stage_id: str, samples: int = SAMPLES) -> float:
    """Mean undo cost of the stages left wrong when ``stage_id`` alone errs and nothing is checked."""
    reference = job.reference()
    weight = {s.id: s.undo_cost * (2 if s.critical else 1) for s in job.stages}
    total = 0.0
    for seed in range(samples):
        out: dict = {}
        for s in job.order():
            value = job.run_tool(s, out)
            out[s.id] = corrupt(value, random.Random(f"placement:{job.id}:{stage_id}:{seed}")) \
                if s.id == stage_id else value
        total += sum(weight[sid] for sid, value in out.items() if not equal(value, reference[sid]))
    return total / samples


def dependency(job: Job, budget: float) -> set[str]:
    n = budget_count(job, budget)
    if n == 0:
        return set()
    loss = {s.id: propagated_loss(job, s.id) for s in job.stages}
    return set(sorted(loss, key=lambda sid: (-loss[sid], sid))[:n])


def uniform(job: Job, budget: float) -> set[str]:
    n = budget_count(job, budget)
    order = [s.id for s in job.order()]
    if n == 0:
        return set()
    step = len(order) / n
    return {order[min(len(order) - 1, int(step * (i + 1)) - 1)] for i in range(n)}


def last(job: Job, budget: float) -> set[str]:
    n = budget_count(job, budget)
    return {s.id for s in job.order()[len(job.stages) - n:]} if n else set()


STRATEGIES = {"dependency": dependency, "uniform": uniform, "last": last}


def place(job: Job, budget: float, strategy: str) -> set[str]:
    try:
        return STRATEGIES[strategy](job, budget)
    except KeyError:
        raise ValueError(f"placement is one of {', '.join(STRATEGIES)}") from None

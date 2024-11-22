"""Where a limited review budget should go.

With a budget of, say, 20% of a job's stages, ``dependency`` placement checks first the stages
whose mistakes would spread furthest and cost most to undo: score = (1 + dependants) x undo
cost, with critical stages always considered. ``uniform`` spaces checks evenly through the
job; ``last`` checks only the final stages, which is how most benchmarks score.
"""

from __future__ import annotations

import math

from eval_harness.stages import Job


def budget_count(job: Job, budget: float) -> int:
    if not 0 <= budget <= 1:
        raise ValueError("review budget is a fraction of stages between 0 and 1")
    return max(1, math.ceil(budget * len(job.stages))) if budget > 0 else 0


def dependency(job: Job, budget: float) -> set[str]:
    n = budget_count(job, budget)
    scored = sorted(job.stages, key=lambda s: (-(1 + len(job.dependants(s.id))) * s.undo_cost
                                               * (2 if s.critical else 1), s.id))
    return {s.id for s in scored[:n]}


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

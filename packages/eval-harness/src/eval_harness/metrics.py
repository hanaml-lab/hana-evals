"""Run-level metrics."""

from __future__ import annotations

from collections import defaultdict
from math import comb
from statistics import median


def pass_hat_k(successes: int, n: int, k: int) -> float:
    """Unbiased estimate of the chance that k fresh repeats all succeed, from n runs with c
    successes: C(c, k) / C(n, k) (as in tau-bench)."""
    if k > n:
        raise ValueError(f"need at least k={k} runs, have {n}")
    return comb(successes, k) / comb(n, k)


def pass_k(runs: list[dict], k: int) -> float:
    """Mean of pass^k over tasks; ``runs`` rows have ``job`` and ``success``."""
    by_job: dict[str, list[bool]] = defaultdict(list)
    for r in runs:
        by_job[r["job"]].append(bool(r["success"]))
    if not by_job:
        return 0.0
    return sum(pass_hat_k(sum(v), len(v), k) for v in by_job.values()) / len(by_job)


def undetected_critical_per_100(runs: list[dict]) -> float:
    if not runs:
        return 0.0
    return 100.0 * sum(r["undetected_critical"] for r in runs) / len(runs)


def reviewer_load(runs: list[dict]) -> float:
    """Checks performed per completed job."""
    done = [r for r in runs if r["success"]]
    return sum(r["checks"] for r in runs) / max(len(done), 1)


def median_recovery(runs: list[dict]) -> float | None:
    values = [r["recovery_steps"] for r in runs if r.get("recovery_steps") is not None]
    return float(median(values)) if values else None

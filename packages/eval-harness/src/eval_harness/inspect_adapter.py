"""Run tau-bench and SWE-bench Verified through Inspect (``inspect_ai`` and ``inspect_evals``).

Those benchmarks need their own environments (a simulated airline backend, a Docker sandbox per
repository), so their jobs are single-stage: Inspect runs the task, and the harness records the
score per sample and repeat, which is what pass^k needs.
"""

from __future__ import annotations

from eval_harness.stages import Job, Stage, Suite, tool


def load_inspect(raw: dict) -> Suite:
    try:
        import inspect_ai  # noqa: F401
    except ImportError as exc:
        raise ImportError('Inspect suites need: pip install "eval-harness[inspect]"') from exc
    task, samples = raw["task"], raw.get("samples", [])
    jobs = [Job(f"{task}:{s}", (Stage("solve", "inspect_task", params={"task": task, "sample": s},
                                      critical=True),), {}) for s in samples]
    return Suite(raw.get("name", task), jobs, "inspect", raw.get("settings", {}))


@tool("inspect_task")
def inspect_task(inputs, params):  # pragma: no cover - needs Inspect and a sandbox
    from inspect_ai import eval as inspect_eval

    logs = inspect_eval(params["task"], sample_id=params["sample"], display="none")
    return logs[0].results.scores[0].metrics["accuracy"].value

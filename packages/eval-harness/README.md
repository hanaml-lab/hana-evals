# eval-harness

Part of [hana-evals](../../README.md). Judged stages are checked with a judge calibrated by
[`judge-cal`](../judge-cal).

Most agent benchmarks score a single answer at the end. Real jobs run for dozens of steps, and a mistake in step 3
is cheaper to catch than the same mistake found in step 30. eval-harness runs long jobs as stages with
dependencies, scores each stage, injects faults to see how quickly a run recovers, and decides where a limited
review budget should go. It is the harness behind the lab's long-job evaluation suite and is used to evaluate
`judge-7b`.

## Install

```bash
# from the root of hana-evals
pip install "./packages/eval-harness[duckdb]"     # add [inspect] for tau-bench and SWE-bench
docker build -f packages/eval-harness/docker/sandbox.Dockerfile -t eval-sandbox .   # each run gets a fresh sandbox
```

## Quickstart

```bash
cd packages/eval-harness
eval-harness run suites/back-office/ --agent configs/simulated.yaml --repeats 8 --out runs/
eval-harness report runs/<run folder> --metric pass^8 --metric undetected-critical
```

`configs/simulated.yaml` is a seeded agent that runs offline; `configs/agent.yaml` calls `Qwen/Qwen3-32B` through an
OpenAI-compatible endpoint (`OPENAI_BASE_URL`).

```python
from eval_harness import Suite, Runner

suite = Suite.load("suites/tau-bench-airline.yaml")    # needs the [inspect] extra
runner = Runner(agent="configs/agent.yaml", repeats=8, review_budget=0.2, placement="dependency")
result = runner.run(suite)
print(result.pass_k(8), result.undetected_critical_per_100)
```

Every run directory holds the scores and the exact settings that produced them (`settings.json`), so a number can
be reproduced from the folder alone.

## How it works

- **Stages.** A job is a graph of stages, each with inputs, outputs and a check. A check may be code (does the ledger balance?) or a judge.
- **Placement.** A check catches the errors made at its own stage, so with a review budget of, say, 20% of stages, `dependency` placement checks first the stages whose errors cost most downstream. It measures that cost: it corrupts each stage's reference output the way agents err, replays the job's pure tools through every dependant and sums the undo cost of the stages left wrong, counting critical stages twice.
- **Faults.** `faults.py` replaces a tool result with a wrong one at a chosen stage (`--fault <stage>`), then measures the steps until the run is back on track.
- **Perturbation test.** A run that passes its checks can still be a reward hack. `perturbation.PerturbationTest` re-runs the agent on renamed and rescaled instances of the job, whose references the tools recompute, standardises the drop in pass rate against honest runs and flags it above a split-conformal threshold ([VPT](https://hanaml.com/research/papers/detecting-reward-hacking)).
- **pass^k.** A task passes only if all k repeats succeed, which rewards consistency rather than one lucky run.
- **Store.** Results go to DuckDB (and JSON Lines), one row per stage per run, with OpenTelemetry trace ids.

## Configuration

```yaml
# configs/agent.yaml
model: Qwen/Qwen3-32B
max_steps: 80
review_budget: 0.2
placement: dependency     # or: uniform, last
judge: hanaml/judge-7b
```

## Benchmarks

Same model and the same review budget in both columns.

| Metric | Fixed checkpoints | Dependency-aware placement |
| --- | --- | --- |
| Undetected critical errors per 100 runs | 7.8 | 2.9 |
| pass^8, tau-bench airline | 21% | 30% |
| Reviewer minutes per completed job | 11.4 | 6.2 |
| Median recovery after an injected fault (steps) | 9 | 3 |

## Licence

Apache-2.0.

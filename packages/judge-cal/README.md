# judge-cal

Part of [hana-evals](../../README.md). `eval-harness` uses the calibrated judge as the check on judged stages.

A model grading other models is cheap, and wrong in predictable ways: it prefers the answer it saw first, prefers the
longer answer, and prefers its own family's style. judge-cal measures those biases on your data, reduces them, and
calibrates the judge's scores against a few hundred human labels, so the number you report comes with an honest
interval.

## Install

```bash
# from the root of hana-evals
pip install ./packages/judge-cal
```

Any judge served behind an OpenAI-compatible endpoint works, including vLLM serving `hanaml/judge-7b`. For tests and
offline use, pass any callable as `backend=` instead of an endpoint.

## Quickstart

Run from `packages/judge-cal`, with your own human labels in `labels/`:

```python
from judge_cal import Judge, bias, calibrate, report

judge = Judge(endpoint="http://localhost:8000/v1", model="hanaml/judge-7b",
              rubric="rubrics/helpfulness.yaml", swap=True)

audit = bias.audit(judge, pairs="data/controlled-length-pairs.jsonl")
print(audit.position_flip_rate, audit.length_win_rate)

own = bias.audit(judge, pairs="labels/cross-family-pairs.jsonl", family="qwen")   # rows carry "families"
print(own.self_preference, own.self_pairs)

cal = calibrate.fit(judge, human_labels="labels/heldout-300.jsonl", method="isotonic")
scores = cal.score(judge, items="runs/candidate.jsonl")
print(report.summary(scores, human_agreement=cal.agreement))   # mean with 95% CI
```

Label rows are `{"question", "answer", "human": 0 | 1}` or carry `human_score` on the rubric scale; item rows are
`{"question", "answer"}`, with an optional `reference`. `notebooks/ci-with-judge-error.ipynb` shows how the
interval behaves as the label set shrinks.

## What it measures

- **Position bias.** Each pair is judged in both orders; a verdict that changes is a flip. With `swap=True` the two verdicts are combined and an inconsistent pair counts as a tie.
- **Length bias.** On pairs that humans rated equal but that differ in length, an unbiased judge picks the longer one 50% of the time.
- **Self-preference.** With `family=` set to the judge's model family, pairs that carry `families` with exactly one answer from it give the judge's win rate for its own family's answers minus the people's win rate on the same pairs; 0 is unbiased.
- **Agreement.** Pairwise agreement with the human majority, and Spearman correlation for 1–5 rubric scores.
- **Calibration.** Isotonic or Platt scaling on held-out human labels; expected calibration error before and after (cross-validated).
- **Reporting.** The final interval widens with the judge's measured disagreement with humans, rather than treating the judge as ground truth: the bootstrap resamples the human labels and refits the calibration, as well as resampling the items.

## Configuration

```yaml
# rubrics/helpfulness.yaml
scale: [1, 5]
criteria:
  - Answers the question that was asked
  - States facts that the sources support
  - Says when it does not know
reference_answer: optional
```

## Benchmarks

From the [PARC paper](https://hanaml.com/research/papers/calibrating-llm-judges), which calibrates its judge per
criterion type with the isotonic step judge-cal provides: Qwen3-32B as the judge, pooled over TheAgentCompany, DevAI
and BackOffice (2,340 runs graded by people), with the maps fitted on 160 graded runs per suite. The criterion judge
and the audit estimator are in the [paper repo](https://github.com/hanaml-lab/calibrating-llm-judges).

| Metric | Rubric in context | Criterion judge with retrieval |
| --- | --- | --- |
| Verdict accuracy against people, calibrated | 82.2% | 90.4% |
| Expected calibration error, raw | 0.267 | 0.174 |
| Expected calibration error, after calibration | 0.031 | 0.025 |
| Partial credit added by an appended success claim | 12.2 pts | 1.8 pts |
| Input tokens per run | 93k | 32k |

Offline, `examples/end_to_end.py` at the root of hana-evals calibrates its reference-based judge on 160 labels and
prints an ECE of 0.194 before calibration and 0.058 after; that toy result shows the mechanism, not a model's quality.

## Licence

Apache-2.0.

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

Qwen3-32B as the judge, 1,200 items with human scores.

| Metric | Single-prompt judge | Rubric + order swap + calibration |
| --- | --- | --- |
| Agreement with human majority (pairwise) | 66% | 79% |
| Position-flip rate | 18% | 4% |
| Length bias (50% is unbiased) | 71% | 54% |
| Spearman correlation with human scores | 0.52 | 0.71 |
| Expected calibration error | 0.14 | 0.05 |

## Licence

Apache-2.0.

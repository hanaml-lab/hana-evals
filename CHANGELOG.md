# Changelog

## 0.6.0 (2026-10-14)

- eval-harness 0.9.0: `dependency` placement ranks stages by the loss their own errors cause downstream, measured by
  replaying corrupted outputs through the job's tools; 0.0 undetected critical errors per 100 runs in the example,
  against 2.1 for the last stages and 4.2 for evenly spaced checks
- eval-harness 0.9.0: `perturbation.PerturbationTest`, verifier perturbation testing for staged jobs (VPT), and
  `agents.ReplayAgent`, a reward hack for testing detectors
- judge-cal 0.4.0: self-preference audit against human verdicts on cross-family pairs
- examples/perturbation_test.py; the end-to-end example compares three placements over five agent seeds
- README with results, papers and citation; CITATION.cff; CONTRIBUTING.md; CI runs both examples and validates the
  citation file

## 0.5.0 (2026-10-06)

- judged stages use a calibrated judge
- end-to-end example and integration test

## 0.4.0 (2025-10-22)

- judge-cal: intervals that include the judge's disagreement with people

## 0.3.0 (2025-10-13)

- judge-cal: rubric judge, order swap, length bias audit, isotonic and Platt calibration

## 0.2.0 (2025-02-12)

- dependency-aware check placement, fault injection, run folders with settings, Inspect suites

## 0.1.0 (2024-11-18)

- eval-harness: staged jobs, simulated and chat agents, pass^k

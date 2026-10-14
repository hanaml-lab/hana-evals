# Contributing to hana-evals

Thank you for helping. hana-evals is maintained by the hanaML lab; issues and pull requests are welcome.

## Set-up

```bash
git clone https://github.com/hanaml-lab/hana-evals && cd hana-evals
uv sync          # Python 3.10+; the three packages under packages/ as one uv workspace
```

## Before you open a pull request

```bash
make lint        # ruff, configured in pyproject.toml
make test        # each package's tests, then tests/integration
make example     # both offline examples must still run
```

- **Tests first.** A fix starts with a test that fails without it; a feature comes with tests through its public
  interface (`Runner`, `place`, `PerturbationTest`, `bias.audit`, `calibrate.fit`, `report.summary`), not its
  internals. Expected values come from a worked example or the specification, not from re-running the code.
- **Offline by default.** Tests and examples run without a network, a GPU or a model server. Code that needs one
  (Inspect suites, `ChatAgent`, the HTTP judge backend) is optional and skipped in CI.
- **Numbers.** If a change moves a number that the README reports under "Reproduced by this repo", update the README
  in the same pull request.
- **Changelog.** Add a line under a new heading at the top of `CHANGELOG.md`.

## Commit messages

Conventional and lowercase, with the package as the scope where there is one, as in the history:

```text
feat(eval-harness): verifier perturbation testing for staged jobs
fix(placement): rank stages by the loss their own errors cause downstream
test(judge-cal): swap, bias, agreement and calibration
docs: README with results, papers and citation
```

## Security

Do not open a public issue for a security problem. Follow the
[hanaML security policy](https://github.com/hanaml-lab/.github/blob/main/SECURITY.md).

## Licence

hana-evals is licensed under Apache-2.0. By contributing you agree that your contribution is licensed under the same
terms (inbound = outbound), as section 5 of the licence provides.

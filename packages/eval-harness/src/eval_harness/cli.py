"""eval-harness run SUITE --agent CONFIG [--repeats N] [--out DIR]
eval-harness report RUN_DIR [--metric pass^8] [--metric undetected-critical]"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from eval_harness import metrics
from eval_harness.faults import Fault
from eval_harness.runner import Runner
from eval_harness.stages import Suite
from eval_harness.store import load


def report(run_dir: str, wanted: list[str]) -> dict:
    settings, _stages, runs = load(run_dir)
    out: dict = {"run": run_dir, "suite": settings["suite"], "runs": len(runs)}
    for m in wanted or ["pass^1", "undetected-critical"]:
        if m.startswith("pass^"):
            out[m] = round(metrics.pass_k(runs, int(m[5:])), 4)
        elif m == "undetected-critical":
            out["undetected_critical_per_100"] = round(metrics.undetected_critical_per_100(runs), 2)
        elif m == "checks":
            out["checks_per_completed_job"] = round(metrics.reviewer_load(runs), 2)
        elif m == "recovery":
            out["median_recovery_steps"] = metrics.median_recovery(runs)
        else:
            raise SystemExit(f"unknown metric {m}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="eval-harness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("suite")
    run.add_argument("--agent", required=True)
    run.add_argument("--repeats", type=int, default=8)
    run.add_argument("--out", default="runs/")
    run.add_argument("--fault", action="append", default=[], help="inject a wrong result at this stage")
    rep = sub.add_parser("report")
    rep.add_argument("run_dir")
    rep.add_argument("--metric", action="append", default=[])
    args = ap.parse_args(argv)

    if args.cmd == "run":
        cfg = yaml.safe_load(open(args.agent, encoding="utf-8"))
        runner = Runner(agent=cfg, repeats=args.repeats, review_budget=cfg.get("review_budget", 0.2),
                        placement=cfg.get("placement", "dependency"), out=args.out,
                        faults=[Fault(s) for s in args.fault])
        result = runner.run(Suite.load(args.suite))
        print(result.path)
        return 0
    print(json.dumps(report(args.run_dir, args.metric), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

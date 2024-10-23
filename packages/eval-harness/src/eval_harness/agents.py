"""Agents that produce a stage's output from its inputs.

``SimulatedAgent`` calls the stage's tool and then, with a per-stage error rate, corrupts the
result the way agents do (an off-by-a-bit figure, a dropped row, a truncated draft). It needs no
model and is seeded, which makes placement and fault experiments exactly repeatable.
``ChatAgent`` asks a model behind an OpenAI-compatible endpoint for the stage output as JSON.
"""

from __future__ import annotations

import copy
import json
import os
import random
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import yaml

from eval_harness.stages import Job, Stage


class Agent(Protocol):
    def act(self, job: Job, stage: Stage, upstream: dict[str, Any], attempt: int) -> Any: ...


def corrupt(value: Any, rng: random.Random) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return round(value * rng.choice([0.9, 1.1]) + rng.choice([-1, 1]), 2)
    if isinstance(value, str):
        if value in {"approve", "hold"}:
            return "hold" if value == "approve" else "approve"
        return value[: max(1, len(value) // 2)]
    if isinstance(value, list):
        if not value:
            return ["UNKNOWN"]
        out = copy.deepcopy(value)
        out.pop(rng.randrange(len(out)))
        return out
    return None


class SimulatedAgent:
    def __init__(self, error_rate: float = 0.08, retry_error_rate: float | None = None, seed: int = 0):
        self.error_rate = error_rate
        self.retry_error_rate = error_rate / 2 if retry_error_rate is None else retry_error_rate
        self.seed = seed

    def act(self, job: Job, stage: Stage, upstream: dict[str, Any], attempt: int) -> Any:
        rng = random.Random(f"{self.seed}:{job.id}:{stage.id}:{attempt}")
        out = job.run_tool(stage, upstream)
        rate = self.error_rate if attempt == 0 else self.retry_error_rate
        return corrupt(out, rng) if rng.random() < rate else out


class ChatAgent:
    def __init__(self, model: str, endpoint: str | None = None, max_steps: int = 80):
        self.model, self.max_steps = model, max_steps
        self.endpoint = (endpoint or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")).rstrip("/")

    def act(self, job: Job, stage: Stage, upstream: dict[str, Any], attempt: int) -> Any:
        prompt = (f"Task stage: {stage.id}\n{stage.instructions}\n\nJob inputs:\n{json.dumps(job.inputs)}\n\n"
                  f"Results of earlier stages:\n{json.dumps({d: upstream[d] for d in stage.deps})}\n\n"
                  'Reply with JSON {"output": ...}.')
        body = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(f"{self.endpoint}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(json.load(resp)["choices"][0]["message"]["content"]).get("output")


def load_agent(config: str | Path | dict | Agent) -> tuple[Agent, dict]:
    if not isinstance(config, str | Path | dict):
        return config, {"agent": type(config).__name__}
    cfg = config if isinstance(config, dict) else yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    model = cfg.get("model", "simulated")
    if model == "simulated":
        return SimulatedAgent(cfg.get("error_rate", 0.08), cfg.get("retry_error_rate"), cfg.get("seed", 0)), cfg
    return ChatAgent(model, cfg.get("endpoint"), cfg.get("max_steps", 80)), cfg

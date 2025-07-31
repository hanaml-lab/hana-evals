"""A judge served behind an OpenAI-compatible endpoint, or any callable with the same contract."""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from judge_cal.swap import combine

Verdict = Literal["A", "B", "tie"]


@dataclass(frozen=True)
class Rubric:
    scale: tuple[int, int] = (1, 5)
    criteria: list[str] = field(default_factory=list)
    reference_answer: str = "none"  # none | optional | required

    @classmethod
    def load(cls, path: str | Path) -> Rubric:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        lo, hi = raw.get("scale", [1, 5])
        if lo >= hi:
            raise ValueError("scale must be [low, high] with low < high")
        return cls((lo, hi), list(raw.get("criteria", [])), raw.get("reference_answer", "none"))

    def text(self) -> str:
        lines = "\n".join(f"- {c}" for c in self.criteria)
        return f"Score from {self.scale[0]} to {self.scale[1]} against these criteria:\n{lines}"


Backend = Callable[[str], str]


def http_backend(endpoint: str, model: str, api_key: str = "EMPTY", timeout: float = 120) -> Backend:
    url = endpoint.rstrip("/") + "/chat/completions"

    def call(prompt: str) -> str:
        body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]

    return call


_PAIR = re.compile(r"\[\[(A|B|tie)\]\]", re.IGNORECASE)
_SCORE = re.compile(r"\[\[(\d+)\]\]")


class Judge:
    """Pairwise verdicts and rubric scores.

    With ``swap=True`` every pair is judged in both orders and the two verdicts are combined
    (see ``judge_cal.swap``); an inconsistent pair becomes a tie.
    """

    def __init__(self, endpoint: str | None = None, model: str | None = None,
                 rubric: str | Path | Rubric | None = None, swap: bool = True,
                 backend: Backend | None = None):
        if backend is None:
            if not (endpoint and model):
                raise ValueError("give endpoint and model, or a backend")
            backend = http_backend(endpoint, model)
        self.backend = backend
        self.model = model or "custom"
        self.rubric = rubric if isinstance(rubric, Rubric) else Rubric.load(rubric) if rubric else Rubric()
        self.swap = swap

    def _pair_prompt(self, question: str, first: str, second: str) -> str:
        return (f"{self.rubric.text()}\n\nQuestion:\n{question}\n\n[Answer A]\n{first}\n\n[Answer B]\n{second}\n\n"
                "Which answer is better? End with [[A]], [[B]] or [[tie]].")

    def compare_once(self, question: str, first: str, second: str) -> Verdict:
        found = _PAIR.findall(self.backend(self._pair_prompt(question, first, second)))
        if not found:
            return "tie"
        v = found[-1]
        return "tie" if v.lower() == "tie" else v.upper()  # type: ignore[return-value]

    def compare(self, question: str, a: str, b: str) -> Verdict:
        forward = self.compare_once(question, a, b)
        if not self.swap:
            return forward
        backward = self.compare_once(question, b, a)
        return combine(forward, backward).verdict

    def score(self, question: str, answer: str, reference: str | None = None) -> int:
        ref = f"\n\nReference answer:\n{reference}" if reference else ""
        prompt = (f"{self.rubric.text()}\n\nQuestion:\n{question}{ref}\n\nAnswer:\n{answer}\n\n"
                  f"End with the score in double brackets, e.g. [[{self.rubric.scale[1]}]].")
        found = _SCORE.findall(self.backend(prompt))
        lo, hi = self.rubric.scale
        if not found:
            raise ValueError("judge reply has no [[score]]")
        return min(max(int(found[-1]), lo), hi)

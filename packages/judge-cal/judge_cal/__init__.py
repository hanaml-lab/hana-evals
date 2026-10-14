"""Measure, reduce and calibrate the errors of a model-as-judge."""

from judge_cal import agreement, bias, calibrate, report, swap
from judge_cal.judge import Judge, Rubric

__all__ = ["Judge", "Rubric", "agreement", "bias", "calibrate", "report", "swap"]
__version__ = "0.4.0"

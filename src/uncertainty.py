"""Uncertainty helpers for rates shown in the dashboard and paper."""
from __future__ import annotations

import math

SMALL_SAMPLE_N = 100  # below this, rates are flagged as "small sample"


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (better than the normal
    approximation for small n or rates near 0% / 100%)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_summary(flags) -> dict:
    """Rate, 95% interval, n and small-sample flag for a 0/1 series."""
    n = int(len(flags))
    k = int(sum(flags))
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"),
            "ci_low": lo, "ci_high": hi, "small_sample": n < SMALL_SAMPLE_N}


def icon_counts(rate: float, total: int = 100) -> int:
    """How many icons out of `total` to fill for a rate (natural frequency)."""
    return int(round(rate * total))

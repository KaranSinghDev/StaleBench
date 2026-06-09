"""Metric aggregation: recovery rate with a Wilson 95% confidence interval + mean catch-up.

A per-fact result is a catch-up latency (ticks from the change to a stable-correct answer), or
+inf if the answer never caught up within the horizon. `summarize` turns a list of these into
a reported recovery rate (fraction that caught up) with a CI, plus the mean catch-up of those
that did. Reporting an interval — not a point — is how the benchmark stays honest about the
non-determinism of LLM answers across trials.
"""
from __future__ import annotations
import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(latencies) -> dict:
    vals = list(latencies)
    n = len(vals)
    recovered = [v for v in vals if math.isfinite(v)]
    lo, hi = wilson(len(recovered), n)
    mean = sum(recovered) / len(recovered) if recovered else math.inf
    return {
        "n": n,
        "recovered": len(recovered),
        "rate": (len(recovered) / n if n else 0.0),
        "rate_ci": [round(lo, 3), round(hi, 3)],
        "mean_catchup": (round(mean, 3) if recovered else math.inf),
    }

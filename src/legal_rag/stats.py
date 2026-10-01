"""Bootstrap intervals over per-question metrics."""

from __future__ import annotations

import random


def bootstrap_interval(values: list[float], samples: int = 1000, seed: int = 42) -> tuple[float, float]:
    if not values:
        raise ValueError("values cannot be empty")
    randomizer = random.Random(seed)
    means = sorted(sum(randomizer.choices(values, k=len(values))) / len(values) for _ in range(samples))
    return means[int(samples * 0.025)], means[int(samples * 0.975)]

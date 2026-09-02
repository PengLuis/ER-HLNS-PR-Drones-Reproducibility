"""Canonical deterministic paired seed-block percentile bootstrap.

The reported controls and the public reproduction script use the same
resampling order, seed derivation, and percentile-index convention.  The
optional namespace keeps the E1 labels distinct while using the same canonical
full-width seed derivation.
"""

from __future__ import annotations

import hashlib
import random
from statistics import mean
from typing import Iterable


BOOTSTRAP_RESAMPLES = 10_000


def stable_seed(label: str, *, namespace: str | None = None) -> int:
    """Return the stable 64-bit seed used by the reported analysis."""
    text = f"{namespace}|{label}" if namespace else label
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def percentile_interval(
    values: Iterable[float],
    label: str,
    *,
    replicates: int = BOOTSTRAP_RESAMPLES,
    namespace: str | None = None,
) -> tuple[float, float]:
    """Compute the reported marginal percentile interval.

    The implementation intentionally mirrors the generator used for the
    E4--E7 Source Data: Python ``random.Random`` with deterministic draws,
    sorted bootstrap means, and the retained order-statistic indices 249 and
    9750 for 10,000 resamples.  ``namespace='bootstrap'`` is used for E1
    labels and does not change the canonical full 64-bit seed derivation.
    """
    vector = [float(value) for value in values]
    if not vector:
        return float("nan"), float("nan")
    rng = random.Random(stable_seed(label, namespace=namespace))
    n = len(vector)
    samples = sorted(
        mean(vector[rng.randrange(n)] for _ in range(n))
        for _ in range(int(replicates))
    )
    low_index = max(0, int(0.025 * len(samples)) - 1)
    high_index = min(len(samples) - 1, int(0.975 * len(samples)))
    return float(samples[low_index]), float(samples[high_index])

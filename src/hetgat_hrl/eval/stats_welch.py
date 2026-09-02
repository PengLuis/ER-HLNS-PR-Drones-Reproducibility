from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_t_test(x: Iterable[float], y: Iterable[float]) -> Dict[str, float]:
    a = np.array(list(x), dtype=np.float64)
    b = np.array(list(y), dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return {"t": 0.0, "dof": 0.0, "pvalue_two_sided": 1.0}
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    na, nb = float(a.size), float(b.size)
    den = math.sqrt(max(va / na + vb / nb, 1e-12))
    t = (ma - mb) / den
    dof_num = (va / na + vb / nb) ** 2
    dof_den = (va * va) / (na * na * max(na - 1.0, 1.0)) + (vb * vb) / (
        nb * nb * max(nb - 1.0, 1.0)
    )
    dof = dof_num / max(dof_den, 1e-12)
    # Two-sided p-value (normal approximation; scipy not required).
    p = 2.0 * (1.0 - _normal_cdf(abs(float(t))))
    return {"t": float(t), "dof": float(dof), "pvalue_two_sided": float(np.clip(p, 0.0, 1.0))}


def mean_ci95(samples: Iterable[float]) -> Dict[str, float]:
    x = np.array(list(samples), dtype=np.float64)
    if x.size == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    m = float(np.mean(x))
    if x.size == 1:
        return {"mean": m, "ci95_low": m, "ci95_high": m}
    s = float(np.std(x, ddof=1))
    # z=1.96 approximation to keep dependency-free.
    half = 1.96 * s / math.sqrt(float(x.size))
    return {"mean": m, "ci95_low": float(m - half), "ci95_high": float(m + half)}


def hedges_g(x: Iterable[float], y: Iterable[float]) -> float:
    """
    Bias-corrected standardized mean difference.
    Positive value means x > y on average.
    """
    a = np.array(list(x), dtype=np.float64)
    b = np.array(list(y), dtype=np.float64)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return 0.0
    s1 = float(np.var(a, ddof=1))
    s2 = float(np.var(b, ddof=1))
    pooled_den = (n1 - 1) * s1 + (n2 - 1) * s2
    pooled = math.sqrt(max(pooled_den / max(n1 + n2 - 2, 1), 1e-12))
    d = (float(np.mean(a)) - float(np.mean(b))) / pooled
    # Small-sample correction.
    j = 1.0 - 3.0 / max(4.0 * (n1 + n2) - 9.0, 1.0)
    return float(d * j)


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR correction.
    Returns adjusted p-values in original order.
    """
    if len(p_values) == 0:
        return []
    arr = np.asarray(p_values, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if not np.any(finite_mask):
        return [float(v) for v in out]
    finite_idx = np.where(finite_mask)[0]
    finite_vals = arr[finite_mask]
    order = np.argsort(finite_vals)
    sorted_vals = finite_vals[order]
    m = float(sorted_vals.size)
    adj_sorted = np.empty_like(sorted_vals)
    prev = 1.0
    # Monotone step-up adjusted p-values.
    for i in range(sorted_vals.size - 1, -1, -1):
        rank = float(i + 1)
        val = float(sorted_vals[i] * m / rank)
        prev = min(prev, val)
        adj_sorted[i] = prev
    adj = np.empty_like(finite_vals)
    adj[order] = np.clip(adj_sorted, 0.0, 1.0)
    out[finite_idx] = adj
    return [float(v) for v in out]


def significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


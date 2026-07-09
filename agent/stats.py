"""Small, dependency-free stats for paired eval comparisons."""
from __future__ import annotations
import math

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z*z/n
    centre = (p + z*z/(2*n)) / d
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (max(0.0, centre - half), min(1.0, centre + half))

def _binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    from math import comb
    probs = [comb(n, i) for i in range(n+1)]
    total = 2**n
    kp = comb(n, k)
    p = sum(c for c in probs if c <= kp) / total
    return min(1.0, p)

def mcnemar(b: int, c: int) -> float:
    """Exact McNemar (binomial on discordant pairs). b, c = the two discordant counts."""
    n = b + c
    return _binom_two_sided(min(b, c), n)

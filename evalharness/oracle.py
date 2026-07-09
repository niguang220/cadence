"""Result oracle: decide whether a predicted result set matches the gold one.

Execution match is the *primary* signal for NL->SQL correctness, not a
deterministic ground truth. Two correct queries can return the same rows in a
different order, the same numbers at a different float precision, or the same
groups built a different way. This module canonicalizes both sides before
comparing so those differences don't read as failures:

- row order is ignored unless the question is inherently ordered (top-N, "first"),
- floats are rounded to a fixed precision (money/averages),
- NULLs compare equal to NULLs.

Structural checks and adversarial fixtures (Phase 3) harden this further; here we
keep the baseline comparison small and explainable.
"""
from __future__ import annotations

import math

Row = tuple
Rows = list


def _decimals(float_tolerance: float) -> int:
    """Number of decimal places implied by a tolerance, e.g. 0.01 -> 2."""
    if float_tolerance <= 0:
        return 6
    return max(0, -int(round(math.log10(float_tolerance))))


def _canon_value(value, decimals: int):
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass; keep it distinct
        return value
    if isinstance(value, float):
        return round(value, decimals)
    if isinstance(value, int):
        return float(round(value))  # 5 and 5.0 should compare equal
    return value


def _sort_key(row: Row):
    # None sorts first, then by type name + string form so mixed types are stable.
    return tuple((v is None, type(v).__name__, str(v)) for v in row)


def canonicalize_result(rows: Rows, *, ordered: bool = False,
                        float_tolerance: float = 0.01) -> Rows:
    """Normalize a result set so equivalent results compare equal."""
    decimals = _decimals(float_tolerance)
    canon = [tuple(_canon_value(v, decimals) for v in row) for row in rows]
    if not ordered:
        canon = sorted(canon, key=_sort_key)
    return canon


def execution_match(predicted: Rows, gold: Rows, *, ordered: bool = False,
                    float_tolerance: float = 0.01) -> bool:
    """True if the predicted rows match the gold rows after canonicalization."""
    return (
        canonicalize_result(predicted, ordered=ordered, float_tolerance=float_tolerance)
        == canonicalize_result(gold, ordered=ordered, float_tolerance=float_tolerance)
    )

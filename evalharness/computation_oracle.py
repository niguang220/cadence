"""Compare a sandbox computation result against a gold value.

The sandbox emits arbitrary JSON (dict / list / scalar), so the row-oriented
``oracle.execution_match`` cannot judge it. This recursive comparator locks the
computation-correctness semantics: numeric tolerance, bool distinct from int, dict
key-set equality, ordered lists, and nesting. Charts are NOT a supported comparison
target (base64 PNG bytes are not a stable oracle) -- a "chart" key raises rather than
silently passing.
"""
from __future__ import annotations

import numbers


def computation_match(predicted, expected, *, tolerance: float = 1e-6) -> bool:
    if isinstance(predicted, dict) and isinstance(expected, dict):
        if "chart" in predicted or "chart" in expected:
            raise ValueError("charts are not a supported comparison target")
        if set(predicted) != set(expected):
            return False
        return all(computation_match(predicted[k], expected[k], tolerance=tolerance)
                   for k in expected)
    if isinstance(predicted, list) and isinstance(expected, list):
        return (len(predicted) == len(expected)
                and all(computation_match(p, e, tolerance=tolerance)
                        for p, e in zip(predicted, expected)))
    # bool must not compare equal to int (True == 1 in Python) -- check it first.
    if isinstance(predicted, bool) or isinstance(expected, bool):
        return predicted is expected
    if isinstance(predicted, numbers.Number) and isinstance(expected, numbers.Number):
        return abs(predicted - expected) <= tolerance
    if type(predicted) is not type(expected):
        return False
    return predicted == expected

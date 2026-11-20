"""Reference values used as regression baselines.

* :data:`F1_REPORTED` — Table 1 of the F1 report (``mo438_f1.pdf``), *as printed*
  (rounded). Each entry stores the printed value and how it was rounded.
* :data:`AUDIT_EDGE_CLASSES` — edge-class counts obtained in the technical audit of
  2026-11-06 (independent numpy/scipy script, before any analysis was run).

Full-precision F1 values are read from ``outputs/obj1_characterization/summary.json`` at run time.
"""

from __future__ import annotations

import math
from typing import Any

# metric -> (printed value, rounding): ("exact", None) | ("decimals", d) | ("sig", s)
F1_REPORTED: dict[str, tuple[float | int, str, int | None]] = {
    "num_nodes": (169_343, "exact", None),
    "num_edges": (1_157_799, "exact", None),
    "mean_degree": (13.6740, "decimals", 4),
    "average_local_clustering": (0.257454, "decimals", 6),   # mean of C_i over k_i >= 2
    "transitivity": (0.016201, "decimals", 6),
    "average_distance": (5.750052, "decimals", 6),
    "diameter": (25, "exact", None),
    "density": (8.07479e-5, "sig", 6),
    "num_connected_components": (1, "exact", None),
    "num_nodes_degree_lt_2": (20_604, "exact", None),        # F1 report, section 3.2
}

AUDIT_EDGE_CLASSES: dict[str, dict[str, int]] = {
    "G_d": {"intra": 763_986, "inter": 402_257, "total": 1_166_243},
    "G_u": {"intra": 757_455, "inter": 400_344, "total": 1_157_799},
    "reciprocal_pairs": {"intra": 6_531, "inter": 1_913, "total": 8_444},
}


def round_like_report(value: float, rounding: str, digits: int | None) -> float:
    """Round ``value`` the way the F1 report printed it."""
    if rounding == "exact":
        return value
    if rounding == "decimals":
        return round(value, digits)
    if rounding == "sig":
        if value == 0:
            return 0.0
        return round(value, digits - 1 - math.floor(math.log10(abs(value))))
    raise ValueError(rounding)


def matches_report(metric: str, value: Any) -> bool:
    """True if ``value`` rounds to the number printed in the F1 report."""
    printed, rounding, digits = F1_REPORTED[metric]
    if rounding == "exact":
        return int(value) == int(printed)
    return math.isclose(round_like_report(float(value), rounding, digits), float(printed),
                        rel_tol=0, abs_tol=1e-15 if rounding == "sig" else 0.5 * 10 ** -digits / 10)

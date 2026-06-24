from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .loaders import SimCase


def align_simulation_diameter(
    sim: pd.DataFrame,
    exp: pd.DataFrame,
    case: SimCase,
    config: dict[str, Any],
) -> pd.DataFrame:
    method = config.get("alignment", {}).get(
        "method", "sim_min_diameter_to_experimental_min_diameter"
    )
    if method != "sim_min_diameter_to_experimental_min_diameter":
        raise ValueError(f"Unsupported alignment method: {method}")

    out = sim.copy()
    exp_min_diameter = float(exp["diameter"].min())
    max_compression_disp = float(out["disp"].max())
    start = float(exp_min_diameter + 2.0 * max_compression_disp)
    out["diameter"] = start - 2.0 * out["disp"]
    out.attrs["diameter_alignment"] = method
    out.attrs["exp_min_diameter"] = exp_min_diameter
    out.attrs["max_compression_disp"] = max_compression_disp
    out.attrs["starting_diameter_used"] = start
    return out

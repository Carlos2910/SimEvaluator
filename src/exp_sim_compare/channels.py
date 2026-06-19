from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def add_derived_channels(sim: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = sim.copy()
    channels = config.get("channels", {})
    for name, channel_cfg in channels.items():
        if channel_cfg.get("formula") == "vector_magnitude":
            cols = channel_cfg.get("columns", ["RF1", "RF2", "RF3"])
            missing = set(cols) - set(out.columns)
            if missing:
                raise ValueError(f"Missing vector magnitude columns for {name}: {sorted(missing)}")
            total = np.zeros(len(out), dtype=float)
            for col in cols:
                total += out[col].to_numpy(dtype=float) ** 2
            out[name] = np.sqrt(total)
    return out


def channel_names(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(config.get("channels", {}).keys())


def comparison_channel(sim: pd.DataFrame, channel: str, config: dict[str, Any]) -> pd.Series:
    values = sim[channel].copy()
    channel_cfg = config.get("channels", {}).get(channel, {})
    if channel_cfg.get("transform") == "abs":
        values = values.abs()
    return values

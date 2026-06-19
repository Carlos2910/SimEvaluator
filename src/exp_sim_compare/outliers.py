from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    return (
        series.rolling(window=window, center=True, min_periods=max(3, window // 3))
        .median()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )


def hampel_outlier_mask(
    values: Iterable[float],
    *,
    window: int = 41,
    sigma: float = 6.0,
    min_relative_prominence: float = 0.03,
) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if len(y) < 5:
        return np.zeros(len(y), dtype=bool)

    if window % 2 == 0:
        window += 1
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    window = max(window, 5)

    med = rolling_median(y, window)
    residual = np.abs(y - med)
    mad = rolling_median(residual, window)
    robust_sigma = 1.4826 * mad
    dynamic_threshold = sigma * np.maximum(robust_sigma, 1e-12)
    absolute_threshold = min_relative_prominence * max(np.nanmax(y) - np.nanmin(y), 1e-12)
    return residual > np.maximum(dynamic_threshold, absolute_threshold)


def add_outlier_masks(
    sim: pd.DataFrame,
    channels: tuple[str, ...],
    *,
    window: int = 41,
    sigma: float = 6.0,
    min_relative_prominence: float = 0.03,
) -> pd.DataFrame:
    out = sim.copy()
    for channel in channels:
        out[f"{channel}_outlier"] = hampel_outlier_mask(
            out[channel],
            window=window,
            sigma=sigma,
            min_relative_prominence=min_relative_prominence,
        )
    out["any_outlier"] = out[[f"{channel}_outlier" for channel in channels]].any(axis=1)
    return out

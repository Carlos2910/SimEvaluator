from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HampelResult:
    flagged: np.ndarray
    excluded: np.ndarray
    residual: np.ndarray
    threshold: np.ndarray
    ratio: np.ndarray
    window: int
    absolute_threshold: float


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    return (
        series.rolling(window=window, center=True, min_periods=max(3, window // 3))
        .median()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )


def normalize_window(window: int, length: int) -> int:
    if window % 2 == 0:
        window += 1
    window = min(window, length if length % 2 == 1 else length - 1)
    return max(window, 5)


def window_from_diameter(
    diameter: Iterable[float] | None,
    *,
    fallback_window: int,
    window_diameter_span: float | None,
    min_window_points: int = 5,
) -> int:
    if diameter is None or window_diameter_span is None:
        return fallback_window

    x = np.asarray(diameter, dtype=float)
    if len(x) < 2:
        return fallback_window
    step = float(np.nanmedian(np.abs(np.diff(x))))
    if not np.isfinite(step) or step <= 0:
        return fallback_window
    return max(int(round(window_diameter_span / step)), min_window_points)


def hampel_outlier_details(
    values: Iterable[float],
    *,
    diameter: Iterable[float] | None = None,
    window: int = 41,
    window_diameter_span: float | None = None,
    min_window_points: int = 5,
    sigma: float = 6.0,
    min_relative_prominence: float = 0.03,
    exclusion_threshold_ratio: float = 3.0,
) -> HampelResult:
    y = np.asarray(values, dtype=float)
    if len(y) < 5:
        empty = np.zeros(len(y), dtype=bool)
        values_out = np.full(len(y), np.nan)
        return HampelResult(empty, empty, values_out, values_out, values_out, 0, np.nan)

    effective_window = window_from_diameter(
        diameter,
        fallback_window=window,
        window_diameter_span=window_diameter_span,
        min_window_points=min_window_points,
    )
    effective_window = normalize_window(effective_window, len(y))

    med = rolling_median(y, effective_window)
    residual = np.abs(y - med)
    mad = rolling_median(residual, effective_window)
    robust_sigma = 1.4826 * mad
    dynamic_threshold = sigma * np.maximum(robust_sigma, 1e-12)
    absolute_threshold = min_relative_prominence * max(np.nanmax(y) - np.nanmin(y), 1e-12)
    threshold = np.maximum(dynamic_threshold, absolute_threshold)
    ratio = residual / np.maximum(threshold, 1e-12)
    flagged = residual > threshold
    excluded = flagged & (ratio >= exclusion_threshold_ratio)
    return HampelResult(
        flagged=flagged,
        excluded=excluded,
        residual=residual,
        threshold=threshold,
        ratio=ratio,
        window=effective_window,
        absolute_threshold=float(absolute_threshold),
    )


def hampel_outlier_mask(
    values: Iterable[float],
    *,
    window: int = 41,
    sigma: float = 6.0,
    min_relative_prominence: float = 0.03,
) -> np.ndarray:
    return hampel_outlier_details(
        values,
        window=window,
        sigma=sigma,
        min_relative_prominence=min_relative_prominence,
        exclusion_threshold_ratio=1.0,
    ).flagged


def branch_index_arrays(length: int, diameter: Iterable[float] | None, split_by_branch: bool) -> list[np.ndarray]:
    if not split_by_branch or diameter is None or length < 5:
        return [np.arange(length)]
    x = np.asarray(diameter, dtype=float)
    idx_min = int(np.nanargmin(x))
    return [np.arange(0, idx_min + 1), np.arange(idx_min, length)]


def add_outlier_masks(
    sim: pd.DataFrame,
    channels: tuple[str, ...],
    *,
    window: int = 41,
    window_diameter_span: float | None = None,
    min_window_points: int = 5,
    sigma: float = 6.0,
    min_relative_prominence: float = 0.03,
    exclusion_threshold_ratio: float = 3.0,
    split_by_branch: bool = True,
    channel_values: dict[str, Iterable[float]] | None = None,
) -> pd.DataFrame:
    out = sim.copy()
    for channel in channels:
        values = np.asarray(
            channel_values[channel] if channel_values and channel in channel_values else out[channel],
            dtype=float,
        )
        flagged = np.zeros(len(out), dtype=bool)
        excluded = np.zeros(len(out), dtype=bool)
        residual = np.full(len(out), np.nan)
        threshold = np.full(len(out), np.nan)
        ratio = np.full(len(out), np.nan)

        diameter = out["diameter"].to_numpy(dtype=float) if "diameter" in out.columns else None
        for idx in branch_index_arrays(len(out), diameter, split_by_branch):
            result = hampel_outlier_details(
                values[idx],
                diameter=diameter[idx] if diameter is not None else None,
                window=window,
                window_diameter_span=window_diameter_span,
                min_window_points=min_window_points,
                sigma=sigma,
                min_relative_prominence=min_relative_prominence,
                exclusion_threshold_ratio=exclusion_threshold_ratio,
            )
            flagged[idx] |= result.flagged
            excluded[idx] |= result.excluded
            residual[idx] = np.nanmax(np.vstack([residual[idx], result.residual]), axis=0)
            threshold[idx] = np.nanmin(np.vstack([threshold[idx], result.threshold]), axis=0)
            ratio[idx] = np.nanmax(np.vstack([ratio[idx], result.ratio]), axis=0)

        out[f"{channel}_outlier"] = flagged
        out[f"{channel}_exclude"] = excluded
        out[f"{channel}_outlier_residual"] = residual
        out[f"{channel}_outlier_threshold"] = threshold
        out[f"{channel}_outlier_ratio"] = ratio
    out["any_outlier"] = out[[f"{channel}_outlier" for channel in channels]].any(axis=1)
    out["any_excluded"] = out[[f"{channel}_exclude" for channel in channels]].any(axis=1)
    return out

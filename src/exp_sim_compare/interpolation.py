from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .loaders import SimCase


CURVE_COLUMNS = [
    "case",
    "sample",
    "condition",
    "node",
    "file",
    "channel",
    "branch",
    "diameter",
    "experimental_force",
    "simulation_force_raw_interpolated",
    "simulation_force_outliers_excluded_interpolated",
    "residual",
    "abs_residual",
]


def metrics_comparison_grid(config: dict[str, Any]) -> str:
    metrics_cfg = config.get("metrics", {})
    grid = metrics_cfg.get("comparison_grid", "auto")
    if grid not in {"auto", "experimental", "simulation_native"}:
        raise ValueError("metrics.comparison_grid must be auto, experimental, or simulation_native")
    return grid


def interpolation_filter_window(config: dict[str, Any]) -> int:
    interpolation_cfg = config.get("interpolation", {})
    window = interpolation_cfg.get("filter_window", interpolation_cfg.get("window", 7))
    return int(window)


def prepare_interp_source(sim_branch: pd.DataFrame, channel: str) -> pd.DataFrame:
    return (
        sim_branch.loc[:, ["diameter", channel]]
        .dropna()
        .sort_values("diameter")
        .groupby("diameter", as_index=False)[channel]
        .mean()
    )


def median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    if len(series) < 3 or window <= 1:
        return series.to_numpy(dtype=float)

    if window % 2 == 0:
        window += 1
    window = min(window, len(series) if len(series) % 2 == 1 else len(series) - 1)
    window = max(window, 3)

    return (
        series.rolling(window=window, center=True, min_periods=max(3, window // 3))
        .median()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )


def prepare_filtered_interp_source(
    sim_branch: pd.DataFrame,
    channel: str,
    *,
    filter_window: int,
) -> pd.DataFrame:
    source = prepare_interp_source(sim_branch, channel)
    if source.empty:
        source["filtered"] = pd.Series(dtype=float)
        return source
    source["filtered"] = median_smooth(source[channel].to_numpy(dtype=float), filter_window)
    return source


def interp_values_at(x: np.ndarray, source: pd.DataFrame, value_column: str) -> np.ndarray:
    if len(source) < 2:
        return np.full(len(x), np.nan)
    return np.interp(
        x,
        source["diameter"].to_numpy(dtype=float),
        source[value_column].to_numpy(dtype=float),
        left=np.nan,
        right=np.nan,
    )


def interp_experimental_at(x: np.ndarray, exp_branch: pd.DataFrame) -> np.ndarray:
    source = (
        exp_branch.loc[:, ["diameter", "force"]]
        .dropna()
        .sort_values("diameter")
        .groupby("diameter", as_index=False)["force"]
        .mean()
    )
    return interp_values_at(x, source.rename(columns={"force": "_force"}), "_force")


def interp_channel_at(x: np.ndarray, sim_branch: pd.DataFrame, channel: str) -> np.ndarray:
    source = prepare_interp_source(sim_branch, channel)
    return interp_values_at(x, source, channel)


def interp_sim_to_test(
    exp_branch: pd.DataFrame,
    sim_branch: pd.DataFrame,
    channel: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin = max(exp_branch["diameter"].min(), sim_branch["diameter"].min())
    xmax = min(exp_branch["diameter"].max(), sim_branch["diameter"].max())

    exp_overlap = exp_branch[
        (exp_branch["diameter"] >= xmin) & (exp_branch["diameter"] <= xmax)
    ].copy()
    sim_overlap = sim_branch[
        (sim_branch["diameter"] >= xmin) & (sim_branch["diameter"] <= xmax)
    ].copy()

    if len(exp_overlap) < 5 or len(sim_overlap) < 5:
        return np.array([]), np.array([]), np.array([])

    x = exp_overlap["diameter"].to_numpy(dtype=float)
    y_exp = exp_overlap["force"].to_numpy(dtype=float)
    y_sim = interp_channel_at(x, sim_overlap, channel)
    valid = ~np.isnan(y_sim)
    return x[valid], y_exp[valid], y_sim[valid]


def interp_test_to_sim(
    exp_branch: pd.DataFrame,
    sim_branch: pd.DataFrame,
    channel: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim_source = prepare_interp_source(sim_branch, channel)
    if len(exp_branch) < 5 or len(sim_source) < 5:
        return np.array([]), np.array([]), np.array([])

    xmin = max(exp_branch["diameter"].min(), sim_source["diameter"].min())
    xmax = min(exp_branch["diameter"].max(), sim_source["diameter"].max())
    sim_overlap = sim_source[
        (sim_source["diameter"] >= xmin) & (sim_source["diameter"] <= xmax)
    ].copy()

    if len(sim_overlap) < 5:
        return np.array([]), np.array([]), np.array([])

    x = sim_overlap["diameter"].to_numpy(dtype=float)
    y_exp = interp_experimental_at(x, exp_branch)
    y_sim = sim_overlap[channel].to_numpy(dtype=float)
    valid = ~(np.isnan(y_exp) | np.isnan(y_sim))
    return x[valid], y_exp[valid], y_sim[valid]


def choose_metric_grid(
    exp_branch: pd.DataFrame,
    sim_branch: pd.DataFrame,
    channel: str,
    config: dict[str, Any],
) -> str:
    grid = metrics_comparison_grid(config)
    if grid != "auto":
        return grid
    sim_source = prepare_interp_source(sim_branch, channel)
    return "simulation_native" if len(sim_source) < len(exp_branch) else "experimental"


def pair_for_metric_grid(
    exp_branch: pd.DataFrame,
    sim_branch: pd.DataFrame,
    channel: str,
    config: dict[str, Any],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    grid = choose_metric_grid(exp_branch, sim_branch, channel, config)
    if grid == "simulation_native":
        x, y_exp, y_sim = interp_test_to_sim(exp_branch, sim_branch, channel)
    else:
        x, y_exp, y_sim = interp_sim_to_test(exp_branch, sim_branch, channel)
    return grid, x, y_exp, y_sim


def interpolated_curve_path(curve_dir: Path, case: SimCase, channel: str, branch: str) -> Path:
    return curve_dir / f"{case.case_key}_{case.node}_{channel}_{branch}.csv"


def build_interpolated_curve(
    case: SimCase,
    exp_branch: pd.DataFrame,
    raw_sim_branch: pd.DataFrame,
    clean_sim_branch: pd.DataFrame,
    channel: str,
    branch: str,
    *,
    filter_window: int = 7,
) -> pd.DataFrame:
    if len(exp_branch) < 5 or len(clean_sim_branch) < 5:
        return pd.DataFrame(columns=CURVE_COLUMNS)

    xmin = max(exp_branch["diameter"].min(), clean_sim_branch["diameter"].min())
    xmax = min(exp_branch["diameter"].max(), clean_sim_branch["diameter"].max())
    exp_overlap = exp_branch[
        (exp_branch["diameter"] >= xmin) & (exp_branch["diameter"] <= xmax)
    ].copy()
    clean_overlap = clean_sim_branch[
        (clean_sim_branch["diameter"] >= xmin) & (clean_sim_branch["diameter"] <= xmax)
    ].copy()
    raw_overlap = raw_sim_branch[
        (raw_sim_branch["diameter"] >= xmin) & (raw_sim_branch["diameter"] <= xmax)
    ].copy()

    if len(exp_overlap) < 5 or len(clean_overlap) < 5:
        return pd.DataFrame(columns=CURVE_COLUMNS)

    x = exp_overlap["diameter"].to_numpy(dtype=float)
    y_exp = exp_overlap["force"].to_numpy(dtype=float)
    raw_source = prepare_interp_source(raw_overlap, channel)
    clean_source = prepare_filtered_interp_source(
        clean_overlap,
        channel,
        filter_window=filter_window,
    )
    y_raw = interp_values_at(x, raw_source, channel)
    y_clean = interp_values_at(x, clean_source, "filtered")
    valid = ~np.isnan(y_clean)

    out = pd.DataFrame(
        {
            "case": case.case_key,
            "sample": case.sample,
            "condition": case.condition,
            "node": case.node,
            "file": case.path.name,
            "channel": channel,
            "branch": branch,
            "diameter": x[valid],
            "experimental_force": y_exp[valid],
            "simulation_force_raw_interpolated": y_raw[valid],
            "simulation_force_outliers_excluded_interpolated": y_clean[valid],
        }
    )
    out["residual"] = (
        out["simulation_force_outliers_excluded_interpolated"] - out["experimental_force"]
    )
    out["abs_residual"] = out["residual"].abs()
    return out[CURVE_COLUMNS]

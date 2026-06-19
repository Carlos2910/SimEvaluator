from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .alignment import align_simulation_diameter
from .branches import split_loading_unloading
from .channels import add_derived_channels, channel_names, comparison_channel
from .config import resolve_path
from .interpolation import (
    build_interpolated_curve,
    interp_sim_to_test,
    interpolated_curve_path,
)
from .loaders import SimCase, list_sim_cases, read_experimental, read_simulation, simulation_folders
from .metrics import calculate_metrics, empty_metrics
from .outliers import add_outlier_masks
from .ranking import make_selection_summary, write_cross_dataset_comparisons
from .reports import outlier_rows


def analysis_dirs(sim_folder: Path) -> dict[str, Path]:
    analysis = sim_folder / "analysis"
    return {
        "analysis": analysis,
        "figures": analysis / "figures",
        "interpolated_curves": analysis / "interpolated_curves",
    }


def ensure_analysis_dirs(dirs: dict[str, Path]) -> None:
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)


def alignment_info(sim: pd.DataFrame) -> dict[str, object]:
    return {
        "diameter_alignment": sim.attrs.get("diameter_alignment"),
        "starting_diameter_used": sim.attrs.get("starting_diameter_used"),
        "nominal_starting_diameter": sim.attrs.get("nominal_starting_diameter"),
        "exp_min_diameter": sim.attrs.get("exp_min_diameter"),
        "max_compression_disp": sim.attrs.get("max_compression_disp"),
    }


def metric_row(
    case: SimCase,
    channel: str,
    variant: str,
    branch: str,
    n_points: int,
    n_sim_total: int,
    n_outliers_channel: int,
    n_outliers_any_channel: int,
    paired_curve_file: str,
    sim: pd.DataFrame,
    row_metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "case": case.case_key,
        "sample": case.sample,
        "condition": case.condition,
        "node": case.node,
        "file": case.path.name,
        "channel": channel,
        "metric_variant": variant,
        "branch": branch,
        "n_points": n_points,
        "n_sim_total": n_sim_total,
        "n_outliers_channel": n_outliers_channel,
        "n_outliers_any_channel": n_outliers_any_channel,
        "paired_curve_file": paired_curve_file,
        **alignment_info(sim),
        **row_metrics,
    }


def compute_case_metrics(
    case: SimCase,
    exp: pd.DataFrame,
    sim: pd.DataFrame,
    config: dict[str, Any],
    curve_dir: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    exp_branches = split_loading_unloading(exp)
    channels = channel_names(config)

    for channel in channels:
        sim_channel = sim.copy()
        sim_channel[channel] = comparison_channel(sim_channel, channel, config)
        mask_col = f"{channel}_outlier"
        raw_branches = split_loading_unloading(sim_channel)
        clean_branches = {
            branch: branch_df.loc[~branch_df[mask_col]].copy()
            for branch, branch_df in raw_branches.items()
        }

        variant_results: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            "raw": [],
            "outliers_excluded": [],
            "outliers_excluded_interpolated": [],
        }

        for variant, branch_source in (
            ("raw", raw_branches),
            ("outliers_excluded", clean_branches),
        ):
            for branch in ("loading", "unloading"):
                x, y_exp, y_sim = interp_sim_to_test(
                    exp_branches[branch], branch_source[branch], channel
                )
                row_metrics = calculate_metrics(y_exp, y_sim) if len(x) else empty_metrics()
                if len(x):
                    variant_results[variant].append((y_exp, y_sim))
                rows.append(
                    metric_row(
                        case,
                        channel,
                        variant,
                        branch,
                        len(x),
                        len(sim),
                        int(sim[mask_col].sum()),
                        int(sim["any_outlier"].sum()),
                        "",
                        sim,
                        row_metrics,
                    )
                )

        for branch in ("loading", "unloading"):
            paired = build_interpolated_curve(
                case,
                exp_branches[branch],
                raw_branches[branch],
                clean_branches[branch],
                channel,
                branch,
            )
            paired_path = interpolated_curve_path(curve_dir, case, channel, branch)
            paired.to_csv(paired_path, index=False)
            if paired.empty:
                row_metrics = empty_metrics()
            else:
                y_exp = paired["experimental_force"].to_numpy(dtype=float)
                y_sim = paired[
                    "simulation_force_outliers_excluded_interpolated"
                ].to_numpy(dtype=float)
                row_metrics = calculate_metrics(y_exp, y_sim)
                variant_results["outliers_excluded_interpolated"].append((y_exp, y_sim))
            rows.append(
                metric_row(
                    case,
                    channel,
                    "outliers_excluded_interpolated",
                    branch,
                    len(paired),
                    len(sim),
                    int(sim[mask_col].sum()),
                    int(sim["any_outlier"].sum()),
                    str(paired_path),
                    sim,
                    row_metrics,
                )
            )

        for variant, branch_values in variant_results.items():
            if branch_values:
                y_exp_combined = np.concatenate([y_exp for y_exp, _ in branch_values])
                y_sim_combined = np.concatenate([y_sim for _, y_sim in branch_values])
                combined = calculate_metrics(y_exp_combined, y_sim_combined)
                n_combined = int(len(y_exp_combined))
            else:
                combined = empty_metrics()
                n_combined = 0
            rows.append(
                metric_row(
                    case,
                    channel,
                    variant,
                    "combined_weighted",
                    n_combined,
                    len(sim),
                    int(sim[mask_col].sum()),
                    int(sim["any_outlier"].sum()),
                    "",
                    sim,
                    combined,
                )
            )

    return rows


def process_dataset(
    dataset: str,
    folder: Path,
    config: dict[str, Any],
    *,
    make_plots: bool = True,
) -> pd.DataFrame:
    dirs = analysis_dirs(folder)
    ensure_analysis_dirs(dirs)

    cases = list_sim_cases(config, dataset, folder)
    channels = channel_names(config)
    outlier_cfg = config.get("outliers", {})

    all_metrics: list[dict[str, object]] = []
    outlier_frames: list[pd.DataFrame] = []
    figure_paths = []

    for case in cases:
        exp = read_experimental(config, case.case_key)
        sim = read_simulation(case, config)
        sim = align_simulation_diameter(sim, exp, case, config)
        sim = add_derived_channels(sim, config)
        sim = add_outlier_masks(
            sim,
            channels,
            window=int(outlier_cfg.get("window", 41)),
            sigma=float(outlier_cfg.get("sigma", 6.0)),
            min_relative_prominence=float(outlier_cfg.get("min_relative_prominence", 0.03)),
        )
        all_metrics.extend(compute_case_metrics(case, exp, sim, config, dirs["interpolated_curves"]))
        outlier_frames.append(outlier_rows(case, sim, channels))
        if make_plots:
            from .plotting import plot_diagnostics

            figure_paths.append(plot_diagnostics(case, exp, sim, dirs["figures"], config))

    metrics_df = pd.DataFrame(all_metrics)
    outliers_df = (
        pd.concat([df for df in outlier_frames if not df.empty], ignore_index=True)
        if any(not df.empty for df in outlier_frames)
        else outlier_frames[0]
    )
    selection_df = make_selection_summary(metrics_df, config)

    metrics_df.to_csv(dirs["analysis"] / "metrics_by_file.csv", index=False)
    outliers_df.to_csv(dirs["analysis"] / "detected_outliers.csv", index=False)
    selection_df.to_csv(dirs["analysis"] / "selection_summary_total_force.csv", index=False)

    print(f"Processed {len(cases)} simulation files from: {folder}")
    print(f"Wrote metrics: {dirs['analysis'] / 'metrics_by_file.csv'}")
    print(f"Wrote outliers: {dirs['analysis'] / 'detected_outliers.csv'}")
    print(f"Wrote selection summary: {dirs['analysis'] / 'selection_summary_total_force.csv'}")
    print(f"Wrote {len(figure_paths)} figures to: {dirs['figures']}")
    return metrics_df


def comparison_output_folder(config: dict[str, Any]) -> Path:
    paths = config.get("paths", {})
    folder = paths.get("comparison_output_folder", config["experimental"]["folder"])
    return resolve_path(folder, base_dir=config.get("_config_dir"))


def run_pipeline(config: dict[str, Any], *, make_plots: bool = True) -> dict[str, Path]:
    dataset_metrics = {}
    for dataset, folder in simulation_folders(config).items():
        dataset_metrics[dataset] = process_dataset(dataset, folder, config, make_plots=make_plots)
    output_folder = comparison_output_folder(config)
    paths = write_cross_dataset_comparisons(dataset_metrics, output_folder, config)
    print(f"Wrote comparison outputs to: {output_folder}")
    return paths


def run_selected_plots(config: dict[str, Any]) -> list[Path]:
    from .plotting import plot_selected_cases

    output_folder = comparison_output_folder(config)
    paths = plot_selected_cases(config, output_folder)
    for path in paths:
        print(f"Wrote selected plot: {path}")
    return paths

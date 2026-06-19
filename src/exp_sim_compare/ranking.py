from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def make_selection_summary(metrics_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selection_cfg = config.get("selection", {})
    metric_variant = selection_cfg.get("metric_variant", "outliers_excluded_interpolated")
    channel = selection_cfg.get("channel", "total_force")
    rank_by = selection_cfg.get("rank_by", ["RMSE", "sMAPE", "NRMSE_percent"])
    total = metrics_df[
        (metrics_df["channel"] == channel)
        & (metrics_df["metric_variant"] == metric_variant)
        & (metrics_df["branch"] == "combined_weighted")
    ].copy()
    total = total.sort_values(["case", *rank_by, "node"])
    columns = [
        "case",
        "sample",
        "condition",
        "node",
        "file",
        "metric_variant",
        "n_points",
        "n_outliers_channel",
        "diameter_alignment",
        "starting_diameter_used",
        "nominal_starting_diameter",
        "exp_min_diameter",
        "max_compression_disp",
        "MAE",
        "RMSE",
        "bias",
        "max_abs_error",
        "sMAPE",
        "NRMSE_peak",
        "NRMSE_percent",
    ]
    return total[columns]


def write_cross_dataset_comparisons(
    dataset_metrics: dict[str, pd.DataFrame],
    output_folder: Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    output_folder.mkdir(parents=True, exist_ok=True)
    selection_cfg = config.get("selection", {})
    metric_variant = selection_cfg.get("metric_variant", "outliers_excluded_interpolated")
    selection_channel = selection_cfg.get("channel", "total_force")
    rank_by = selection_cfg.get("rank_by", ["RMSE", "sMAPE", "NRMSE_percent"])
    metric_rank_cols = list(dict.fromkeys([*rank_by, "NRMSE_peak"]))

    frames = []
    for dataset, df in dataset_metrics.items():
        selected = df[
            (df["metric_variant"] == metric_variant)
            & (df["branch"] == "combined_weighted")
        ].copy()
        selected.insert(0, "dataset", dataset)
        frames.append(selected)
    all_df = pd.concat(frames, ignore_index=True)

    for metric in metric_rank_cols:
        all_df[f"rank_{metric}"] = all_df.groupby(["case", "channel"])[metric].rank(
            method="min"
        )
    all_df["rank_sum"] = all_df[[f"rank_{metric}" for metric in rank_by]].sum(axis=1)

    cols = [
        "case",
        "channel",
        "dataset",
        "node",
        "file",
        "metric_variant",
        "RMSE",
        "sMAPE",
        "NRMSE_percent",
        "NRMSE_peak",
        "rank_RMSE",
        "rank_sMAPE",
        "rank_NRMSE_percent",
        "rank_NRMSE_peak",
        "rank_sum",
    ]
    comparison = all_df[cols].sort_values(
        ["case", "channel", "rank_sum", "RMSE", "sMAPE", "NRMSE_percent"]
    )
    best = comparison.groupby(["case", "channel"], as_index=False).first()

    paths = {}
    paths["comparison_by_channel"] = output_folder / "simulation_dataset_comparison_by_channel.csv"
    paths["best_by_channel"] = output_folder / "simulation_dataset_best_by_channel.csv"
    paths["comparison_total_force"] = output_folder / "simulation_dataset_comparison_total_force.csv"
    comparison.to_csv(paths["comparison_by_channel"], index=False)
    best.to_csv(paths["best_by_channel"], index=False)
    comparison[comparison["channel"] == selection_channel].to_csv(
        paths["comparison_total_force"], index=False
    )

    selected = (
        comparison[comparison["channel"] == selection_channel]
        .groupby("case", as_index=False)
        .first()
    )
    selected = selected[
        [
            "case",
            "dataset",
            "node",
            "file",
            "metric_variant",
            "RMSE",
            "sMAPE",
            "NRMSE_percent",
            "NRMSE_peak",
            "rank_RMSE",
            "rank_sMAPE",
            "rank_NRMSE_percent",
            "rank_sum",
        ]
    ]
    paths["selected_total_force"] = output_folder / "selected_best_simulations_total_force.csv"
    selected.to_csv(paths["selected_total_force"], index=False)

    branch_paths = write_branch_comparisons(dataset_metrics, output_folder, config)
    paths.update(branch_paths)
    paths["total_force_summary"] = write_total_force_summary(comparison, output_folder, selection_channel)
    return paths


def write_branch_comparisons(
    dataset_metrics: dict[str, pd.DataFrame],
    output_folder: Path,
    config: dict[str, Any],
) -> dict[str, Path]:
    selection_cfg = config.get("selection", {})
    metric_variant = selection_cfg.get("metric_variant", "outliers_excluded_interpolated")
    selection_channel = selection_cfg.get("channel", "total_force")
    rank_by = selection_cfg.get("rank_by", ["RMSE", "sMAPE", "NRMSE_percent"])
    metric_rank_cols = list(dict.fromkeys([*rank_by, "NRMSE_peak"]))

    frames = []
    for dataset, df in dataset_metrics.items():
        selected = df[
            (df["metric_variant"] == metric_variant)
            & (df["branch"].isin(["loading", "unloading"]))
        ].copy()
        selected.insert(0, "dataset", dataset)
        frames.append(selected)
    all_df = pd.concat(frames, ignore_index=True)

    for metric in metric_rank_cols:
        all_df[f"rank_{metric}"] = all_df.groupby(["case", "channel", "branch"])[metric].rank(
            method="min"
        )
    all_df["rank_sum"] = all_df[[f"rank_{metric}" for metric in rank_by]].sum(axis=1)

    cols = [
        "case",
        "branch",
        "channel",
        "dataset",
        "node",
        "file",
        "metric_variant",
        "n_points",
        "n_outliers_channel",
        "RMSE",
        "sMAPE",
        "NRMSE_percent",
        "NRMSE_peak",
        "rank_RMSE",
        "rank_sMAPE",
        "rank_NRMSE_percent",
        "rank_NRMSE_peak",
        "rank_sum",
        "MAE",
        "bias",
        "max_abs_error",
        "starting_diameter_used",
        "exp_min_diameter",
    ]
    comparison = all_df[cols].sort_values(
        ["case", "branch", "channel", "rank_sum", "RMSE", "sMAPE", "NRMSE_percent"]
    )
    best = comparison.groupby(["case", "branch", "channel"], as_index=False).first()

    paths = {}
    paths["comparison_by_channel_branch"] = (
        output_folder / "simulation_dataset_comparison_by_channel_branch.csv"
    )
    paths["best_by_channel_branch"] = output_folder / "simulation_dataset_best_by_channel_branch.csv"
    paths["branch_winners"] = output_folder / "sim_results_comparison_branch_winners.csv"
    comparison.to_csv(paths["comparison_by_channel_branch"], index=False)
    best.to_csv(paths["best_by_channel_branch"], index=False)
    best[best["channel"] == selection_channel].to_csv(paths["branch_winners"], index=False)

    selected = (
        comparison[comparison["channel"] == selection_channel]
        .groupby(["case", "branch"], as_index=False)
        .first()
    )
    selected = selected[
        [
            "case",
            "branch",
            "dataset",
            "node",
            "file",
            "metric_variant",
            "RMSE",
            "sMAPE",
            "NRMSE_percent",
            "NRMSE_peak",
            "rank_sum",
        ]
    ]
    paths["selected_total_force_by_branch"] = (
        output_folder / "selected_best_simulations_total_force_by_branch.csv"
    )
    selected.to_csv(paths["selected_total_force_by_branch"], index=False)
    return paths


def write_total_force_summary(comparison: pd.DataFrame, output_folder: Path, channel: str) -> Path:
    total = comparison[comparison["channel"] == channel].copy()
    rows = []
    for case, group in total.groupby("case", sort=True):
        revision = group[group["dataset"] == "sim_raw_data_revision"].sort_values(
            ["rank_sum", "RMSE", "sMAPE", "NRMSE_percent"]
        )
        data2 = group[group["dataset"] == "sim_raw_data2"].sort_values(
            ["rank_sum", "RMSE", "sMAPE", "NRMSE_percent"]
        )
        overall = group.sort_values(["rank_sum", "RMSE", "sMAPE", "NRMSE_percent"]).iloc[0]
        row = {
            "case": case,
            "metric_variant": overall["metric_variant"],
            "overall_best_dataset": overall["dataset"],
            "overall_best_node": overall["node"],
            "overall_best_RMSE": overall["RMSE"],
            "overall_best_sMAPE": overall["sMAPE"],
            "overall_best_NRMSE_percent": overall["NRMSE_percent"],
            "overall_best_NRMSE_peak": overall["NRMSE_peak"],
            "overall_rank_sum": overall["rank_sum"],
        }
        if not revision.empty:
            r = revision.iloc[0]
            row.update(
                {
                    "revision_best_node": r["node"],
                    "revision_RMSE": r["RMSE"],
                    "revision_sMAPE": r["sMAPE"],
                    "revision_NRMSE_percent": r["NRMSE_percent"],
                    "revision_NRMSE_peak": r["NRMSE_peak"],
                    "revision_rank_sum": r["rank_sum"],
                }
            )
        if not data2.empty:
            d = data2.iloc[0]
            row.update(
                {
                    "data2_best_node": d["node"],
                    "data2_RMSE": d["RMSE"],
                    "data2_sMAPE": d["sMAPE"],
                    "data2_NRMSE_percent": d["NRMSE_percent"],
                    "data2_NRMSE_peak": d["NRMSE_peak"],
                    "data2_rank_sum": d["rank_sum"],
                }
            )
        rows.append(row)
    path = output_folder / "sim_results_comparison_total_force.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

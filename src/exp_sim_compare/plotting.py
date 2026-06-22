from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "exp_sim_compare_mplconfig"),
)

import matplotlib

matplotlib.use("Agg")


def configure_matplotlib(output_dir: Path) -> None:
    mplconfig = output_dir / ".mplconfig"
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig))


import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .alignment import align_simulation_diameter
from .branches import split_loading_unloading
from .channels import add_derived_channels, channel_names, comparison_channel
from .config import feature_enabled, resolve_path, study_root
from .interpolation import build_interpolated_curve, interpolation_filter_window, median_smooth
from .loaders import parse_simulation_path, read_experimental, read_simulation, simulation_folders
from .outliers import add_outlier_masks


def load_selected_simulation_curve(
    curve_dir: Path,
    case_key: str,
    node: str,
    channel: str,
    branches: list[str],
) -> pd.DataFrame:
    frames = []
    for branch in branches:
        curve_path = curve_dir / f"{case_key}_{node}_{channel}_{branch}.csv"
        if not curve_path.exists():
            continue
        curve = pd.read_csv(curve_path)
        if not curve.empty:
            frames.append(curve)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def concat_branches(branch_frames: dict[str, pd.DataFrame], branches: list[str]) -> pd.DataFrame:
    frames = [branch_frames[branch] for branch in branches if branch in branch_frames and not branch_frames[branch].empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def selected_native_simulation_curves(
    dataset: str,
    folder: Path,
    file_name: str,
    case_key: str,
    node: str,
    channel: str,
    branches: list[str],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    sim_path = folder / file_name
    case = parse_simulation_path(dataset, folder, sim_path, config["simulation"]["filename_regex"])
    if case is None:
        return {}

    exp = read_experimental(config, case_key)
    sim = read_simulation(case, config)
    sim = align_simulation_diameter(sim, exp, case, config)
    sim = add_derived_channels(sim, config)
    channels = channel_names(config)
    outlier_cfg = config.get("outliers", {})
    channel_values = {name: comparison_channel(sim, name, config) for name in channels}
    sim = add_outlier_masks(
        sim,
        channels,
        window=int(outlier_cfg.get("window", 41)),
        window_diameter_span=(
            float(outlier_cfg["window_diameter_span"])
            if outlier_cfg.get("window_diameter_span") is not None
            else None
        ),
        min_window_points=int(outlier_cfg.get("min_window_points", 21)),
        sigma=float(outlier_cfg.get("sigma", 6.0)),
        min_relative_prominence=float(outlier_cfg.get("min_relative_prominence", 0.03)),
        exclusion_threshold_ratio=float(outlier_cfg.get("exclusion_threshold_ratio", 3.0)),
        split_by_branch=bool(outlier_cfg.get("split_by_branch", True)),
        channel_values=channel_values,
    )
    sim[channel] = comparison_channel(sim, channel, config)

    filter_window = interpolation_filter_window(config)
    raw_branches = {}
    cleaned_branches = {}
    smoothed_branches = {}
    for branch, branch_df in split_loading_unloading(sim).items():
        raw = branch_df.loc[:, ["diameter", channel]].copy()
        raw["branch"] = branch
        raw_branches[branch] = raw

        cleaned = branch_df.loc[~branch_df[f"{channel}_exclude"], ["diameter", channel]].copy()
        cleaned["branch"] = branch
        cleaned_branches[branch] = cleaned

        smoothed = cleaned.copy()
        if not smoothed.empty:
            smoothed[channel] = median_smooth(smoothed[channel].to_numpy(dtype=float), filter_window)
        smoothed_branches[branch] = smoothed

    curves = {
        "raw": concat_branches(raw_branches, branches),
        "cleaned": concat_branches(cleaned_branches, branches),
        "smoothed": concat_branches(smoothed_branches, branches),
    }
    for kind, curve in curves.items():
        if not curve.empty:
            curve["simulation_curve_kind"] = kind
            curve["case"] = case_key
            curve["dataset"] = dataset
            curve["node"] = node
    return curves


def plot_diagnostics(case, exp: pd.DataFrame, sim: pd.DataFrame, output_dir: Path, config: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(output_dir.parent)

    channels = tuple(config.get("channels", {}).keys())
    exp_branches = split_loading_unloading(exp)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()
    channel_titles = {
        "RF1": "RF1 magnitude",
        "RF2": "RF2 magnitude",
        "RF3": "RF3 magnitude",
        "total_force": "Combined force magnitude",
    }

    for ax, channel in zip(axes, channels):
        for branch, color, alpha in (("loading", "black", 0.95), ("unloading", "0.45", 0.9)):
            ax.plot(
                exp_branches[branch]["diameter"],
                exp_branches[branch]["force"],
                color=color,
                alpha=alpha,
                linewidth=2.0,
                label=f"Experimental {branch}",
            )

        sim_plot = sim.copy()
        sim_plot[channel] = comparison_channel(sim_plot, channel, config)
        sim_plot_branches = split_loading_unloading(sim_plot)

        for branch, color in (("loading", "#1f77b4"), ("unloading", "#ff7f0e")):
            ax.plot(
                sim_plot_branches[branch]["diameter"],
                sim_plot_branches[branch][channel],
                color=color,
                linewidth=1.4,
                alpha=0.85,
                label=f"Simulation {branch}",
            )

        mask = sim[f"{channel}_outlier"]
        y_plot = comparison_channel(sim, channel, config)
        if mask.any():
            ax.scatter(
                sim.loc[mask, "diameter"],
                y_plot.loc[mask],
                marker="x",
                s=55,
                linewidths=1.8,
                color="#d62728",
                label="Detected outlier",
                zorder=5,
            )

        ax.set_title(channel_titles.get(channel, channel))
        ax.set_ylabel("Radial force (N/mm)")
        ax.grid(True, alpha=0.25)

    axes[-2].set_xlabel("Diameter")
    axes[-1].set_xlabel("Diameter")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.suptitle(f"{case.case_key} {case.node} simulation vs experimental", y=0.995)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path = output_dir / f"{case.case_key}_{case.node}_diagnostics.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_interpolated_diagnostics(
    case,
    exp: pd.DataFrame,
    sim: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib(output_dir.parent)

    channels = tuple(config.get("channels", {}).keys())
    exp_branches = split_loading_unloading(exp)
    filter_window = interpolation_filter_window(config)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.ravel()
    channel_titles = {
        "RF1": "RF1 magnitude",
        "RF2": "RF2 magnitude",
        "RF3": "RF3 magnitude",
        "total_force": "Combined force magnitude",
    }

    for ax, channel in zip(axes, channels):
        for branch, color, alpha in (("loading", "black", 0.95), ("unloading", "0.45", 0.9)):
            ax.plot(
                exp_branches[branch]["diameter"],
                exp_branches[branch]["force"],
                color=color,
                alpha=alpha,
                linewidth=2.0,
                label=f"Experimental {branch}",
            )

        sim_plot = sim.copy()
        sim_plot[channel] = comparison_channel(sim_plot, channel, config)
        sim_plot_branches = split_loading_unloading(sim_plot)
        clean_branches = {
            branch: branch_df.loc[~branch_df[f"{channel}_exclude"]].copy()
            for branch, branch_df in sim_plot_branches.items()
        }

        for branch, color in (("loading", "#2ca02c"), ("unloading", "#9467bd")):
            paired = build_interpolated_curve(
                case,
                exp_branches[branch],
                sim_plot_branches[branch],
                clean_branches[branch],
                channel,
                branch,
                filter_window=filter_window,
            )
            if paired.empty:
                continue
            ax.plot(
                paired["diameter"],
                paired["simulation_force_outliers_excluded_interpolated"],
                color=color,
                linewidth=1.6,
                linestyle="--",
                alpha=0.95,
                label=f"Filtered interpolated {branch}",
            )

        ax.set_title(channel_titles.get(channel, channel))
        ax.set_ylabel("Radial force (N/mm)")
        ax.grid(True, alpha=0.25)

    axes[-2].set_xlabel("Diameter")
    axes[-1].set_xlabel("Diameter")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.suptitle(f"{case.case_key} {case.node} filtered interpolation vs experimental", y=0.995)
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path = output_dir / f"{case.case_key}_{case.node}_interpolated.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_selected_cases(config: dict[str, Any], comparison_output_folder: Path) -> list[Path]:
    plot_cfg = config.get("selected_plot", {})
    source_value = plot_cfg.get("source", "selected_best_simulations_total_force.csv")
    source = resolve_path(source_value, base_dir=comparison_output_folder)
    should_plot = feature_enabled(
        plot_cfg.get("enabled", "auto"),
        auto_condition=source.exists(),
        feature_name="selected_plot",
        require_message=f"selected_plot.enabled is true, but selected cases file is missing: {source}",
    )
    if not should_plot:
        print("Skipped selected plots: selected cases file is missing or selected_plot.enabled is false.")
        return []

    if not source.exists():
        raise FileNotFoundError(f"Selected plot source not found: {source}")
    configure_matplotlib(comparison_output_folder)
    selected = pd.read_csv(source)

    fig_cfg = plot_cfg.get("figure", {})
    figsize = tuple(fig_cfg.get("figsize", [10, 6]))
    linewidth = float(fig_cfg.get("linewidth", 3))
    experimental_linestyle = fig_cfg.get("experimental_linestyle", "-")
    simulation_linestyle = fig_cfg.get("simulation_linestyle", "--")
    simulation_linestyles = {
        "raw": ":",
        "cleaned": "--",
        "smoothed": "-.",
        "interpolated": simulation_linestyle,
        **fig_cfg.get("simulation_linestyles", {}),
    }
    simulation_alphas = {
        "raw": 0.35,
        "cleaned": 0.75,
        "smoothed": 0.95,
        "interpolated": 0.9,
        **fig_cfg.get("simulation_alphas", {}),
    }
    colors = fig_cfg.get("colors", {})
    channel = plot_cfg.get("channel", config.get("selection", {}).get("channel", "total_force"))
    branches = plot_cfg.get("branches", ["loading", "unloading"])
    simulation_data = plot_cfg.get("simulation_data", ["interpolated"])
    split_branches = bool(plot_cfg.get("split_branches", False))
    output_base = study_root(config) or comparison_output_folder
    output = resolve_path(plot_cfg.get("output_folder", "selected_plots"), base_dir=output_base)
    output.mkdir(parents=True, exist_ok=True)
    sim_folders = simulation_folders(config)

    grouped = {
        "WY-AP": selected[selected["case"].str.endswith("-AP")],
        "WY-CEEP": selected[selected["case"].str.endswith("-CEEP")],
    }
    written = []
    for group_name, group in grouped.items():
        if group.empty:
            continue
        fig, ax = plt.subplots(figsize=figsize)
        for _, row in group.sort_values("case").iterrows():
            case_key = row["case"]
            width_key = case_key.split("-")[0].replace("W", "")
            color = colors.get(width_key, "gray")
            exp = read_experimental(config, case_key)
            ax.plot(
                exp["diameter"],
                exp["force"],
                color=color,
                linewidth=linewidth,
                linestyle=experimental_linestyle,
                label=f"{case_key} experimental",
            )

            curve_dir = sim_folders[row["dataset"]] / "analysis" / "interpolated_curves"
            native_curves: dict[str, pd.DataFrame] | None = None
            for data_kind in simulation_data:
                if data_kind == "interpolated":
                    if split_branches:
                        for branch in branches:
                            curve_path = curve_dir / f"{case_key}_{row['node']}_{channel}_{branch}.csv"
                            if not curve_path.exists():
                                continue
                            curve = pd.read_csv(curve_path)
                            ax.plot(
                                curve["diameter"],
                                curve["simulation_force_outliers_excluded_interpolated"],
                                color=color,
                                linewidth=max(1.2, linewidth * 0.65),
                                linestyle=simulation_linestyles.get(data_kind, simulation_linestyle),
                                alpha=float(simulation_alphas.get(data_kind, 0.9)),
                                label=f"{case_key} {row['dataset']} {row['node']} interpolated {branch}",
                            )
                        continue
                    curve = load_selected_simulation_curve(
                        curve_dir,
                        case_key,
                        row["node"],
                        channel,
                        branches,
                    )
                    if curve.empty:
                        continue
                    ax.plot(
                        curve["diameter"],
                        curve["simulation_force_outliers_excluded_interpolated"],
                        color=color,
                        linewidth=max(1.2, linewidth * 0.65),
                        linestyle=simulation_linestyles.get(data_kind, simulation_linestyle),
                        alpha=float(simulation_alphas.get(data_kind, 0.9)),
                        label=f"{case_key} {row['dataset']} {row['node']} interpolated",
                    )
                    continue

                if data_kind not in {"raw", "cleaned", "smoothed"}:
                    raise ValueError(
                        "selected_plot.simulation_data entries must be raw, cleaned, smoothed, or interpolated"
                    )
                if native_curves is None:
                    native_curves = selected_native_simulation_curves(
                        row["dataset"],
                        sim_folders[row["dataset"]],
                        row["file"],
                        case_key,
                        row["node"],
                        channel,
                        branches,
                        config,
                    )
                curve = native_curves.get(data_kind, pd.DataFrame())
                if curve.empty:
                    continue
                ax.plot(
                    curve["diameter"],
                    curve[channel],
                    color=color,
                    linewidth=max(1.2, linewidth * 0.65),
                    linestyle=simulation_linestyles.get(data_kind, simulation_linestyle),
                    alpha=float(simulation_alphas.get(data_kind, 0.9)),
                    label=f"{case_key} {row['dataset']} {row['node']} {data_kind}",
                )

        ax.set_xlabel("Diameter")
        ax.set_ylabel("Radial force (N/mm)")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=1)
        fig.tight_layout()
        out_path = output / f"{group_name}_{channel}_selected.png"
        fig.savefig(out_path, dpi=int(fig_cfg.get("dpi", 300)), bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    return written

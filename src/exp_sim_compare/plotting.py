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

from .branches import split_loading_unloading
from .channels import comparison_channel
from .config import feature_enabled, resolve_path, study_root
from .interpolation import build_interpolated_curve, interpolation_filter_window
from .loaders import read_experimental, simulation_folders


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
            branch: branch_df.loc[~branch_df[f"{channel}_outlier"]].copy()
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
    colors = fig_cfg.get("colors", {})
    channel = plot_cfg.get("channel", config.get("selection", {}).get("channel", "total_force"))
    branches = plot_cfg.get("branches", ["loading", "unloading"])
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
                        linestyle=simulation_linestyle,
                        alpha=0.9,
                        label=f"{case_key} {row['dataset']} {row['node']} {branch}",
                    )
            else:
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
                    linestyle=simulation_linestyle,
                    alpha=0.9,
                    label=f"{case_key} {row['dataset']} {row['node']}",
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

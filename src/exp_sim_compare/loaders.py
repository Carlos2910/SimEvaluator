from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import resolve_path, resolve_study_path


@dataclass(frozen=True)
class SimCase:
    dataset: str
    folder: Path
    path: Path
    sample: str
    condition: str
    node: str
    case_key: str


def simulation_folders(config: dict[str, Any]) -> dict[str, Path]:
    sim_cfg = config["simulation"]
    if "datasets" in sim_cfg:
        root = resolve_study_path(config, sim_cfg.get("folder", "simulations"))
        datasets = sim_cfg["datasets"]
        if isinstance(datasets, list):
            return {name: (root / name).resolve() for name in datasets}
        if isinstance(datasets, dict):
            out = {}
            for name, dataset_cfg in datasets.items():
                if isinstance(dataset_cfg, dict):
                    folder = dataset_cfg.get("folder", name)
                else:
                    folder = dataset_cfg
                out[name] = resolve_path(folder, base_dir=root)
            return out
        raise ValueError("simulation.datasets must be a mapping or list")

    folders = sim_cfg["folders"]
    base_dir = config.get("_config_dir")
    if isinstance(folders, dict):
        return {name: resolve_path(path, base_dir=base_dir) for name, path in folders.items()}
    if isinstance(folders, list):
        return {Path(path).name: resolve_path(path, base_dir=base_dir) for path in folders}
    raise ValueError("simulation.folders must be a mapping or list")


def parse_simulation_path(dataset: str, folder: Path, path: Path, regex: str) -> SimCase | None:
    match = re.match(regex, path.name)
    if not match:
        return None
    sample, condition, node = match.groups()
    return SimCase(
        dataset=dataset,
        folder=folder,
        path=path,
        sample=sample,
        condition=condition,
        node=f"Node{node}",
        case_key=f"{sample}-{condition}",
    )


def list_sim_cases(config: dict[str, Any], dataset: str, folder: Path) -> list[SimCase]:
    regex = config["simulation"].get(
        "filename_regex", r"^sim-(W\d+)-(AP|CEEP)-Node(\d+)\.xlsx$"
    )
    cases = [
        case
        for path in sorted(folder.glob("*.xlsx"))
        if (case := parse_simulation_path(dataset, folder, path, regex))
    ]
    if not cases:
        raise FileNotFoundError(f"No simulation files matched in {folder}")
    return cases


def read_experimental(config: dict[str, Any], case_key: str) -> pd.DataFrame:
    exp_cfg = config["experimental"]
    folder = resolve_study_path(config, exp_cfg["folder"])
    pattern = exp_cfg.get("pattern", "{case}.txt")
    path = folder / pattern.format(case=case_key)
    if not path.exists():
        raise FileNotFoundError(f"Missing experimental file for {case_key}: {path}")

    data = np.loadtxt(path)
    return pd.DataFrame({"diameter": data[:, 0], "force": data[:, 1]})


def read_simulation(case: SimCase, config: dict[str, Any]) -> pd.DataFrame:
    sim_cfg = config["simulation"]
    displacement = sim_cfg.get("displacement_column", "disp")
    force_columns = sim_cfg.get("force_columns", ["RF1", "RF2", "RF3"])
    required = set(force_columns + [displacement])

    df = pd.read_excel(case.path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{case.path.name} is missing columns: {sorted(missing)}")

    out = df.loc[:, list(force_columns) + [displacement]].copy()
    out = out.rename(columns={displacement: "disp"}).apply(pd.to_numeric, errors="coerce")
    if out.isna().any().any():
        bad = out.isna().sum()
        raise ValueError(f"{case.path.name} contains non-numeric values:\n{bad}")
    return out

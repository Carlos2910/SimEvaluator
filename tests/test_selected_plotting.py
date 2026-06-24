from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from exp_sim_compare.plotting import load_selected_simulation_curve, plot_selected_cases


def test_load_selected_simulation_curve_concatenates_branches_in_order():
    with TemporaryDirectory() as tmp:
        curve_dir = Path(tmp)
        loading = pd.DataFrame(
            {
                "diameter": [6.0, 5.0, 4.0],
                "simulation_force_outliers_excluded_interpolated": [0.0, 1.0, 2.0],
            }
        )
        unloading = pd.DataFrame(
            {
                "diameter": [4.0, 5.0, 6.0],
                "simulation_force_outliers_excluded_interpolated": [2.1, 1.1, 0.1],
            }
        )
        # simulation_id format: {case_token}-{node}  (no underscore between them)
        loading.to_csv(curve_dir / "W6-AP-Node1_total_force_loading.csv", index=False)
        unloading.to_csv(curve_dir / "W6-AP-Node1_total_force_unloading.csv", index=False)

        curve = load_selected_simulation_curve(
            curve_dir,
            "W6-AP-Node1",       # simulation_id (was: case_key, node separately)
            "total_force",
            ["loading", "unloading"],
        )

        assert list(curve["diameter"]) == [6.0, 5.0, 4.0, 4.0, 5.0, 6.0]
        assert list(curve["simulation_force_outliers_excluded_interpolated"]) == [
            0.0,
            1.0,
            2.0,
            2.1,
            1.1,
            0.1,
        ]


def test_plot_selected_cases_writes_overall_and_dataset_specific_plots():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        study_root = root / "studies" / "crimpability_radial_force"
        comparison_dir = study_root / "comparison"
        experimental_dir = study_root / "datasets" / "experimental"
        sim_root = study_root / "datasets" / "simulations"
        comparison_dir.mkdir(parents=True)
        experimental_dir.mkdir(parents=True)

        for dataset in ("sim_raw_data_revision", "sim_raw_data2"):
            (sim_root / dataset / "analysis" / "interpolated_curves").mkdir(parents=True)

        exp_curve = np.array([[6.0, 0.0], [5.5, 1.0], [5.0, 2.0], [4.5, 3.0]])
        np.savetxt(experimental_dir / "W6-AP.txt", exp_curve)
        np.savetxt(experimental_dir / "W6-CEEP.txt", exp_curve)

        for dataset, node_by_case in {
            "sim_raw_data_revision": {"W6-AP": "Node1", "W6-CEEP": "Node1"},
            "sim_raw_data2": {"W6-AP": "Node6", "W6-CEEP": "Node6"},
        }.items():
            curve_dir = sim_root / dataset / "analysis" / "interpolated_curves"
            for case_key, node in node_by_case.items():
                # simulation_id = case_token-NodeN (derived from file column)
                simulation_id = f"{case_key}-{node}"
                for branch, forces in (
                    ("loading", [0.0, 1.0, 2.0]),
                    ("unloading", [2.0, 1.0, 0.0]),
                ):
                    pd.DataFrame(
                        {
                            "diameter": [6.0, 5.5, 5.0],
                            "simulation_force_outliers_excluded_interpolated": forces,
                        }
                    ).to_csv(
                        curve_dir / f"{simulation_id}_total_force_{branch}.csv",
                        index=False,
                    )

        pd.DataFrame(
            [
                {
                    "case": "W6-AP",
                    "dataset": "sim_raw_data_revision",
                    "node": "Node1",
                    "file": "sim-W6-AP-Node1.xlsx",
                },
                {
                    "case": "W6-CEEP",
                    "dataset": "sim_raw_data2",
                    "node": "Node6",
                    "file": "sim-W6-CEEP-Node6.xlsx",
                },
            ]
        ).to_csv(comparison_dir / "selected_best_simulations_total_force.csv", index=False)

        pd.DataFrame(
            [
                {
                    "dataset": "sim_raw_data_revision",
                    "case": "W6-AP",
                    "node": "Node1",
                    "file": "sim-W6-AP-Node1.xlsx",
                },
                {
                    "dataset": "sim_raw_data_revision",
                    "case": "W6-CEEP",
                    "node": "Node1",
                    "file": "sim-W6-CEEP-Node1.xlsx",
                },
                {
                    "dataset": "sim_raw_data2",
                    "case": "W6-AP",
                    "node": "Node6",
                    "file": "sim-W6-AP-Node6.xlsx",
                },
                {
                    "dataset": "sim_raw_data2",
                    "case": "W6-CEEP",
                    "node": "Node6",
                    "file": "sim-W6-CEEP-Node6.xlsx",
                },
            ]
        ).to_csv(comparison_dir / "selected_best_simulations_total_force_by_dataset.csv", index=False)

        config = {
            "study": {"folder": str(study_root)},
            "experimental": {"folder": "datasets/experimental", "pattern": "{case}.txt"},
            "simulation": {
                "folder": "datasets/simulations",
                "datasets": {
                    "sim_raw_data_revision": {"folder": "sim_raw_data_revision"},
                    "sim_raw_data2": {"folder": "sim_raw_data2"},
                },
            },
            "selection": {"channel": "total_force"},
            "selected_plot": {
                "enabled": True,
                "channel": "total_force",
                "output_folder": "selected_plots",
                "branches": ["loading", "unloading"],
                "simulation_data": ["interpolated"],
                "figure": {"show_legend": False, "grid": False},
            },
        }

        paths = plot_selected_cases(config, comparison_dir)
        names = {path.name for path in paths}
        # One combined figure (overall best) + one per dataset
        assert names == {
            "total_force_selected.png",
            "total_force_selected_sim_raw_data2.png",
            "total_force_selected_sim_raw_data_revision.png",
        }

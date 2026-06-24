from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from exp_sim_compare.ranking import write_cross_dataset_comparisons


def test_write_cross_dataset_comparisons_writes_dataset_specific_selection():
    config = {
        "selection": {
            "metric_variant": "outliers_excluded",
            "channel": "total_force",
            "rank_by": ["RMSE", "sMAPE", "NRMSE_percent"],
        }
    }

    def rows(node: str, rmse: float) -> list[dict[str, object]]:
        return [
            {
                "case": "W6-AP",
                "channel": "total_force",
                "metric_variant": "outliers_excluded",
                "branch": branch,
                "node": node,
                "file": f"sim-W6-AP-{node}.xlsx",
                "RMSE": rmse,
                "sMAPE": rmse / 10.0,
                "NRMSE_percent": rmse * 5.0,
                "NRMSE_peak": rmse / 20.0,
                "n_points": 10,
                "n_outliers_channel": 0,
                "MAE": rmse / 2.0,
                "bias": 0.0,
                "max_abs_error": rmse,
                "starting_diameter_used": 6.0,
                "exp_min_diameter": 4.5,
            }
            for branch in ("combined_weighted", "loading", "unloading")
        ]

    dataset_metrics = {
        "sim_raw_data_revision": pd.DataFrame(rows("Node1", 2.0)),
        "sim_raw_data2": pd.DataFrame(rows("Node6", 1.0)),
    }

    with TemporaryDirectory() as tmp:
        paths = write_cross_dataset_comparisons(dataset_metrics, Path(tmp), config)
        by_dataset_path = paths["selected_total_force_by_dataset"]
        assert by_dataset_path.exists()

        selected = pd.read_csv(by_dataset_path).sort_values(["dataset", "case"]).reset_index(drop=True)
        assert list(selected["dataset"]) == ["sim_raw_data2", "sim_raw_data_revision"]
        assert list(selected["node"]) == ["Node6", "Node1"]

        assert (Path(tmp) / "selected_best_simulations_total_force_sim_raw_data2.csv").exists()
        assert (Path(tmp) / "selected_best_simulations_total_force_sim_raw_data_revision.csv").exists()

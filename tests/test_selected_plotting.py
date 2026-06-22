from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from exp_sim_compare.plotting import load_selected_simulation_curve


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
        loading.to_csv(curve_dir / "W6-AP_Node1_total_force_loading.csv", index=False)
        unloading.to_csv(curve_dir / "W6-AP_Node1_total_force_unloading.csv", index=False)

        curve = load_selected_simulation_curve(
            curve_dir,
            "W6-AP",
            "Node1",
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

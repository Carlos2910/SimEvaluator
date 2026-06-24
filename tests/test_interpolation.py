import numpy as np
import pandas as pd

from pathlib import Path

from exp_sim_compare.loaders import SimCase
from exp_sim_compare.interpolation import interp_sim_to_test
from exp_sim_compare.interpolation import build_interpolated_curve
from exp_sim_compare.interpolation import pair_for_metric_grid


def test_interp_sim_to_test_uses_experimental_grid():
    exp = pd.DataFrame({"diameter": [0, 1, 2, 3, 4], "force": [0, 1, 2, 3, 4]})
    sim = pd.DataFrame({"diameter": [0, 1, 2, 3, 4], "total_force": [0, 2, 4, 6, 8]})

    x, y_exp, y_sim = interp_sim_to_test(exp, sim, "total_force")

    assert list(x) == [0, 1, 2, 3, 4]
    assert list(y_exp) == [0, 1, 2, 3, 4]
    assert np.allclose(y_sim, [0, 2, 4, 6, 8])


def test_interp_sim_to_test_restricts_to_overlap():
    exp = pd.DataFrame({"diameter": [0, 1, 2, 3, 4, 5], "force": [0, 1, 2, 3, 4, 5]})
    sim = pd.DataFrame({"diameter": [1, 2, 3, 4, 5], "total_force": [10, 20, 30, 40, 50]})

    x, _, y_sim = interp_sim_to_test(exp, sim, "total_force")

    assert list(x) == [1, 2, 3, 4, 5]
    assert list(y_sim) == [10, 20, 30, 40, 50]


def test_pair_for_metric_grid_auto_uses_simulation_grid_when_simulation_is_sparser():
    exp = pd.DataFrame(
        {"diameter": [0, 1, 2, 3, 4, 5, 6, 7, 8], "force": [0, 1, 2, 3, 4, 5, 6, 7, 8]}
    )
    sim = pd.DataFrame({"diameter": [0, 2, 4, 6, 8], "total_force": [0, 20, 40, 60, 80]})

    grid, x, y_exp, y_sim = pair_for_metric_grid(
        exp,
        sim,
        "total_force",
        {"metrics": {"comparison_grid": "auto"}},
    )

    assert grid == "simulation_native"
    assert list(x) == [0, 2, 4, 6, 8]
    assert list(y_exp) == [0, 2, 4, 6, 8]
    assert list(y_sim) == [0, 20, 40, 60, 80]


def test_build_interpolated_curve_filters_local_peak():
    exp = pd.DataFrame(
        {"diameter": [0, 1, 2, 3, 4, 5, 6], "force": [0, 1, 2, 3, 4, 5, 6]}
    )
    sim = pd.DataFrame(
        {"diameter": [0, 1, 2, 3, 4, 5, 6], "total_force": [0, 1, 2, 50, 3, 4, 5]}
    )
    case = SimCase(
        dataset="sim_raw_data_revision",
        folder=Path("."),
        path=Path("sim-W6-AP-Node1.xlsx"),
        case_token="W6-AP",
        simulation_id="W6-AP-Node1",
        sample="W6-AP",
        condition="",
        node="Node1",
        case_key="W6-AP",
    )

    paired = build_interpolated_curve(
        case,
        exp,
        sim,
        sim.copy(),
        "total_force",
        "loading",
        filter_window=3,
    )

    spike_row = paired.loc[paired["diameter"] == 3].iloc[0]
    assert spike_row["simulation_force_raw_interpolated"] == 50.0
    assert spike_row["simulation_force_outliers_excluded_interpolated"] == 3.0

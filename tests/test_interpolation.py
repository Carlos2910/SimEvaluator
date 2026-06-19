import numpy as np
import pandas as pd

from exp_sim_compare.interpolation import interp_sim_to_test


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

import math

import numpy as np

from exp_sim_compare.metrics import calculate_metrics


def test_calculate_metrics_values():
    y_exp = np.array([0.0, 2.0, 4.0])
    y_sim = np.array([0.0, 3.0, 2.0])

    metrics = calculate_metrics(y_exp, y_sim)

    residual = y_sim - y_exp
    expected_rmse = math.sqrt(np.mean(residual**2))
    assert metrics["RMSE"] == expected_rmse
    assert metrics["MAE"] == np.mean(np.abs(residual))
    assert metrics["bias"] == np.mean(residual)
    assert metrics["max_abs_error"] == 2.0
    assert metrics["NRMSE_peak"] == expected_rmse / (max(abs(y_exp)) + 1e-9)
    assert metrics["NRMSE_percent"] == 100.0 * expected_rmse / (max(y_exp) - min(y_exp) + 1e-9)

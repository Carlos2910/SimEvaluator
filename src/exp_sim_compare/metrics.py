from __future__ import annotations

import numpy as np


METRIC_NAMES = [
    "MAE",
    "RMSE",
    "bias",
    "max_abs_error",
    "sMAPE",
    "NRMSE_peak",
    "NRMSE_percent",
]


def calculate_metrics(y_exp: np.ndarray, y_sim: np.ndarray) -> dict[str, float]:
    if len(y_exp) == 0 or len(y_sim) == 0:
        return empty_metrics()

    residual = y_sim - y_exp
    eps = 1e-9
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))
    max_abs_error = float(np.max(np.abs(residual)))
    peak = float(np.max(np.abs(y_exp)) + eps)
    value_range = float((np.max(y_exp) - np.min(y_exp)) + eps)
    smape = float(np.mean(2.0 * np.abs(residual) / (np.abs(y_exp) + np.abs(y_sim) + eps)))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "bias": bias,
        "max_abs_error": max_abs_error,
        "sMAPE": smape,
        "NRMSE_peak": rmse / peak,
        "NRMSE_percent": 100.0 * rmse / value_range,
    }


def empty_metrics() -> dict[str, float]:
    return {name: np.nan for name in METRIC_NAMES}

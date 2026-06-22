from __future__ import annotations

import pandas as pd


def outlier_rows(case, sim: pd.DataFrame, channels: tuple[str, ...]) -> pd.DataFrame:
    rows = sim.loc[sim["any_outlier"]].copy()
    columns = [
        "case",
        "sample",
        "condition",
        "node",
        "file",
        "index",
        "diameter",
        "disp",
        *channels,
        "outlier_channels",
        "excluded_channels",
        "max_outlier_ratio",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    out = rows.reset_index(names="index")
    out["case"] = case.case_key
    out["sample"] = case.sample
    out["condition"] = case.condition
    out["node"] = case.node
    out["file"] = case.path.name
    out["outlier_channels"] = out.apply(
        lambda row: ",".join(channel for channel in channels if row[f"{channel}_outlier"]),
        axis=1,
    )
    out["excluded_channels"] = out.apply(
        lambda row: ",".join(channel for channel in channels if row.get(f"{channel}_exclude", False)),
        axis=1,
    )
    ratio_columns = [f"{channel}_outlier_ratio" for channel in channels]
    out["max_outlier_ratio"] = out[ratio_columns].max(axis=1, skipna=True)
    return out[columns]

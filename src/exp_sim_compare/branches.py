from __future__ import annotations

import pandas as pd


def split_loading_unloading(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    idx_min = int(df["diameter"].to_numpy().argmin())
    return {
        "loading": df.iloc[: idx_min + 1].copy(),
        "unloading": df.iloc[idx_min:].copy(),
    }

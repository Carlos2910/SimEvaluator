import pandas as pd

from exp_sim_compare.branches import split_loading_unloading


def test_split_loading_unloading_includes_minimum_in_both_branches():
    df = pd.DataFrame({"diameter": [6, 5, 4, 5, 6], "force": [0, 1, 2, 1, 0]})

    branches = split_loading_unloading(df)

    assert list(branches["loading"]["diameter"]) == [6, 5, 4]
    assert list(branches["unloading"]["diameter"]) == [4, 5, 6]
    assert branches["loading"].iloc[-1]["diameter"] == branches["unloading"].iloc[0]["diameter"]

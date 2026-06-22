import numpy as np
import pandas as pd

from exp_sim_compare.outliers import add_outlier_masks, hampel_outlier_details


def test_hampel_flags_borderline_without_excluding_until_ratio_threshold():
    result = hampel_outlier_details(
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        window=7,
        sigma=0.0,
        min_relative_prominence=0.4,
        exclusion_threshold_ratio=3.0,
    )

    assert result.flagged[3]
    assert not result.excluded[3]
    assert np.isclose(result.ratio[3], 2.5)


def test_add_outlier_masks_uses_channel_values_and_separates_flagged_from_excluded():
    sim = pd.DataFrame(
        {
            "diameter": [6.0, 5.5, 5.0, 4.5, 4.0, 4.5, 5.0],
            "RF1": [0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0],
        }
    )

    out = add_outlier_masks(
        sim,
        ("RF1",),
        window=7,
        sigma=0.0,
        min_relative_prominence=0.4,
        exclusion_threshold_ratio=3.0,
        split_by_branch=False,
        channel_values={"RF1": sim["RF1"].abs()},
    )

    assert out.loc[3, "RF1_outlier"]
    assert not out.loc[3, "RF1_exclude"]
    assert out["any_outlier"].any()
    assert not out["any_excluded"].any()

"""Agreement statistics for the order-effect counterbalancing check (Project_Proposal.md §6.3.2).

Pure numpy/pandas implementation, deliberately not using pingouin/statsmodels/scipy --
those pull in compiled extensions that are blocked by Smart App Control on this
machine (see notes/environment_constraints.md), so an implementation that can't
even be tested locally isn't "clean and traceable," it's just unverified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def icc_2_1(ratings: pd.DataFrame) -> float:
    """Two-way random-effects, absolute-agreement, single-rater ICC (Shrout & Fleiss 1979, ICC(2,1)).

    Args:
        ratings: subjects (rows) x raters (columns), e.g. one row per post,
            one column per counterbalancing ordering. No missing values.

    Returns:
        ICC(2,1) as a float in roughly [0, 1] (can go slightly negative for
        worse-than-chance agreement). Matches Project_Proposal.md §6.3.2's plan
        to "treat each ordering like a separate coder."
    """
    data = ratings.to_numpy(dtype=float)
    if np.isnan(data).any():
        raise ValueError("icc_2_1 does not accept missing values -- drop or impute first.")

    n, k = data.shape
    if n < 2 or k < 2:
        raise ValueError(f"Need >=2 subjects and >=2 raters, got n={n}, k={k}.")

    grand_mean = data.mean()
    subject_means = data.mean(axis=1)
    rater_means = data.mean(axis=0)

    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    ss_raters = n * np.sum((rater_means - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_raters

    ms_subjects = ss_subjects / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denominator = ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    if denominator == 0:
        return float("nan")
    return float((ms_subjects - ms_error) / denominator)


def order_sensitivity(ratings: pd.DataFrame) -> pd.Series:
    """Per-subject SD across orderings/raters -- the per-post order-sensitivity
    diagnostic from §6.3.2 (posts with genuinely ambiguous framing should show
    more spread across orderings than clear-cut ones).
    """
    return ratings.std(axis=1, ddof=1)

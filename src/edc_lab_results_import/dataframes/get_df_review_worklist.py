"""The review worklist: CRF values that disagree with the imported
result by more than is usual for that analyte.

A single percentage threshold across every analyte is crude. What a
normal disagreement looks like depends on the analyte and on the
precision it is keyed at, so each group of `utestid` and units is
judged against its own distribution instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..comparison_rules import quantize
from ..constants import DECIMAL_SLIP, NOT_SCORABLE, OUTLIER, SMALL_N
from .get_df_result_comparison import get_df_result_comparison

__all__ = [
    "DEFAULT_MIN_N",
    "DEFAULT_Z_THRESHOLD",
    "MAX_COMPARED_PLACES",
    "REVIEW_REASONS",
    "get_df_review_worklist",
]

# the conventional cut-off for a median/MAD robust z score
DEFAULT_Z_THRESHOLD = 3.5
# a group with fewer scorable rows than this has no distribution worth
# estimating, see `SMALL_N`
DEFAULT_MIN_N = 30
# values are compared at no more than this many decimal places,
# whatever the CRF stores
MAX_COMPARED_PLACES = 2
# how close |ln ratio| must be to ln 10 or ln 100 to read as a decimal
# point slip, about 5% either way
DECIMAL_SLIP_TOLERANCE = 0.05
DECIMAL_SLIP_LOG_RATIOS = (np.log(10.0), np.log(100.0))
# scales a MAD to a standard deviation for normally distributed data
MAD_TO_SD = 1.4826

# the comparison is in the CRF's units, so that is the group's units.
# Rows whose units differ from the CRF have no comparable value and are
# not in the population, see section 2.5 of the notebook
GROUP_KEY = ["utestid", "crf_units"]

# most actionable first, also the sort order of the worklist
REVIEW_REASONS = (DECIMAL_SLIP, OUTLIER, NOT_SCORABLE, SMALL_N)


def get_df_review_worklist(
    df: pd.DataFrame | None = None,
    *,
    z_threshold: float | None = None,
    min_n: int | None = None,
    flagged_only: bool | None = None,
) -> pd.DataFrame:
    """Return the CRF values to review against the imported result.

    `df` is the frame from `get_df_result_comparison`, which is read
    if not given.

    The population is every row with both a CRF value and an imported
    value in the CRF's units. Where more than one result was imported
    for the same CRF value, for example a corrected report, only the
    latest by `result_datetime` is compared. `n_imported_for_key` is
    kept so the repeats stay visible.

    Both values are rounded to the CRF's `decimal_places`, capped at
    `MAX_COMPARED_PLACES`. The discrepancy is `ln(crf / imported)`,
    which treats keying ten times too high and ten times too low
    alike. A difference of one unit in the last compared digit is
    precision, not error, and counts as no discrepancy at all.

    Each group of `utestid` and units is scored by a robust z score,
    the distance from the group median in MADs. Most rows in a group
    agree exactly, which puts the MAD at 0, so the scale is floored at
    the group's typical size of one unit in the last compared digit.

    A row is flagged, in this order, as:
        `decimal_slip`: the ratio is within about 5% of 10 or 100,
            whatever the score.
        `outlier`: |score| >= `z_threshold` in a group with at least
            `min_n` scorable rows.
        `not_scorable`: either value is zero or negative, where ln is
            undefined, and the two differ at the compared precision.
        `small_n`: any discrepancy in a group with fewer than `min_n`
            scorable rows.

    A discrepancy shared by a whole group moves the group median, so it
    is not an outlier. See section 2.3 of the notebook for that.

    Sorted by reason, then by |score|, largest first. With
    `flagged_only` False every row in the population is returned,
    scored, with `reason` NA where nothing is flagged.
    """
    z_threshold = DEFAULT_Z_THRESHOLD if z_threshold is None else z_threshold
    min_n = DEFAULT_MIN_N if min_n is None else min_n
    flagged_only = True if flagged_only is None else flagged_only
    if df is None:
        df = get_df_result_comparison()
    attrs = dict(df.attrs)

    df = add_group_scores(add_discrepancy(get_population(df)))
    df["reason"] = get_reason(df, z_threshold=z_threshold, min_n=min_n)
    if flagged_only:
        df = df.loc[df["reason"].notna()]
    df = (
        df.assign(
            _reason_order=df["reason"].map({r: i for i, r in enumerate(REVIEW_REASONS)}),
            _abs_score=df["score"].abs(),
        )
        .sort_values(["_reason_order", "_abs_score"], ascending=[True, False])
        .drop(columns=["_reason_order", "_abs_score"])
        .reset_index(drop=True)
    )
    df.attrs = attrs
    return df


def get_population(df: pd.DataFrame) -> pd.DataFrame:
    """Return the rows with both values, the latest import per CRF
    value.
    """
    crf_value = pd.to_numeric(df["crf_value"], errors="coerce")
    comparable_value = pd.to_numeric(df["comparable_value"], errors="coerce")
    return (
        df.loc[crf_value.notna() & comparable_value.notna()]
        .sort_values("result_datetime", na_position="first", kind="stable")
        .drop_duplicates(subset=["crf_id", "utestid"], keep="last")
        .copy()
    )


def add_discrepancy(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with the values at the compared precision and the
    discrepancy between them.

    `precision` is one unit in the last compared digit on the ln scale,
    relative to the imported value.
    """
    places = (
        pd.to_numeric(df["decimal_places"], errors="coerce")
        .astype("float64")
        .clip(upper=MAX_COMPARED_PLACES)
        .fillna(MAX_COMPARED_PLACES)
    )
    unit = np.power(10.0, -places)
    crf = quantize(df["crf_value"], places)
    imported = quantize(df["comparable_value"], places)
    scorable = (crf > 0) & (imported > 0)
    # a relative epsilon, the quantized values are float64
    within_precision = (crf - imported).abs() <= unit * (1 + 1e-9)

    df["compared_places"] = places.astype("int64")
    df["crf_compared"] = crf
    df["result_compared"] = imported
    df["differs"] = crf != imported
    df["scorable"] = scorable
    df["log_ratio"] = np.log(crf.where(scorable) / imported.where(scorable))
    df["deviation"] = df["log_ratio"].where(~within_precision, 0.0).where(scorable)
    df["precision"] = np.log1p(unit / imported.where(scorable))
    return df


def add_group_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with each scorable row's robust z score within its
    group of `utestid` and units.

    Rows that are not scorable take no part in their group's
    statistics and are left unscored.
    """
    scorable = df.loc[df["scorable"]]
    grouped = scorable.groupby(GROUP_KEY, dropna=False)
    centre = grouped["deviation"].transform("median")
    mad = (
        (scorable["deviation"] - centre)
        .abs()
        .groupby([scorable[k] for k in GROUP_KEY], dropna=False)
        .transform("median")
    )
    floor = grouped["precision"].transform("median")
    scale = np.maximum(MAD_TO_SD * mad, floor)

    df["group_n"] = grouped["deviation"].transform("size").reindex(df.index).astype("Int64")
    df["group_centre"] = centre.reindex(df.index)
    df["group_scale"] = scale.reindex(df.index)
    df["score"] = ((scorable["deviation"] - centre) / scale).reindex(df.index)
    return df


def get_reason(df: pd.DataFrame, *, z_threshold: float, min_n: int) -> pd.Series:
    """Return why each row is on the worklist, or NA if it is not."""
    scorable = df["scorable"].to_numpy(dtype=bool)
    has_discrepancy = (df["deviation"].fillna(0.0) != 0.0).to_numpy(dtype=bool)
    abs_log_ratio = df["log_ratio"].abs()
    is_slip = np.zeros(len(df), dtype=bool)
    for log_ratio in DECIMAL_SLIP_LOG_RATIOS:
        is_slip |= ((abs_log_ratio - log_ratio).abs() <= DECIMAL_SLIP_TOLERANCE).to_numpy(
            dtype=bool
        )
    large_group = (df["group_n"].fillna(0) >= min_n).to_numpy(dtype=bool)
    beyond = (df["score"].abs() >= z_threshold).fillna(False).to_numpy(dtype=bool)
    reason = np.select(
        [
            scorable & is_slip,
            scorable & large_group & has_discrepancy & beyond,
            ~scorable & df["differs"].to_numpy(dtype=bool),
            scorable & ~large_group & has_discrepancy,
        ],
        list(REVIEW_REASONS),
        default="",
    )
    return pd.Series(reason, index=df.index, dtype="string").replace("", pd.NA)

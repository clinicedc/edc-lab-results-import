from __future__ import annotations

from datetime import datetime, timedelta
from itertools import count
from zoneinfo import ZoneInfo

import pandas as pd
from django.test import SimpleTestCase, tag

from edc_lab_results_import.constants import DECIMAL_SLIP, NOT_SCORABLE, OUTLIER, SMALL_N
from edc_lab_results_import.dataframes import get_df_review_worklist

RESULT_DATETIME = datetime(2025, 3, 3, 9, 0, tzinfo=ZoneInfo("UTC"))


class Rows:
    """Builds rows shaped like `get_df_result_comparison`, only the
    columns the worklist reads plus an identity column.
    """

    def __init__(self):
        self.crf_ids = count(1)
        self.rows: list[dict] = []

    def add(
        self,
        crf_value: float | None,
        result_value: float | None,
        *,
        utestid: str = "haemoglobin",
        units: str = "g/dL",
        decimal_places: int | None = 1,
        crf_id: int | None = None,
        result_datetime: datetime = RESULT_DATETIME,
        n_imported_for_key: int = 1,
    ) -> int:
        crf_id = next(self.crf_ids) if crf_id is None else crf_id
        self.rows.append(
            dict(
                subject_identifier=f"101-{crf_id:08d}-1",
                crf_id=str(crf_id),
                utestid=utestid,
                crf_value=crf_value,
                crf_units=units,
                decimal_places=decimal_places,
                result_value=result_value,
                units=units,
                comparable_value=result_value,
                result_datetime=result_datetime,
                n_imported_for_key=n_imported_for_key,
            )
        )
        return crf_id

    def add_agreeing(self, n: int, value: float = 13.4, **kwargs) -> None:
        """`n` rows keyed exactly as the lab reported them."""
        for _ in range(n):
            self.add(value, value, **kwargs)

    @property
    def df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.rows).astype(
            {"crf_id": "string", "utestid": "string", "crf_units": "string", "units": "string"}
        )
        df["result_datetime"] = pd.to_datetime(df["result_datetime"], utc=True)
        df["decimal_places"] = df["decimal_places"].astype("Int64")
        df["n_imported_for_key"] = df["n_imported_for_key"].astype("Int64")
        df.attrs["pulled_datetime"] = RESULT_DATETIME
        return df


def reason_of(df: pd.DataFrame, crf_id: int) -> str | None:
    rows = df.loc[df["crf_id"] == str(crf_id), "reason"]
    if rows.empty or pd.isna(rows.iloc[0]):
        return None
    return rows.iloc[0]


@tag("lab_results_import")
class TestReviewWorklist(SimpleTestCase):
    """Each group of utest id and units is judged against its own
    distribution of discrepancies, rather than one percentage
    threshold across every analyte.
    """

    def test_outlier_in_a_large_group(self):
        rows = Rows()
        rows.add_agreeing(40)
        crf_id = rows.add(14.4, 13.4)
        df = get_df_review_worklist(rows.df)
        self.assertEqual(reason_of(df, crf_id), OUTLIER)
        self.assertEqual(len(df), 1)
        self.assertGreater(df.iloc[0]["score"], 0)

    def test_score_is_signed(self):
        rows = Rows()
        rows.add_agreeing(40)
        crf_id = rows.add(12.4, 13.4)
        df = get_df_review_worklist(rows.df)
        self.assertEqual(reason_of(df, crf_id), OUTLIER)
        self.assertLess(df.iloc[0]["score"], 0)

    def test_one_unit_in_the_last_digit_is_precision(self):
        rows = Rows()
        rows.add_agreeing(40)
        high = rows.add(13.5, 13.4)
        low = rows.add(13.3, 13.4)
        df = get_df_review_worklist(rows.df, flagged_only=False)
        self.assertIsNone(reason_of(df, high))
        self.assertIsNone(reason_of(df, low))
        self.assertEqual(df.loc[df["crf_id"] == str(high), "deviation"].iloc[0], 0.0)

    def test_compared_at_no_more_than_two_decimal_places(self):
        rows = Rows()
        rows.add_agreeing(40, value=1.23, decimal_places=4)
        crf_id = rows.add(1.2341, 1.2349, decimal_places=4)
        df = get_df_review_worklist(rows.df, flagged_only=False)
        row = df.loc[df["crf_id"] == str(crf_id)].iloc[0]
        self.assertEqual(row["compared_places"], 2)
        self.assertIsNone(reason_of(df, crf_id))

    def test_no_decimal_places_compares_at_two(self):
        rows = Rows()
        crf_id = rows.add(1.234, 1.2, decimal_places=None)
        df = get_df_review_worklist(rows.df, flagged_only=False)
        self.assertEqual(df.loc[df["crf_id"] == str(crf_id), "compared_places"].iloc[0], 2)

    def test_the_spread_is_the_groups_own(self):
        """A difference usual for one analyte is an outlier for
        another. Creatinine here is keyed loosely, haemoglobin exactly.
        """
        rows = Rows()
        rows.add_agreeing(40)
        for i in range(40):
            rows.add(80.0 + (i % 9) - 4, 80.0, utestid="creatinine", units="umol/L")
        hb = rows.add(14.4, 13.4)
        creat = rows.add(86.0, 80.0, utestid="creatinine", units="umol/L")
        df = get_df_review_worklist(rows.df, flagged_only=False)
        self.assertEqual(reason_of(df, hb), OUTLIER)
        self.assertIsNone(reason_of(df, creat))

    def test_groups_are_split_by_units(self):
        rows = Rows()
        rows.add_agreeing(40)
        rows.add_agreeing(5, value=134.0, units="g/L")
        crf_id = rows.add(144.0, 134.0, units="g/L")
        df = get_df_review_worklist(rows.df, flagged_only=False)
        row = df.loc[df["crf_id"] == str(crf_id)].iloc[0]
        self.assertEqual(row["group_n"], 6)
        self.assertEqual(row["reason"], SMALL_N)

    def test_rows_with_units_that_differ_are_not_scored(self):
        """No comparable value, see section 2.5 of the notebook."""
        rows = Rows()
        rows.add_agreeing(40)
        rows.rows[0]["comparable_value"] = None
        df = get_df_review_worklist(rows.df, flagged_only=False)
        self.assertEqual(len(df), 39)

    def test_decimal_slip_whatever_the_group(self):
        rows = Rows()
        rows.add_agreeing(3)
        ten = rows.add(134.0, 13.4)
        tenth = rows.add(1.3, 13.4)
        hundred = rows.add(1340.0, 13.4)
        df = get_df_review_worklist(rows.df)
        self.assertEqual(reason_of(df, ten), DECIMAL_SLIP)
        self.assertEqual(reason_of(df, tenth), DECIMAL_SLIP)
        self.assertEqual(reason_of(df, hundred), DECIMAL_SLIP)

    def test_decimal_slip_sorts_first(self):
        rows = Rows()
        rows.add_agreeing(40)
        rows.add(14.4, 13.4)
        slip = rows.add(134.0, 13.4)
        df = get_df_review_worklist(rows.df)
        self.assertEqual(df.iloc[0]["crf_id"], str(slip))
        self.assertEqual(list(df["reason"]), [DECIMAL_SLIP, OUTLIER])

    def test_small_group_lists_any_discrepancy(self):
        rows = Rows()
        rows.add_agreeing(5)
        crf_id = rows.add(13.7, 13.4)
        df = get_df_review_worklist(rows.df)
        self.assertEqual(list(df["crf_id"]), [str(crf_id)])
        self.assertEqual(reason_of(df, crf_id), SMALL_N)

    def test_min_n_is_configurable(self):
        rows = Rows()
        rows.add_agreeing(5)
        crf_id = rows.add(14.4, 13.4)
        df = get_df_review_worklist(rows.df, min_n=5)
        self.assertEqual(reason_of(df, crf_id), OUTLIER)

    def test_z_threshold_is_configurable(self):
        """Four units in the last digit, about z 4 against a floor of
        one unit.
        """
        rows = Rows()
        rows.add_agreeing(40)
        crf_id = rows.add(13.8, 13.4)
        self.assertEqual(reason_of(get_df_review_worklist(rows.df), crf_id), OUTLIER)
        self.assertIsNone(reason_of(get_df_review_worklist(rows.df, z_threshold=5), crf_id))

    def test_zero_or_negative_is_not_scorable(self):
        rows = Rows()
        rows.add_agreeing(40, value=0.1, utestid="basophils", units="10^9/L", decimal_places=2)
        zero = rows.add(0.0, 0.5, utestid="basophils", units="10^9/L", decimal_places=2)
        negative = rows.add(-2.0, 2.0, utestid="base_excess", units="mmol/L")
        df = get_df_review_worklist(rows.df)
        self.assertEqual(reason_of(df, zero), NOT_SCORABLE)
        self.assertEqual(reason_of(df, negative), NOT_SCORABLE)
        self.assertTrue(df.loc[df["crf_id"] == str(zero), "score"].isna().all())

    def test_zero_that_agrees_is_not_listed(self):
        rows = Rows()
        crf_id = rows.add(0.0, 0.0, decimal_places=2)
        df = get_df_review_worklist(rows.df)
        self.assertIsNone(reason_of(df, crf_id))

    def test_not_scorable_rows_take_no_part_in_the_group(self):
        rows = Rows()
        rows.add_agreeing(40)
        rows.add(0.0, 13.4)
        crf_id = rows.add(14.4, 13.4)
        df = get_df_review_worklist(rows.df)
        self.assertEqual(df.loc[df["crf_id"] == str(crf_id), "group_n"].iloc[0], 41)

    def test_compares_the_latest_report(self):
        """A corrected report replaces the original."""
        rows = Rows()
        rows.add_agreeing(40)
        crf_id = rows.add(13.4, 16.4, n_imported_for_key=2)
        rows.add(
            13.4,
            13.4,
            crf_id=crf_id,
            result_datetime=RESULT_DATETIME + timedelta(days=1),
            n_imported_for_key=2,
        )
        df = get_df_review_worklist(rows.df, flagged_only=False)
        row = df.loc[df["crf_id"] == str(crf_id)]
        self.assertEqual(len(row), 1)
        self.assertEqual(row.iloc[0]["result_compared"], 13.4)
        self.assertEqual(row.iloc[0]["n_imported_for_key"], 2)
        self.assertIsNone(reason_of(df, crf_id))

    def test_a_shared_discrepancy_is_not_an_outlier(self):
        """A whole analyte off by the same factor moves the median. That
        is a units fault, see section 2.3 of the notebook.
        """
        rows = Rows()
        for _ in range(40):
            rows.add(13.4 * 1.2, 13.4)
        df = get_df_review_worklist(rows.df)
        self.assertTrue(df.empty)

    def test_flagged_only_false_keeps_every_scored_row(self):
        rows = Rows()
        rows.add_agreeing(40)
        rows.add(14.4, 13.4)
        df = get_df_review_worklist(rows.df, flagged_only=False)
        self.assertEqual(len(df), 41)
        self.assertEqual(df.iloc[0]["reason"], OUTLIER)
        self.assertEqual(df["reason"].notna().sum(), 1)

    def test_keeps_attrs(self):
        rows = Rows()
        rows.add_agreeing(3)
        df = get_df_review_worklist(rows.df)
        self.assertEqual(df.attrs["pulled_datetime"], RESULT_DATETIME)

    def test_empty(self):
        rows = Rows()
        rows.add(None, 13.4)
        df = get_df_review_worklist(rows.df, flagged_only=False)
        self.assertTrue(df.empty)
        self.assertIn("reason", df.columns)

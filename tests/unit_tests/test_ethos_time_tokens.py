import unittest
from datetime import datetime

from pyspark.sql import SparkSession, functions as F

from cehrbert_data.decorators import AttType, att_tokens_column
from cehrbert_data.decorators.patient_event_decorator_base import (
    ETHOS_SIX_MONTH_MINUTES,
    ethos_time_tokens_func,
)

MINUTES_PER_DAY = 24 * 60


def minutes_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 60


class EthosTimeTokensFuncTest(unittest.TestCase):
    def test_gaps_shorter_than_5_minutes_have_no_token(self):
        self.assertEqual(ethos_time_tokens_func(0), [])
        self.assertEqual(ethos_time_tokens_func(4.99), [])
        # 1m08s, like the gap between the last two events of the ethos-ares example trajectory
        self.assertEqual(ethos_time_tokens_func(68 / 60), [])

    def test_interval_lower_bounds(self):
        expected = [
            (5, "5m-15m"),
            (15, "15m-45m"),
            (45, "45m-1h15m"),
            (75, "1h15m-2h"),
            (2 * 60, "2h-3h"),
            (3 * 60, "3h-5h"),
            (5 * 60, "5h-8h"),
            (8 * 60, "8h-12h"),
            (12 * 60, "12h-18h"),
            (18 * 60, "18h-1d"),
            (1 * MINUTES_PER_DAY, "1d-2d"),
            (2 * MINUTES_PER_DAY, "2d-4d"),
            (4 * MINUTES_PER_DAY, "4d-7d"),
            (7 * MINUTES_PER_DAY, "7d-12d"),
            (12 * MINUTES_PER_DAY, "12d-20d"),
            (20 * MINUTES_PER_DAY, "20d-30d"),
            (30 * MINUTES_PER_DAY, "30d-2mt"),
            (60 * MINUTES_PER_DAY, "2mt-6mt"),
        ]
        for lower_bound, token in expected:
            with self.subTest(token=token):
                self.assertEqual(ethos_time_tokens_func(lower_bound), [token])
                # just below the lower bound falls into the previous (smaller) interval
                below = ethos_time_tokens_func(lower_bound - 0.01)
                self.assertNotEqual(below, [token])

    def test_just_below_six_months_is_a_single_interval_token(self):
        self.assertEqual(ethos_time_tokens_func(ETHOS_SIX_MONTH_MINUTES - 1), ["2mt-6mt"])

    def test_six_months_and_longer_are_repeated_and_rounded(self):
        self.assertEqual(ethos_time_tokens_func(180 * MINUTES_PER_DAY), ["=6mt"])
        # 269 days is 1.49 x 180 days, 270 days is exactly 1.5 x which rounds half up
        self.assertEqual(ethos_time_tokens_func(269 * MINUTES_PER_DAY), ["=6mt"])
        self.assertEqual(ethos_time_tokens_func(270 * MINUTES_PER_DAY), ["=6mt"] * 2)
        self.assertEqual(ethos_time_tokens_func(450 * MINUTES_PER_DAY), ["=6mt"] * 3)  # 2.5 -> 3

    def test_ethos_ares_example_trajectory(self):
        # Gaps between consecutive events in the ethos-ares example, and the tokens it injects for them
        cases = [
            # between visits, 2010-07-31 -> 2011-10-30: three =6mt
            (datetime(2010, 7, 31, 21, 40, 0, 100010), datetime(2011, 10, 30), ["=6mt"] * 3),
            (datetime(2011, 10, 30, 0, 0, 0), datetime(2011, 10, 30, 19, 51, 0, 100020), ["18h-1d"]),
            (datetime(2011, 10, 30, 19, 51, 0, 100030), datetime(2011, 10, 30, 21, 17, 6, 100000), ["1h15m-2h"]),
            (datetime(2011, 10, 30, 21, 17, 6, 100000), datetime(2011, 10, 31), ["2h-3h"]),
            (datetime(2011, 10, 31), datetime(2011, 10, 31, 5, 29, 7, 400000), ["5h-8h"]),
            (datetime(2011, 10, 31, 5, 29, 7, 400000), datetime(2011, 10, 31, 5, 30, 15, 700000), []),
        ]
        for start, end, tokens in cases:
            with self.subTest(start=start, end=end):
                self.assertEqual(ethos_time_tokens_func(minutes_between(start, end)), tokens)

    def test_missing_delta(self):
        self.assertEqual(ethos_time_tokens_func(None), [])
        self.assertEqual(ethos_time_tokens_func(float("nan")), [])


class AttTokensColumnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("EthosTimeTokensUnitTest").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def _tokens(self, att_type, time_deltas):
        df = self.spark.createDataFrame(list(enumerate(time_deltas)), ["gap_id", "time_delta"])
        rows = df.withColumn("standard_concept_id", att_tokens_column(att_type, "time_delta")).collect()
        tokens = {}
        for row in rows:
            tokens.setdefault(row["gap_id"], []).append(row["standard_concept_id"])
        return tokens

    def test_ethos_and_comet_expand_a_gap_into_zero_or_more_rows(self):
        time_deltas = [2, 90, 455 * MINUTES_PER_DAY, 600 * MINUTES_PER_DAY]
        for att_type in (AttType.ETHOS, AttType.COMET):
            with self.subTest(att_type=att_type):
                tokens = self._tokens(att_type, time_deltas)
                self.assertNotIn(0, tokens)  # under 5 minutes: no row at all
                self.assertEqual(tokens[1], ["1h15m-2h"])
                self.assertEqual(tokens[2], ["=6mt"] * 3)
                self.assertEqual(tokens[3], ["=6mt"] * 3)  # 3.33 x 180 days

    def test_other_att_types_are_unchanged(self):
        tokens = self._tokens(AttType.DAY, [0, 3, 2000])
        self.assertEqual(tokens, {0: ["D0"], 1: ["D3"], 2: ["LT"]})


if __name__ == "__main__":
    unittest.main()

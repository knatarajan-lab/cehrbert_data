import os
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Union, Set, Callable

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import Column, functions as F, types as T


class AttType(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    CEHR_BERT = "cehr_bert"
    MIX = "mix"
    ETHOS = "ethos"
    COMET = "comet"
    NONE = "none"


class PatientEventDecorator(ABC):
    def __init__(self, spark: SparkSession = None, persistence_folder: str = None,):
        self.spark = spark
        self.persistence_folder = persistence_folder

    @abstractmethod
    def _decorate(self, patient_events):
        pass

    @abstractmethod
    def get_name(self):
        pass

    def decorate(self, patient_events):
        decorated_patient_events = self._decorate(patient_events)
        # parent_concept_id/element_id identify, for a code split into multiple tokens (e.g.
        # ATC/0/C07, ATC/1/A, ATC/2/B03), which original un-split code and which split position
        # a token came from; see extract_events_by_domain() in spark_utils.py. Decorators create
        # brand new tokens (ATT, demographic, death, visit boundary, etc.) that don't have a
        # split origin, so backfill them here rather than in every decorator implementation.
        if "parent_concept_id" not in decorated_patient_events.columns:
            decorated_patient_events = decorated_patient_events.withColumn(
                "parent_concept_id", F.col("standard_concept_id").cast("string")
            )
        else:
            decorated_patient_events = decorated_patient_events.withColumn(
                "parent_concept_id",
                F.coalesce(F.col("parent_concept_id"), F.col("standard_concept_id").cast("string")),
            )
        if "element_id" not in decorated_patient_events.columns:
            decorated_patient_events = decorated_patient_events.withColumn("element_id", F.lit(0))
        else:
            decorated_patient_events = decorated_patient_events.withColumn(
                "element_id", F.coalesce(F.col("element_id"), F.lit(0))
            )
        self.validate(decorated_patient_events)
        return decorated_patient_events

    def try_persist_data(self, data: DataFrame, folder_name: str) -> DataFrame:
        if self.persistence_folder and self.spark:
            temp_folder = os.path.join(self.persistence_folder, folder_name)
            data.write.mode("overwrite").parquet(temp_folder)
            return self.spark.read.parquet(temp_folder)
        return data

    def load_recursive(self) -> Optional[DataFrame]:
        if self.persistence_folder and self.spark:
            temp_folder = os.path.join(self.persistence_folder, self.get_name())
            return self.spark.read.option("recursiveFileLookup", "true").parquet(temp_folder)
        return None

    @classmethod
    def get_required_columns(cls) -> Set[str]:
        return {
            "cohort_member_id",
            "person_id",
            "standard_concept_id",
            "unit",
            "date",
            "datetime",
            "visit_occurrence_id",
            "domain",
            "concept_as_value",
            "is_numeric_type",
            "number_as_value",
            "visit_rank_order",
            "visit_segment",
            "priority",
            "date_in_week",
            "concept_value_mask",
            "mlm_skip_value",
            "age",
            "visit_concept_id",
            "visit_start_date",
            "visit_start_datetime",
            "visit_concept_order",
            "concept_order",
            "event_group_id",
            "parent_concept_id",
            "element_id",
        }

    def validate(self, patient_events: DataFrame):
        actual_column_set = set(patient_events.columns)
        expected_column_set = set(self.get_required_columns())
        if actual_column_set != expected_column_set:
            diff_left = actual_column_set - expected_column_set
            diff_right = expected_column_set - actual_column_set
            raise RuntimeError(
                f"{self}\n"
                f"actual_column_set - expected_column_set: {diff_left}\n"
                f"expected_column_set - actual_column_set: {diff_right}"
            )


def time_token_func(time_delta: int) -> Optional[str]:
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 0:
        return "W-1"
    if time_delta < 28:
        return f"W{str(math.floor(time_delta / 7))}"
    if time_delta < 360:
        return f"M{str(math.floor(time_delta / 30))}"
    return "LT"


def time_day_token(time_delta: int) -> Optional[str]:
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f"D{str(time_delta)}"
    return "LT"


def time_week_token(time_delta: int) -> Optional[str]:
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f"W{str(math.floor(time_delta / 7))}"
    return "LT"


def time_month_token(time_delta: int) -> Optional[str]:
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f"M{str(math.floor(time_delta / 30))}"
    return "LT"


def time_mix_token(time_delta: int) -> Optional[str]:
    #        WHEN day_diff <= 7 THEN CONCAT('D', day_diff)
    #         WHEN day_diff <= 30 THEN CONCAT('W', ceil(day_diff / 7))
    #         WHEN day_diff <= 360 THEN CONCAT('M', ceil(day_diff / 30))
    #         WHEN day_diff <= 720 THEN CONCAT('Q', ceil(day_diff / 90))
    #         WHEN day_diff <= 1440 THEN CONCAT('Y', ceil(day_diff / 360))
    #         ELSE 'LT'
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta <= 7:
        return f"D{str(time_delta)}"
    if time_delta <= 30:
        # e.g. 8 -> W2
        return f"W{str(math.ceil(time_delta / 7))}"
    if time_delta <= 360:
        # e.g. 31 -> M2
        return f"M{str(math.ceil(time_delta / 30))}"
    # if time_delta <= 720:
    #     # e.g. 361 -> Q5
    #     return f'Q{str(math.ceil(time_delta / 90))}'
    # if time_delta <= 1080:
    #     # e.g. 1081 -> Y2
    #     return f'Y{str(math.ceil(time_delta / 360))}'
    return "LT"


# Time-interval tokens of the original ETHOS (ethos-ares tokenization.yaml `time_intervals_spec`), as
# (token, lower bound in minutes). A gap gets the token with the largest lower bound it reaches.
ETHOS_TIME_INTERVALS = [
    ("5m-15m", 5),
    ("15m-45m", 15),
    ("45m-1h15m", 45),
    ("1h15m-2h", 75),
    ("2h-3h", 120),
    ("3h-5h", 180),
    ("5h-8h", 300),
    ("8h-12h", 480),
    ("12h-18h", 720),
    ("18h-1d", 1080),
    ("1d-2d", 1440),
    ("2d-4d", 2880),
    ("4d-7d", 5760),
    ("7d-12d", 10080),
    ("12d-20d", 17280),
    ("20d-30d", 28800),
    ("30d-2mt", 43200),
    ("2mt-6mt", 86400),
]
ETHOS_SIX_MONTH_TOKEN = "=6mt"
ETHOS_SIX_MONTH_MINUTES = 180 * 24 * 60


def ethos_time_tokens_func(time_delta_minutes: float) -> List[str]:
    """Map a time delta (in minutes) to the list of ETHOS time-interval tokens expressing it.

    Mirrors ethos-ares' inject_time_intervals:
        < 5 minutes -> no token
        >= 180 days -> "=6mt" repeated round(delta / 180 days) times, with no remainder token
        otherwise   -> the single interval token whose lower bound is the largest one reached
    """
    if time_delta_minutes is None or np.isnan(time_delta_minutes):
        return []
    if time_delta_minutes >= ETHOS_SIX_MONTH_MINUTES:
        # half away from zero like polars' round(), not Python's banker's rounding
        return [ETHOS_SIX_MONTH_TOKEN] * int(math.floor(time_delta_minutes / ETHOS_SIX_MONTH_MINUTES + 0.5))
    for token, lower_bound_minutes in reversed(ETHOS_TIME_INTERVALS):
        if time_delta_minutes >= lower_bound_minutes:
            return [token]
    return []


def get_att_function(att_type: Union[AttType, str]) -> Callable:
    # Convert the att_type str to the corresponding enum type
    if isinstance(att_type, str):
        att_type = AttType(att_type)

    if att_type == AttType.DAY:
        return time_day_token
    elif att_type == AttType.WEEK:
        return time_week_token
    elif att_type == AttType.MONTH:
        return time_month_token
    elif att_type == AttType.MIX:
        return time_mix_token
    elif att_type == AttType.CEHR_BERT:
        return time_token_func
    elif att_type in (AttType.ETHOS, AttType.COMET):
        # unlike the others this returns a list of tokens (possibly empty), not a single token
        return ethos_time_tokens_func
    return None


def att_tokens_column(att_type: Union[AttType, str], time_delta_col: str) -> Column:
    """Column of the time token(s) for a time delta, to be used with withColumn("standard_concept_id", ...).

    ETHOS/CoMET expand a gap into zero or more rows (see ethos_time_tokens_func), the other ATT types
    always produce exactly one. The time delta is in minutes for ETHOS/CoMET and in days otherwise.
    """
    if isinstance(att_type, str):
        att_type = AttType(att_type)
    if att_type in (AttType.ETHOS, AttType.COMET):
        return F.explode(F.udf(ethos_time_tokens_func, T.ArrayType(T.StringType()))(time_delta_col))
    return F.udf(get_att_function(att_type), T.StringType())(time_delta_col)

import os
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union, Set, Callable

import numpy as np
from pyspark.sql import DataFrame, SparkSession


class AttType(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    CEHR_BERT = "cehr_bert"
    MIX = "mix"
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
            "event_group_id"
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
    return None

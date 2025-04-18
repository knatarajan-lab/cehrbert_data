import math

import numpy as np
from pyspark.sql import SparkSession, DataFrame, functions as F, Window as W, types as T

from .patient_event_decorator_base import PatientEventDecorator
from .token_priority import PREDICTION_TOKEN_PRIORITY
from ..const.common import NA

LARGE_INTEGER = 1_000_000


class PredictionEventDecorator(PatientEventDecorator):
    # output_columns = [
    #     'cohort_member_id', 'person_id', 'concept_ids', 'visit_segments', 'orders',
    #     'dates', 'ages', 'visit_concept_orders', 'num_of_visits', 'num_of_concepts',
    #     'concept_value_masks', 'value_as_numbers', 'value_as_concepts', 'mlm_skip_values',
    #     'visit_concept_ids', "units"
    # ]
    def __init__(self, cohort_index: DataFrame = None, spark: SparkSession = None, persistence_folder: str = None):
        self._cohort_index = cohort_index
        super().__init__(spark=spark, persistence_folder=persistence_folder)

    def get_name(self):
        return "prediction_events"

    def _decorate(self, patient_events: DataFrame):
        """
        Patient_events contains the following columns (cohort_member_id, person_id,.

        standard_concept_id, date, visit_occurrence_id, domain, concept_value)

        :param patient_events:
        :return:
        """

        if self._cohort_index is None:
            return patient_events

        prediction_events = patient_events.select("person_id", "cohort_member_id").distinct().select(
            "person_id",
            "cohort_member_id",
            F.lit(0).alias("visit_occurrence_id"),
            F.lit(F.current_date()).alias("date"),
            F.lit(F.current_date()).alias("visit_start_date"),
            F.lit(F.current_timestamp()).alias("visit_start_datetime"),
            F.lit(0).cast(T.IntegerType()).alias("visit_concept_id"),
            F.lit("prediction_token").alias("domain"),
            F.lit(0.0).alias("number_as_value"),
            F.lit("0").alias("concept_as_value"),
            F.lit(0).alias("is_numeric_type"),
            F.lit(0).alias("concept_value_mask"),
            F.lit(0).alias("mlm_skip_value"),
            F.lit(0).alias("age"),
            F.lit(0).alias("visit_segment"),
            F.lit(LARGE_INTEGER).alias("visit_rank_order"),
            F.lit(LARGE_INTEGER).alias("date_in_week"),
            F.lit(F.current_timestamp()).alias("datetime"),
            F.lit("[END]").alias("standard_concept_id"),
            F.lit(LARGE_INTEGER).alias("visit_concept_order"),
            F.lit(LARGE_INTEGER).alias("concept_order"),
            F.lit(PREDICTION_TOKEN_PRIORITY).alias("priority"),
            F.lit(NA).alias("unit"),
            F.lit(NA).alias("event_group_id")
        )

        # Try persisting the prediction tokens
        prediction_events = self.try_persist_data(
            prediction_events,
            self.get_name()
        )
        patient_events = patient_events.unionByName(prediction_events)
        return patient_events

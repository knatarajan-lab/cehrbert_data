import os
from pyspark.sql import SparkSession, DataFrame, functions as F, Window as W, types as T

from .patient_event_decorator_base import PatientEventDecorator

from ..const.common import NA
from .token_priority import (
    YEAR_TOKEN_PRIORITY,
    AGE_TOKEN_PRIORITY,
    GENDER_TOKEN_PRIORITY,
    RACE_TOKEN_PRIORITY
)
from cehrbert_data.const.artificial_tokens import (
    RACE_UNKNOWN_TOKEN,
    GENDER_UNKNOWN_TOKEN
)


class DemographicEventDecorator(PatientEventDecorator):
    def __init__(
            self, patient_demographic,
            use_age_group: bool = False,
            spark: SparkSession = None,
            persistence_folder: str = None
    ):
        self._patient_demographic = patient_demographic
        self._use_age_group = use_age_group
        super().__init__(spark=spark, persistence_folder=persistence_folder)

    def get_name(self):
        return "demographic_events"

    def _decorate(self, patient_events: DataFrame):
        if self._patient_demographic is None:
            return patient_events

        # set(['cohort_member_id', 'person_id', 'standard_concept_id', 'date',
        #      'visit_occurrence_id', 'domain', 'value_as_number', 'value_as_concept', 'visit_rank_order',
        #      'is_numeric_type', 'visit_segment', 'priority', 'date_in_week', 'concept_value_mask',
        #      'mlm_skip_value', 'age', 'visit_concept_id'])

        # Get the first token of the patient history
        first_token_udf = F.row_number().over(
            W.partitionBy("cohort_member_id", "person_id").orderBy(
                "visit_start_datetime",
                "visit_occurrence_id",
                "priority",
                "standard_concept_id",
            )
        )

        # Identify the first token of each patient history
        patient_first_token = (
            patient_events.withColumn("token_order", first_token_udf)
            .withColumn("concept_value_mask", F.lit(0))
            .withColumn("number_as_value", F.lit(0.0).cast("float"))
            .withColumn("concept_as_value", F.lit("0").cast("string"))
            .withColumn("is_numeric_type", F.lit(0))
            .withColumn("unit", F.lit(NA))
            .withColumn("event_group_id", F.lit(NA))
            .where("token_order = 1")
            .drop("token_order")
        )

        # Udf for identifying the earliest date associated with a visit_occurrence_id
        sequence_start_year_token = (
            patient_first_token.withColumn(
                "standard_concept_id",
                F.concat(F.lit("year:"), F.year("date").cast(T.StringType())),
            )
            .withColumn("priority", F.lit(YEAR_TOKEN_PRIORITY))
            .withColumn("visit_segment", F.lit(0))
            .withColumn("date_in_week", F.lit(0))
            .withColumn("age", F.lit(-1))
            .withColumn("visit_rank_order", F.lit(0))
            .withColumn("visit_concept_order", F.lit(0))
            .withColumn("concept_order", F.lit(0))
        )

        # Try persisting the start year tokens
        sequence_start_year_token = self.try_persist_data(
            sequence_start_year_token, os.path.join(self.get_name(), "sequence_start_year_tokens")
        )

        if self._use_age_group:
            calculate_age_group_at_first_visit_udf = F.ceil(
                F.floor(F.months_between(F.col("date"), F.col("birth_datetime")) / F.lit(12) / 10)
            )
            age_at_first_visit_udf = F.concat(
                F.lit("age:"),
                (calculate_age_group_at_first_visit_udf * 10).cast(T.StringType()),
                F.lit("-"),
                ((calculate_age_group_at_first_visit_udf + 1) * 10).cast(T.StringType()),
            )
        else:
            calculate_age_at_first_visit_udf = F.ceil(
                F.months_between(F.col("date"), F.col("birth_datetime")) / F.lit(12)
            )
            age_at_first_visit_udf = F.concat(F.lit("age:"), calculate_age_at_first_visit_udf.cast(T.StringType()))

        sequence_age_token = (
            self._patient_demographic.select(F.col("person_id"), F.col("birth_datetime"))
            .join(sequence_start_year_token, "person_id")
            .withColumn("standard_concept_id", age_at_first_visit_udf)
            .withColumn("priority", F.lit(AGE_TOKEN_PRIORITY))
            .drop("birth_datetime")
        )

        # Try persisting the age tokens
        sequence_age_token = self.try_persist_data(
            sequence_age_token, os.path.join(self.get_name(), "sequence_age_tokens")
        )

        gender_token_expr = F.when(
            F.coalesce(F.col("gender_concept_id"), F.lit(0)) != 0,
            F.col("gender_concept_id").cast(T.StringType())
        ).otherwise(
            F.lit(GENDER_UNKNOWN_TOKEN)
        )
        sequence_gender_token = (
            self._patient_demographic.select(F.col("person_id"), F.col("gender_concept_id"))
            .join(sequence_start_year_token, "person_id")
            .withColumn("standard_concept_id", gender_token_expr)
            .withColumn("priority", F.lit(GENDER_TOKEN_PRIORITY))
            .drop("gender_concept_id")
        )

        # Try persisting the gender tokens
        sequence_gender_token = self.try_persist_data(
            sequence_gender_token, os.path.join(self.get_name(), "sequence_gender_tokens")
        )

        race_token_expr = F.when(
            F.coalesce(F.col("race_concept_id"), F.lit(0)) != 0,
            F.col("race_concept_id").cast(T.StringType())
        ).otherwise(
            F.lit(RACE_UNKNOWN_TOKEN)
        )
        sequence_race_token = (
            self._patient_demographic.select(F.col("person_id"), F.col("race_concept_id"))
            .join(sequence_start_year_token, "person_id")
            .withColumn("standard_concept_id", race_token_expr)
            .withColumn("priority", F.lit(RACE_TOKEN_PRIORITY))
            .drop("race_concept_id")
        )

        # Try persisting the race tokens
        sequence_race_token = self.try_persist_data(
            sequence_race_token, os.path.join(self.get_name(), "sequence_race_tokens")
        )

        patient_events = patient_events.unionByName(sequence_start_year_token)
        patient_events = patient_events.unionByName(sequence_age_token)
        patient_events = patient_events.unionByName(sequence_gender_token)
        patient_events = patient_events.unionByName(sequence_race_token)

        return patient_events

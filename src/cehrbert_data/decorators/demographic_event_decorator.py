from pyspark.sql import DataFrame, functions as F, Window as W, types as T

from .patient_event_decorator_base import PatientEventDecorator

from .token_priority import (
    YEAR_TOKEN_PRIORITY,
    AGE_TOKEN_PRIORITY,
    GENDER_TOKEN_PRIORITY,
    RACE_TOKEN_PRIORITY
)


class DemographicEventDecorator(PatientEventDecorator):
    def __init__(self, patient_demographic, use_age_group: bool = False):
        self._patient_demographic = patient_demographic
        self._use_age_group = use_age_group

    def _decorate(self, patient_events: DataFrame):
        if self._patient_demographic is None:
            return patient_events

        # set(['cohort_member_id', 'person_id', 'standard_concept_id', 'date',
        #      'visit_occurrence_id', 'domain', 'concept_value', 'visit_rank_order',
        #      'visit_segment', 'priority', 'date_in_week', 'concept_value_mask',
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
            .withColumn("concept_value", F.lit(0.0))
            .withColumn("unit", F.lit(None).cast("string"))
            .withColumn("event_group_id", F.lit("N/A"))
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

        sequence_start_year_token.cache()

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

        sequence_gender_token = (
            self._patient_demographic.select(F.col("person_id"), F.col("gender_concept_id"))
            .join(sequence_start_year_token, "person_id")
            .withColumn("standard_concept_id", F.col("gender_concept_id").cast(T.StringType()))
            .withColumn("priority", F.lit(GENDER_TOKEN_PRIORITY))
            .drop("gender_concept_id")
        )

        sequence_race_token = (
            self._patient_demographic.select(F.col("person_id"), F.col("race_concept_id"))
            .join(sequence_start_year_token, "person_id")
            .withColumn("standard_concept_id", F.col("race_concept_id").cast(T.StringType()))
            .withColumn("priority", F.lit(RACE_TOKEN_PRIORITY))
            .drop("race_concept_id")
        )

        patient_events = patient_events.unionByName(sequence_start_year_token)
        patient_events = patient_events.unionByName(sequence_age_token)
        patient_events = patient_events.unionByName(sequence_gender_token)
        patient_events = patient_events.unionByName(sequence_race_token)

        return patient_events

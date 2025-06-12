from ..const.common import (
    MEASUREMENT,
    CATEGORICAL_MEASUREMENT,
    OBSERVATION,
)
from pyspark.sql import SparkSession, DataFrame, functions as F, Window as W, types as T

from .patient_event_decorator_base import PatientEventDecorator
from .token_priority import DEFAULT_PRIORITY


class ClinicalEventDecorator(PatientEventDecorator):
    # output_columns = [
    #     'cohort_member_id', 'person_id', 'concept_ids', 'visit_segments', 'orders',
    #     'dates', 'ages', 'visit_concept_orders', 'num_of_visits', 'num_of_concepts',
    #     'concept_value_masks', 'value_as_numbers', 'value_as_concepts', 'mlm_skip_values',
    #     'visit_concept_ids', "units"
    # ]
    def __init__(self, visit_occurrence, spark: SparkSession = None, persistence_folder: str = None):
        self._visit_occurrence = visit_occurrence
        super().__init__(spark=spark, persistence_folder=persistence_folder)

    def get_name(self):
        return "clinical_events"

    def _decorate(self, patient_events: DataFrame):
        """
        Patient_events contains the following columns (cohort_member_id, person_id,.

        standard_concept_id, date, visit_occurrence_id, domain, concept_value)

        :param patient_events:
        :return:
        """

        # todo: create an assertion the dataframe contains the above columns
        valid_visit_ids = patient_events.select("visit_occurrence_id", "cohort_member_id").distinct()

        # Add visit_start_date to the patient_events dataframe and create the visit rank
        visit_rank_udf = F.row_number().over(
            W.partitionBy("person_id", "cohort_member_id").orderBy(
                "visit_start_datetime", "is_inpatient", "expired", "visit_occurrence_id"
            )
        )
        visit_segment_udf = F.col("visit_rank_order") % F.lit(2) + 1

        # The visit records are joined to the cohort members (there could be multiple entries for the same patient)
        # if multiple entries are present, we duplicate the visit records for those. If the visit_occurrence dataframe
        # contains visits for each cohort member, then we need to add cohort_member_id to the joined expression as well.
        if "cohort_member_id" in self._visit_occurrence.columns:
            joined_expr = ["visit_occurrence_id", "cohort_member_id"]
        else:
            joined_expr = ["visit_occurrence_id"]

        visits = (
            self._visit_occurrence.join(
                valid_visit_ids,
                joined_expr
            ).select(
                "person_id",
                "cohort_member_id",
                "visit_occurrence_id",
                "visit_end_date",
                F.col("visit_start_date").cast(T.DateType()).alias("visit_start_date"),
                F.to_timestamp("visit_start_datetime").alias("visit_start_datetime"),
                F.col("visit_concept_id").cast("int").isin([9201, 262, 8971, 8920]).cast("int").alias("is_inpatient"),
                F.when(F.col("discharged_to_concept_id").cast("int") == 4216643, F.lit(1))
                .otherwise(F.lit(0))
                .alias("expired"),
            )
            .withColumn("visit_rank_order", visit_rank_udf)
            .withColumn("visit_segment", visit_segment_udf)
            .drop("person_id", "expired")
        )

        # Determine the concept order depending on the visit type. For outpatient visits, we assume the concepts to
        # have the same order, whereas for inpatient visits, the concept order is determined by the time stamp.
        # the concept order needs to be generated per each cohort member because the same visit could be used
        # in multiple cohort's histories of the same patient
        concept_order_udf = F.when(
            F.col("is_inpatient") == 1,
            F.dense_rank().over(W.partitionBy("cohort_member_id", "visit_occurrence_id").orderBy("datetime")),
        ).otherwise(F.lit(1))

        # Determine the global visit concept order for each patient, this takes both visit_rank_order and concept_order
        # into account when assigning this new order.
        # e.g. visit_rank_order = [1, 1, 2, 2], concept_order = [1, 1, 1, 2] -> visit_concept_order = [1, 1, 2, 3]
        visit_concept_order_udf = F.dense_rank().over(
            W.partitionBy("person_id", "cohort_member_id").orderBy("visit_rank_order", "concept_order")
        )

        # We need to set the visit_end_date as the visit_start_date for outpatient visits
        # For inpatient visits, we use the original visit_end_date if available, otherwise
        # we will infer the visit_end_date using the max(date) of the current visit
        visit_end_date_udf = F.when(
            F.col("is_inpatient") == 1,
            F.coalesce(
                F.col("visit_end_date"),
                F.max("date").over(W.partitionBy("cohort_member_id", "visit_occurrence_id")),
            ),
        ).otherwise(F.col("visit_start_date"))

        # We need to set the visit_start_datetime at the beginning of the visit start date
        # because there could be outpatient visit records whose visit_start_datetime is set to the end of the day
        visit_start_datetime_udf = (
            F.when(
                F.col("is_inpatient") == 0,
                F.col("visit_start_date")
            ).otherwise(F.col("visit_start_datetime"))
        ).cast(T.TimestampType())

        patient_events = (
            patient_events.join(visits, ["cohort_member_id", "visit_occurrence_id"])
            .withColumn("datetime", F.coalesce(F.to_timestamp("datetime"), F.to_timestamp("date")))
            .withColumn("visit_start_datetime", visit_start_datetime_udf)
            .withColumn("visit_end_date", visit_end_date_udf)
            .withColumn("visit_end_datetime", F.date_add("visit_end_date", 1).cast(T.TimestampType()))
            .withColumn("visit_end_datetime", F.expr("visit_end_datetime - INTERVAL 1 MINUTE"))
            .withColumn("concept_order", concept_order_udf)
            .withColumn("visit_concept_order", visit_concept_order_udf)
            .drop("is_inpatient", "visit_end_date", "visit_end_datetime")
            .distinct()
        )

        # Set the priority for the events. Create the week since epoch UDF
        patient_events = (
            patient_events
            .withColumn("priority", F.lit(DEFAULT_PRIORITY))
            .withColumn(
                "date_in_week", (F.unix_timestamp("date") / F.lit(24 * 60 * 60 * 7)).cast("int")
            )
        )

        # Create the concept_value_mask field to indicate whether domain values should be skipped
        # As of now only measurement has values, so other domains would be skipped.
        patient_events = patient_events.withColumn(
            "concept_value_mask", (F.col("number_as_value").isNotNull() | F.col("concept_as_value").isNotNull()).cast("int")
        ).withColumn(
            "is_numeric_type", F.col("number_as_value").isNotNull().cast("int")
        ).withColumn(
            "mlm_skip_value",
            (F.col("domain").isin([MEASUREMENT, CATEGORICAL_MEASUREMENT])).cast("int"),
        )

        if "number_as_value" not in patient_events.schema.fieldNames():
            patient_events = patient_events.withColumn("number_as_value", F.lit(None).cast("float"))

        if "concept_as_value" not in patient_events.schema.fieldNames():
            patient_events = patient_events.withColumn("concept_as_value", F.lit(None).cast("string"))

        # Try persisting the clinical events
        patient_events = self.try_persist_data(
            patient_events,
            self.get_name()
        )

        return patient_events

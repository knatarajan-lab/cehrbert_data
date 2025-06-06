import os.path

from pyspark.sql import SparkSession, DataFrame, functions as F, types as T, Window as W

from ..const.common import NA
from ..const.artificial_tokens import (
    VS_TOKEN,
    VE_TOKEN,
    VISIT_UNKNOWN_TOKEN,
    DISCHARGE_UNKNOWN_TOKEN
)
from .patient_event_decorator_base import (
    PatientEventDecorator, AttType, get_att_function
)
from .token_priority import (
    ATT_TOKEN_PRIORITY,
    VS_TOKEN_PRIORITY,
    VISIT_TYPE_TOKEN_PRIORITY,
    DISCHARGE_TOKEN_PRIORITY,
    VE_TOKEN_PRIORITY,
    FIRST_VISIT_HOUR_TOKEN_PRIORITY,
    get_inpatient_token_priority,
    get_inpatient_att_token_priority,
    get_inpatient_hour_token_priority,
)


class AttEventDecorator(PatientEventDecorator):
    def __init__(
            self,
            visit_occurrence,
            include_visit_type,
            exclude_visit_tokens,
            att_type: AttType,
            inpatient_att_type: AttType,
            include_inpatient_hour_token: bool = False,
            spark: SparkSession = None,
            persistence_folder: str = None,
    ):
        self._visit_occurrence = visit_occurrence
        self._include_visit_type = include_visit_type
        self._exclude_visit_tokens = exclude_visit_tokens
        self._att_type = att_type
        self._inpatient_att_type = inpatient_att_type
        self._include_inpatient_hour_token = include_inpatient_hour_token
        super().__init__(spark=spark, persistence_folder=persistence_folder)

    def get_name(self):
        return "att_events"

    def _decorate(self, patient_events: DataFrame):
        if self._att_type == AttType.NONE:
            return patient_events

        # visits should the following columns (person_id,
        # visit_concept_id, visit_start_date, visit_occurrence_id, domain)
        cohort_member_person_pair = patient_events.select("person_id", "cohort_member_id").distinct()
        valid_visit_ids = patient_events.groupby(
            "cohort_member_id",
            "visit_occurrence_id",
            "visit_segment",
            "visit_rank_order",
        ).agg(
            F.min("visit_concept_order").alias("min_visit_concept_order"),
            F.max("visit_concept_order").alias("max_visit_concept_order"),
            F.min("concept_order").alias("min_concept_order"),
            F.max("concept_order").alias("max_concept_order"),
        )

        # The visit records are joined to the cohort members (there could be multiple entries for the same patient)
        # if multiple entries are present, we duplicate the visit records for those. If the visit_occurrence dataframe
        # contains visits for each cohort member, then we need to add cohort_member_id to the joined expression as well.
        if "cohort_member_id" in self._visit_occurrence.columns:
            joined_expr = ["person_id", "cohort_member_id"]
        else:
            joined_expr = ["person_id"]

        visit_occurrence = (
            self._visit_occurrence.join(
                cohort_member_person_pair,
                joined_expr
            ).select(
                "person_id",
                "cohort_member_id",
                F.col("visit_start_date").cast(T.DateType()).alias("date"),
                F.col("visit_start_date").cast(T.DateType()).alias("visit_start_date"),
                F.col("visit_start_datetime").cast(T.TimestampType()).alias("visit_start_datetime"),
                F.coalesce("visit_end_date", "visit_start_date").cast(T.DateType()).alias("visit_end_date"),
                "visit_concept_id",
                "visit_occurrence_id",
                F.lit("visit").alias("domain"),
                F.lit(0.0).cast("float").alias("number_as_value"),
                F.lit("0").cast("string").alias("concept_as_value"),
                F.lit(0).alias("is_numeric_type"),
                F.lit(0).alias("concept_value_mask"),
                F.lit(0).alias("mlm_skip_value"),
                "age",
                "discharged_to_concept_id",
            )
            .join(
                valid_visit_ids,
                ["visit_occurrence_id", "cohort_member_id"]
            )
        )

        # We assume outpatient visits end on the same day, therefore we start visit_end_date to visit_start_date due
        # to bad date
        visit_occurrence = visit_occurrence.withColumn(
            "visit_end_date",
            F.when(
                F.col("visit_concept_id").isin([9201, 262, 8971, 8920]),
                F.col("visit_end_date"),
            ).otherwise(F.col("visit_start_date")),
        )

        weeks_since_epoch_udf = (F.unix_timestamp("date") / F.lit(24 * 60 * 60 * 7)).cast("int")
        visit_occurrence = visit_occurrence.withColumn("date_in_week", weeks_since_epoch_udf)

        # Cache visit for faster processing
        visit_occurrence = self.try_persist_data(
            visit_occurrence,
            os.path.join(self.get_name(), "visit_occurrence_temp"),
        )

        visits = visit_occurrence.drop("discharged_to_concept_id")

        # (cohort_member_id, person_id, standard_concept_id, date, visit_occurrence_id, domain,
        # concept_value, visit_rank_order, visit_segment, priority, date_in_week,
        # concept_value_mask, mlm_skip_value, visit_end_date)
        visit_start_events = (
            visits.withColumn("date", F.col("visit_start_date"))
            .withColumn("datetime", F.to_timestamp("visit_start_date"))
            .withColumn("standard_concept_id", F.lit(VS_TOKEN))
            .withColumn("visit_concept_order", F.col("min_visit_concept_order"))
            .withColumn("concept_order", F.col("min_concept_order") - 1)
            .withColumn("priority", F.lit(VS_TOKEN_PRIORITY))
            .withColumn("unit", F.lit(NA))
            .withColumn("event_group_id", F.lit(NA))
            .drop("min_visit_concept_order", "max_visit_concept_order")
            .drop("min_concept_order", "max_concept_order")
        )

        visit_end_events = (
            visits.withColumn("date", F.col("visit_end_date"))
            .withColumn("datetime", F.date_add(F.to_timestamp("visit_end_date"), 1))
            .withColumn("datetime", F.expr("datetime - INTERVAL 1 MINUTE"))
            .withColumn("standard_concept_id", F.lit(VE_TOKEN))
            .withColumn("visit_concept_order", F.col("max_visit_concept_order"))
            .withColumn("concept_order", F.col("max_concept_order") + 1)
            .withColumn("priority", F.lit(VE_TOKEN_PRIORITY))
            .withColumn("unit", F.lit(NA))
            .withColumn("event_group_id", F.lit(NA))
            .drop("min_visit_concept_order", "max_visit_concept_order")
            .drop("min_concept_order", "max_concept_order")
        )

        # Get the prev days_since_epoch
        prev_visit_end_date_udf = F.lag("visit_end_date").over(
            W.partitionBy("person_id", "cohort_member_id").orderBy("visit_rank_order")
        )

        # Compute the time difference between the current record and the previous record
        time_delta_udf = F.when(F.col("prev_visit_end_date").isNull(), 0).otherwise(
            F.datediff("visit_start_date", "prev_visit_end_date")
        )

        # Udf for calculating the time token
        time_token_udf = F.udf(get_att_function(self._att_type), T.StringType())

        att_tokens = (
            visits.withColumn("datetime", F.to_timestamp("date"))
            .withColumn("prev_visit_end_date", prev_visit_end_date_udf)
            .where(F.col("prev_visit_end_date").isNotNull())
            .withColumn("time_delta", time_delta_udf)
            .withColumn(
                "time_delta",
                F.when(F.col("time_delta") < 0, F.lit(0)).otherwise(F.col("time_delta")),
            )
            .withColumn("standard_concept_id", time_token_udf("time_delta"))
            .withColumn("priority", F.lit(ATT_TOKEN_PRIORITY))
            .withColumn("visit_rank_order", F.col("visit_rank_order"))
            .withColumn("visit_concept_order", F.col("min_visit_concept_order"))
            .withColumn("concept_order", F.lit(0))
            .withColumn("unit", F.lit(NA))
            .withColumn("event_group_id", F.lit(NA))
            .drop("prev_visit_end_date", "time_delta")
            .drop("min_visit_concept_order", "max_visit_concept_order")
            .drop("min_concept_order", "max_concept_order")
        )

        if self._exclude_visit_tokens:
            artificial_tokens = att_tokens
        else:
            artificial_tokens = visit_start_events.unionByName(att_tokens).unionByName(visit_end_events)

        if self._include_visit_type:
            # make sure we don't insert 0 as the visit_type because 0 could be used in other contexts
            visit_type_token_expr = F.when(
                F.col("visit_concept_id").cast("string") == "0",
                F.lit(VISIT_UNKNOWN_TOKEN)
            ).otherwise(
                F.col("visit_concept_id").cast("string")
            )
            # insert visit type after the VS token
            visit_type_tokens = (
                visits.withColumn("standard_concept_id", visit_type_token_expr)
                .withColumn("datetime", F.to_timestamp("date"))
                .withColumn("visit_concept_order", F.col("min_visit_concept_order"))
                .withColumn("concept_order", F.lit(0))
                .withColumn("priority", F.lit(VISIT_TYPE_TOKEN_PRIORITY))
                .withColumn("unit", F.lit(NA))
                .withColumn("event_group_id", F.lit(NA))
                .drop("min_visit_concept_order", "max_visit_concept_order")
                .drop("min_concept_order", "max_concept_order")
            )
            artificial_tokens = artificial_tokens.unionByName(visit_type_tokens)

        artificial_tokens = artificial_tokens.drop("visit_end_date")

        # Try persisting artificial events
        artificial_tokens = self.try_persist_data(
            artificial_tokens,
            os.path.join(self.get_name(), "artificial_tokens"),
        )

        # Retrieving the events that are ONLY linked to inpatient visits
        inpatient_visits = (
            visit_occurrence
            .where(F.col("visit_concept_id").isin([9201, 262, 8971, 8920]))
            .select("visit_occurrence_id", "visit_end_date", "cohort_member_id")
        )
        inpatient_events = patient_events.join(
            inpatient_visits, ["visit_occurrence_id", "cohort_member_id"]
        )

        inpatient_time_token_udf = F.udf(get_att_function(self._inpatient_att_type), T.StringType())
        # Fill in the visit_end_date if null (because some visits are still ongoing at the time of data extraction)
        # Bound the event dates within visit_start_date and visit_end_date
        # Generate a span rank to indicate the position of the group of events
        # Update the priority for each span
        inpatient_events = (
            inpatient_events.withColumn(
                "visit_end_date",
                F.coalesce(
                    "visit_end_date",
                    F.max("date").over(W.partitionBy("cohort_member_id", "visit_occurrence_id")),
                ),
            )
            .withColumn(
                "date",
                F.when(F.col("date") < F.col("visit_start_date"), F.col("visit_start_date")).otherwise(
                    F.when(F.col("date") > F.col("visit_end_date"), F.col("visit_end_date")).otherwise(F.col("date"))
                ),
            )
            .withColumn("priority", get_inpatient_token_priority())
            .drop("visit_end_date")
        )

        discharge_events = (
            visit_occurrence.where(F.col("visit_concept_id").isin([9201, 262, 8971, 8920]))
            .withColumn(
                "standard_concept_id",
                F.coalesce(F.col("discharged_to_concept_id"), F.lit("0")),
            )
            .withColumn("visit_concept_order", F.col("max_visit_concept_order"))
            .withColumn("concept_order", F.col("max_concept_order") + 1)
            .withColumn("date", F.col("visit_end_date"))
            .withColumn("datetime", F.date_add(F.to_timestamp("visit_end_date"), 1))
            .withColumn("datetime", F.expr("datetime - INTERVAL 1 MINUTE"))
            .withColumn("priority", F.lit(DISCHARGE_TOKEN_PRIORITY))
            .withColumn("unit", F.lit(NA))
            .withColumn("event_group_id", F.lit(NA))
            .drop("discharged_to_concept_id", "visit_end_date")
            .drop("min_visit_concept_order", "max_visit_concept_order")
            .drop("min_concept_order", "max_concept_order")
        )

        # Set standard_concept_id to "Discharge/0" instead of "0"
        discharge_events = discharge_events.withColumn(
            "standard_concept_id",
            F.when(
                F.col("standard_concept_id").cast("string") == "0",
                F.lit(DISCHARGE_UNKNOWN_TOKEN)
            ).otherwise(
                F.col("standard_concept_id")
            )
        )

        # Add discharge events to the inpatient visits
        inpatient_events = inpatient_events.unionByName(discharge_events)

        # Try persisting the inpatient events for fasting processing
        inpatient_events = self.try_persist_data(
            inpatient_events, os.path.join(self.get_name(), "inpatient_events")
        )

        # Get the prev days_since_epoch
        inpatient_prev_date_udf = F.lag("date").over(
            W.partitionBy("cohort_member_id", "visit_occurrence_id").orderBy("concept_order")
        )

        # Compute the date difference in terms of number of days between the current record and the previous record
        inpatient_date_delta_udf = F.when(F.col("prev_date").isNull(), 0).otherwise(
            F.datediff("date", "prev_date")
        )

        # Create ATT tokens within the inpatient visits between groups of events that occur on different dates
        inpatient_att_events = (
            inpatient_events.withColumn(
                "is_span_boundary",
                F.row_number().over(
                    W.partitionBy("cohort_member_id", "visit_occurrence_id", "concept_order").orderBy("priority")
                ),
            )
            .where(F.col("is_span_boundary") == 1)
            .withColumn("prev_date", inpatient_prev_date_udf)
            .withColumn("date_delta", inpatient_date_delta_udf)
            .where(F.col("date_delta") != 0)
            .where(F.col("prev_date").isNotNull())
            .withColumn(
                "standard_concept_id",
                F.concat(F.lit("i-"), inpatient_time_token_udf("date_delta")),
            )
            .withColumn("visit_concept_order", F.col("visit_concept_order"))
            .withColumn("priority", get_inpatient_att_token_priority())
            .withColumn("concept_value_mask", F.lit(0))
            .withColumn("number_as_value", F.lit(0.0).cast("float"))
            .withColumn("concept_as_value", F.lit("0").cast("string"))
            .withColumn("is_numeric_type", F.lit(0))
            .withColumn("unit", F.lit(NA))
            .withColumn("event_group_id", F.lit(NA))
            .drop("prev_date", "date_delta", "is_span_boundary")
        )

        if self._include_inpatient_hour_token:
            # Get the previous datetime based on the concept_order within the same visit
            inpatient_prev_datetime_udf = F.lag("datetime").over(
                W.partitionBy("cohort_member_id", "visit_occurrence_id").orderBy("concept_order")
            )
            # We need to insert an ATT token between midnight and the visit start datetime
            first_inpatient_hour_delta_udf = (
                F.floor((F.unix_timestamp("visit_start_datetime") - F.unix_timestamp(
                    F.col("visit_start_datetime").cast("date"))) / 3600)
            )
            # Construct the first hour token events that are calculated as the number of hours since midnight
            first_hour_token_events = (
                visits.where(F.col("visit_concept_id").isin([9201, 262, 8971, 8920]))
                .withColumn("hour_delta", first_inpatient_hour_delta_udf)
                .where(F.col("hour_delta") > 0)
                .withColumn("date", F.col("visit_start_date"))
                .withColumn("datetime", F.to_timestamp("date"))
                .withColumn("standard_concept_id", F.concat(F.lit("i-H"), F.col("hour_delta")))
                .withColumn("visit_concept_order", F.col("min_visit_concept_order"))
                .withColumn("concept_order", F.lit(0))
                .withColumn("priority", F.lit(FIRST_VISIT_HOUR_TOKEN_PRIORITY))
                .withColumn("unit", F.lit(NA))
                .withColumn("event_group_id", F.lit(NA))
                .drop("min_visit_concept_order", "max_visit_concept_order")
                .drop("min_concept_order", "max_concept_order")
                .drop("hour_delta", "visit_end_date")
            )

            # We calculate the hour difference between groups of events if they occur on the same day.
            inpatient_hour_delta_between_groups_udf = F.when(F.col("prev_datetime").isNull(), 0).otherwise(
                F.floor((F.unix_timestamp("datetime") - F.unix_timestamp("prev_datetime")) / 3600)
            )
            # The groups of events occur on different dates, we calculate the hour difference since midnight
            hour_token_on_new_day_udf = F.floor(
                (F.unix_timestamp("datetime") - F.unix_timestamp(F.col("datetime").cast("date"))) / 3600
            )
            # Compute the time difference between the current record and the previous record using the combined logic
            inpatient_hour_delta_udf = F.when(
                F.col("prev_date") == F.col("date"), inpatient_hour_delta_between_groups_udf
            ).otherwise(
                hour_token_on_new_day_udf
            )

            # Construct hour token events between different groups of events within the inpatient visits
            inpatient_hour_events = (
                inpatient_events.withColumn("prev_date", inpatient_prev_date_udf)
                .where(F.col("prev_date").isNotNull())
                .withColumn("prev_datetime", inpatient_prev_datetime_udf)
                .withColumn("hour_delta", inpatient_hour_delta_udf)
                .where(F.col("hour_delta") > 0)
                .withColumn("standard_concept_id",  F.concat(F.lit("i-H"), F.col("hour_delta")))
                .withColumn("visit_concept_order", F.col("visit_concept_order"))
                .withColumn("priority", get_inpatient_hour_token_priority())
                .withColumn("concept_value_mask", F.lit(0))
                .withColumn("number_as_value", F.lit(0.0).cast("float"))
                .withColumn("concept_as_value", F.lit("0").cast("string"))
                .withColumn("is_numeric_type", F.lit(0))
                .withColumn("unit", F.lit(NA))
                .withColumn("event_group_id", F.lit(NA))
                .drop("prev_date", "prev_datetime", "hour_delta")
            )

            # Insert the first hour tokens between the visit type and first medical event
            inpatient_att_events = inpatient_att_events.unionByName(first_hour_token_events)
            # Insert the hour tokens between different groups of events that occur at different hours s
            inpatient_att_events = inpatient_att_events.unionByName(inpatient_hour_events)


        # Try persisting the inpatient att events
        inpatient_att_events = self.try_persist_data(
            inpatient_att_events, os.path.join(self.get_name(), "inpatient_att_events")
        )

        self.validate(inpatient_events)
        self.validate(inpatient_att_events)

        # Retrieving the events that are NOT linked to inpatient visits
        other_events = patient_events.join(
            inpatient_visits.select("visit_occurrence_id", "cohort_member_id"),
            ["visit_occurrence_id", "cohort_member_id"],
            how="left_anti",
        )
        # Try persisting the other events
        other_events = self.try_persist_data(
            other_events, os.path.join(self.get_name(), "other_events")
        )

        patient_events = inpatient_events.unionByName(inpatient_att_events).unionByName(other_events)

        self.validate(patient_events)
        self.validate(artificial_tokens)

        # artificial_tokens = artificial_tokens.select(sorted(artificial_tokens.columns))
        return patient_events.unionByName(artificial_tokens)

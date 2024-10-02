from pyspark.sql import DataFrame, functions as F, types as T, Window as W

from ..const.artificial_tokens import VS_TOKEN, VE_TOKEN
from .patient_event_decorator_base import (
    PatientEventDecorator, AttType,
    time_day_token,
    time_week_token,
    time_month_token,
    time_mix_token,
    time_token_func
)
from .token_priority import (
    ATT_TOKEN_PRIORITY,
    VS_TOKEN_PRIORITY,
    VISIT_TYPE_TOKEN_PRIORITY,
    DISCHARGE_TOKEN_PRIORITY,
    VE_TOKEN_PRIORITY,
    get_inpatient_token_priority,
    get_inpatient_att_token_priority
)


class AttEventDecorator(PatientEventDecorator):
    def __init__(
            self,
            visit_occurrence,
            include_visit_type,
            exclude_visit_tokens,
            att_type: AttType,
            include_inpatient_hour_token: bool = False,
    ):
        self._visit_occurrence = visit_occurrence
        self._include_visit_type = include_visit_type
        self._exclude_visit_tokens = exclude_visit_tokens
        self._att_type = att_type
        self._include_inpatient_hour_token = include_inpatient_hour_token

    def _decorate(self, patient_events: DataFrame):
        if self._att_type == AttType.NONE:
            return patient_events

        # visits should the following columns (person_id,
        # visit_concept_id, visit_start_date, visit_occurrence_id, domain, concept_value)
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

        visit_occurrence = (
            self._visit_occurrence.select(
                "person_id",
                F.col("visit_start_date").cast(T.DateType()).alias("date"),
                F.col("visit_start_date").cast(T.DateType()).alias("visit_start_date"),
                F.col("visit_start_datetime").cast(T.TimestampType()).alias("visit_start_datetime"),
                F.coalesce("visit_end_date", "visit_start_date").cast(T.DateType()).alias("visit_end_date"),
                "visit_concept_id",
                "visit_occurrence_id",
                F.lit("visit").alias("domain"),
                F.lit(0.0).alias("concept_value"),
                F.lit(0).alias("concept_value_mask"),
                F.lit(0).alias("mlm_skip_value"),
                "age",
                "discharged_to_concept_id",
            )
            .join(valid_visit_ids, "visit_occurrence_id")
            .join(cohort_member_person_pair, ["person_id", "cohort_member_id"])
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
        visit_occurrence.cache()

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
            .withColumn("unit", F.lit(None).cast("string"))
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
            .withColumn("unit", F.lit(None).cast("string"))
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
        if self._att_type == AttType.DAY:
            att_func = time_day_token
        elif self._att_type == AttType.WEEK:
            att_func = time_week_token
        elif self._att_type == AttType.MONTH:
            att_func = time_month_token
        elif self._att_type == AttType.MIX:
            att_func = time_mix_token
        else:
            att_func = time_token_func

        time_token_udf = F.udf(att_func, T.StringType())

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
            .withColumn("unit", F.lit(None).cast("string"))
            .drop("prev_visit_end_date", "time_delta")
            .drop("min_visit_concept_order", "max_visit_concept_order")
            .drop("min_concept_order", "max_concept_order")
        )

        if self._exclude_visit_tokens:
            artificial_tokens = att_tokens
        else:
            artificial_tokens = visit_start_events.unionByName(att_tokens).unionByName(visit_end_events)

        if self._include_visit_type:
            # insert visit type after the VS token
            visit_type_tokens = (
                visits.withColumn("standard_concept_id", F.col("visit_concept_id"))
                .withColumn("datetime", F.to_timestamp("date"))
                .withColumn("visit_concept_order", F.col("min_visit_concept_order"))
                .withColumn("concept_order", F.lit(0))
                .withColumn("priority", F.lit(VISIT_TYPE_TOKEN_PRIORITY))
                .withColumn("unit", F.lit(None).cast("string"))
                .drop("min_visit_concept_order", "max_visit_concept_order")
                .drop("min_concept_order", "max_concept_order")
            )

            artificial_tokens = artificial_tokens.unionByName(visit_type_tokens)

        artificial_tokens = artificial_tokens.drop("visit_end_date")

        # Retrieving the events that are ONLY linked to inpatient visits
        inpatient_visits = visit_occurrence.where(F.col("visit_concept_id").isin([9201, 262, 8971, 8920])).select(
            "visit_occurrence_id", "visit_end_date", "cohort_member_id"
        )
        inpatient_events = patient_events.join(inpatient_visits, ["visit_occurrence_id", "cohort_member_id"])

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
                F.coalesce(F.col("discharged_to_concept_id"), F.lit(0)),
            )
            .withColumn("visit_concept_order", F.col("max_visit_concept_order"))
            .withColumn("concept_order", F.col("max_concept_order") + 1)
            .withColumn("date", F.col("visit_end_date"))
            .withColumn("datetime", F.date_add(F.to_timestamp("visit_end_date"), 1))
            .withColumn("datetime", F.expr("datetime - INTERVAL 1 MINUTE"))
            .withColumn("priority", F.lit(DISCHARGE_TOKEN_PRIORITY))
            .withColumn("unit", F.lit(None).cast("string"))
            .drop("discharged_to_concept_id", "visit_end_date")
            .drop("min_visit_concept_order", "max_visit_concept_order")
            .drop("min_concept_order", "max_concept_order")
        )

        # Add discharge events to the inpatient visits
        inpatient_events = inpatient_events.unionByName(discharge_events)

        # Get the prev days_since_epoch
        inpatient_prev_date_udf = F.lag("date").over(
            W.partitionBy("cohort_member_id", "visit_occurrence_id").orderBy("concept_order")
        )

        # Compute the time difference between the current record and the previous record
        inpatient_time_delta_udf = F.when(F.col("prev_date").isNull(), 0).otherwise(F.datediff("date", "prev_date"))

        if self._include_inpatient_hour_token:
            # Create ATT tokens within the inpatient visits
            inpatient_prev_datetime_udf = F.lag("datetime").over(
                W.partitionBy("cohort_member_id", "visit_occurrence_id").orderBy("concept_order")
            )
            # Compute the time difference between the current record and the previous record
            inpatient_hour_delta_udf = F.when(F.col("prev_datetime").isNull(), 0).otherwise(
                F.floor((F.unix_timestamp("datetime") - F.unix_timestamp("prev_datetime")) / 3600)
            )
            inpatient_att_token = F.when(
                F.col("hour_delta") < 24, F.concat(F.lit("i-H"), F.col("hour_delta"))
            ).otherwise(F.concat(F.lit("i-"), time_token_udf("time_delta")))
            # Create ATT tokens within the inpatient visits
            inpatient_att_events = (
                inpatient_events.withColumn(
                    "is_span_boundary",
                    F.row_number().over(
                        W.partitionBy("cohort_member_id", "visit_occurrence_id", "concept_order").orderBy("priority")
                    ),
                )
                .where(F.col("is_span_boundary") == 1)
                .withColumn("prev_date", inpatient_prev_date_udf)
                .withColumn("time_delta", inpatient_time_delta_udf)
                .withColumn("prev_datetime", inpatient_prev_datetime_udf)
                .withColumn("hour_delta", inpatient_hour_delta_udf)
                .where(F.col("prev_date").isNotNull())
                .where(F.col("hour_delta") > 0)
                .withColumn("standard_concept_id", inpatient_att_token)
                .withColumn("visit_concept_order", F.col("visit_concept_order"))
                .withColumn("priority", get_inpatient_att_token_priority())
                .withColumn("concept_value_mask", F.lit(0))
                .withColumn("concept_value", F.lit(0.0))
                .withColumn("unit", F.lit(None).cast("string"))
                .drop("prev_date", "time_delta", "is_span_boundary")
                .drop("prev_datetime", "hour_delta")
            )
        else:
            # Create ATT tokens within the inpatient visits
            inpatient_att_events = (
                inpatient_events.withColumn(
                    "is_span_boundary",
                    F.row_number().over(
                        W.partitionBy("cohort_member_id", "visit_occurrence_id", "concept_order").orderBy("priority")
                    ),
                )
                .where(F.col("is_span_boundary") == 1)
                .withColumn("prev_date", inpatient_prev_date_udf)
                .withColumn("time_delta", inpatient_time_delta_udf)
                .where(F.col("time_delta") != 0)
                .where(F.col("prev_date").isNotNull())
                .withColumn(
                    "standard_concept_id",
                    F.concat(F.lit("i-"), time_token_udf("time_delta")),
                )
                .withColumn("visit_concept_order", F.col("visit_concept_order"))
                .withColumn("priority", get_inpatient_att_token_priority())
                .withColumn("concept_value_mask", F.lit(0))
                .withColumn("concept_value", F.lit(0.0))
                .withColumn("unit", F.lit(None).cast("string"))
                .drop("prev_date", "time_delta", "is_span_boundary")
            )

        self.validate(inpatient_events)
        self.validate(inpatient_att_events)

        # Retrieving the events that are NOT linked to inpatient visits
        other_events = patient_events.join(
            inpatient_visits.select("visit_occurrence_id", "cohort_member_id"),
            ["visit_occurrence_id", "cohort_member_id"],
            how="left_anti",
        )

        patient_events = inpatient_events.unionByName(inpatient_att_events).unionByName(other_events)

        self.validate(patient_events)
        self.validate(artificial_tokens)

        # artificial_tokens = artificial_tokens.select(sorted(artificial_tokens.columns))
        return patient_events.unionByName(artificial_tokens)

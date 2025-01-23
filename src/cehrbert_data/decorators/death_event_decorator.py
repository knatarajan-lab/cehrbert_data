import os
from pyspark.sql import SparkSession, DataFrame, functions as F, Window as W, types as T

from ..const.common import NA
from ..const.artificial_tokens import VS_TOKEN, VE_TOKEN, DEATH_TOKEN
from .patient_event_decorator_base import (
    PatientEventDecorator,
    AttType,
    time_day_token,
    time_week_token,
    time_month_token,
    time_mix_token,
    time_token_func
)
from .token_priority import (
    VS_TOKEN_PRIORITY,
    VE_TOKEN_PRIORITY,
    ATT_TOKEN_PRIORITY,
    DEATH_TOKEN_PRIORITY
)


class DeathEventDecorator(PatientEventDecorator):
    def __init__(self, death, att_type, spark: SparkSession = None, persistence_folder: str = None):
        self._death = death
        self._att_type = att_type
        super().__init__(spark=spark, persistence_folder=persistence_folder)

    def get_name(self):
        return "death_tokens"

    def _decorate(self, patient_events: DataFrame):
        if self._death is None:
            return patient_events

        death_records = patient_events.join(self._death.select("person_id", "death_date"), "person_id")

        max_visit_occurrence_id = death_records.select(F.max("visit_occurrence_id").alias("max_visit_occurrence_id"))

        last_ve_events = (
            death_records.where(F.col("standard_concept_id") == VE_TOKEN)
            .withColumn(
                "record_rank",
                F.row_number().over(
                    W.partitionBy("person_id", "cohort_member_id").orderBy(
                        F.desc("datetime"),
                        F.desc("visit_rank_order")
                    )
                ),
            )
            .where(F.col("record_rank") == 1)
            .drop("record_rank")
        )

        last_ve_events.cache()
        # set(['cohort_member_id', 'person_id', 'standard_concept_id', 'date',
        #      'visit_occurrence_id', 'domain', 'concept_value', 'visit_rank_order',
        #      'visit_segment', 'priority', 'date_in_week', 'concept_value_mask',
        #      'mlm_skip_value', 'age', 'visit_concept_id'])
        artificial_visit_id = F.row_number().over(
            W.partitionBy(F.lit(0)).orderBy("person_id", "cohort_member_id")
        ) + F.col("max_visit_occurrence_id")

        death_records = (
            last_ve_events.crossJoin(max_visit_occurrence_id)
            .withColumn("visit_occurrence_id", artificial_visit_id)
            .withColumn("standard_concept_id", F.lit(DEATH_TOKEN))
            .withColumn("domain", F.lit("death"))
            .withColumn("visit_rank_order", F.lit(100) + F.col("visit_rank_order"))
            .withColumn("priority", F.lit(DEATH_TOKEN_PRIORITY))
            .withColumn("event_group_id", F.lit(NA))
            .drop("max_visit_occurrence_id")
        )

        vs_records = (
            death_records
            .withColumn("standard_concept_id", F.lit(VS_TOKEN))
            .withColumn("priority", F.lit(VS_TOKEN_PRIORITY))
            .withColumn("unit", F.lit(NA))
            .withColumn("event_group_id", F.lit(NA))
        )

        ve_records = (
            death_records
            .withColumn("standard_concept_id", F.lit(VE_TOKEN))
            .withColumn("priority", F.lit(VE_TOKEN_PRIORITY))
            .withColumn("unit", F.lit(NA))
            .withColumn("event_group_id", F.lit(NA))
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

        death_events = death_records.withColumn(
            "death_date",
            F.when(F.col("death_date") < F.col("date"), F.col("date")).otherwise(F.col("death_date")),
        )
        death_events = (
            death_events.withColumn("time_delta", F.datediff("death_date", "date"))
            .withColumn("standard_concept_id", time_token_udf("time_delta"))
            .withColumn("priority", F.lit(ATT_TOKEN_PRIORITY))
            .withColumn("unit", F.lit(NA))
            .withColumn("event_group_id", F.lit(NA))
            .drop("time_delta")
        )

        new_tokens = death_events.unionByName(vs_records).unionByName(death_records).unionByName(ve_records)
        new_tokens = new_tokens.drop("death_date")
        new_tokens = self.try_persist_data(
            new_tokens,
            os.path.join(self.get_name(), "death_events")
        )
        self.validate(new_tokens)

        return patient_events.unionByName(new_tokens)

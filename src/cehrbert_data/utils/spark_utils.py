import argparse
import logging
import os.path
from os import path
from typing import List, Tuple

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window as W
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql import DataFrame, SparkSession

from cehrbert_data.config.output_names import QUALIFIED_CONCEPT_LIST_PATH
from cehrbert_data.const.common import (
    MEASUREMENT,
    OBSERVATION,
    DEVICE_EXPOSURE,
    PROCESSED_MEASUREMENT,
    PROCESSED_OBSERVATION,
    PROCESSED_DEVICE,
    CDM_TABLES,
    PERSON,
    VISIT_OCCURRENCE,
    CONCEPT,
    NA,
)
from cehrbert_data.decorators import (
    AttType,
    DeathEventDecorator,
    DemographicEventDecorator,
    AttEventDecorator,
    ClinicalEventDecorator,
    PredictionEventDecorator,
    time_token_func,
)

from cehrbert_data.utils.vocab_utils import roll_up_to_drug_ingredients, roll_up_diagnosis, roll_up_procedure

DOMAIN_KEY_FIELDS = {
    "condition_occurrence_id": [
        (
            "condition_concept_id",
            "condition_start_date",
            "condition_start_datetime",
            "condition"
        )
    ],
    "procedure_occurrence_id": [
        (
            "procedure_concept_id",
            "procedure_date",
            "procedure_datetime",
            "procedure"
        )
    ],
    "drug_exposure_id": [
        (
            "drug_concept_id",
            "drug_exposure_start_date",
            "drug_exposure_start_datetime",
            "drug"
        )
    ],
    "measurement_id": [
        (
            "measurement_concept_id",
            "measurement_date",
            "measurement_datetime",
            "measurement"
        )
    ],
    "observation_id": [
        (
            "observation_concept_id",
            "observation_date",
            "observation_datetime",
            "observation"
        )
    ],
    "device_exposure_id": [
        (
            "device_concept_id",
            "device_exposure_start_date",
            "device_exposure_start_datetime",
            "device"
        )
    ],
    "death_date": [("cause_concept_id", "death_date", "death_datetime", "death")],
    "visit_concept_id": [
        ("visit_concept_id", "visit_start_date", "visit"),
        ("discharged_to_concept_id", "visit_end_date", "visit"),
    ],
}

LOGGER = logging.getLogger(__name__)


def get_key_fields(domain_table: DataFrame) -> List[Tuple[str, str, str, str]]:
    field_names = domain_table.schema.fieldNames()
    for k, v in DOMAIN_KEY_FIELDS.items():
        if k in field_names:
            return v
    return [
        (
            get_concept_id_field(domain_table),
            get_domain_date_field(domain_table),
            get_domain_datetime_field(domain_table),
            get_domain_field(domain_table)
        )
    ]


def domain_has_unit(domain_table: DataFrame) -> bool:
    for f in domain_table.schema.fieldNames():
        if "unit_concept_id" in f:
            return True
    return False


def get_domain_id_field(domain_table: DataFrame) -> str:
    table_fields = domain_table.schema.fieldNames()
    candidate_id_fields = [
        f for f in table_fields
        if not f.endswith("_concept_id") and f.endswith("_id")
    ]
    if candidate_id_fields:
        return candidate_id_fields[0]
    raise ValueError(f"{domain_table} does not have a valid id columns: {table_fields}")


def get_domain_date_field(domain_table: DataFrame) -> str:
    # extract the domain start_date column
    return [f for f in domain_table.schema.fieldNames() if "date" in f][0]


def get_domain_datetime_field(domain_table: DataFrame) -> str:
    # extract the domain start_date column
    return [f for f in domain_table.schema.fieldNames() if "datetime" in f][0]


def get_concept_id_field(domain_table: DataFrame) -> str:
    return [f for f in domain_table.schema.fieldNames() if "concept_id" in f][0]


def get_domain_field(domain_table: DataFrame) -> str:
    return get_concept_id_field(domain_table).replace("_concept_id", "")


def is_domain_numeric(domain_table_name: str) -> bool:
    for numeric_domain_table in [MEASUREMENT, OBSERVATION, DEVICE_EXPOSURE]:
        if numeric_domain_table.startswith(domain_table_name):
            return True
    return False


def extract_events_by_domain(
        domain_table: DataFrame,
        **kwargs
) -> DataFrame:
    """Standardize the format of OMOP domain tables using a time frame.
    Keyword arguments:
    domain_tables -- the array containing the OMOP domain tables except visit_occurrence
        except measurement

    The output columns of the domain table is converted to the same standard format as the following
    (person_id, standard_concept_id, date, lower_bound, upper_bound, domain).
    In this case, co-occurrence is defined as those concept ids that have co-occurred
    within the same time window of a patient.
    """
    ehr_events = None
    # extract the domain concept_id from the table fields. E.g. condition_concept_id from
    # condition_occurrence extract the domain start_date column extract the name of the table
    for (
            concept_id_field,
            date_field,
            datetime_field,
            domain_table_name
    ) in get_key_fields(domain_table):

        if is_domain_numeric(domain_table_name):
            concept = kwargs.get("concept")
            spark = kwargs.get("spark", None)
            persistence_folder = kwargs.get("persistence_folder", None)
            refresh = kwargs.get("refresh_measurement", False)
            aggregate_by_hour = kwargs.get("aggregate_by_hour", False)

            if domain_table_name == MEASUREMENT:
                get_events_func = get_measurement_events
            elif domain_table_name == OBSERVATION:
                get_events_func = get_observation_events
            elif DEVICE_EXPOSURE.startswith(domain_table_name):
                get_events_func = get_device_events
            else:
                raise RuntimeError("Cannot extract events by domain table")

            domain_records = get_events_func(
                domain_table,
                concept=concept,
                refresh=refresh,
                spark=spark,
                persistence_folder=persistence_folder,
                aggregate_by_hour=aggregate_by_hour
            )
            # Filter out the zero concept numeric events
            domain_records = domain_records.where(F.col("standard_concept_id") != "0")
        else:
            # Remove records that don't have a date or standard_concept_id
            domain_records = domain_table.where(F.col(date_field).isNotNull()).where(
                F.col(concept_id_field).isNotNull()
            )
            datetime_field_udf = F.to_timestamp(F.coalesce(datetime_field, date_field), "yyyy-MM-dd HH:mm:ss")
            domain_records = (
                domain_records.where(F.col(concept_id_field).cast("string") != "0")
                .withColumn("date", F.to_date(F.col(date_field)))
                .withColumn("datetime", datetime_field_udf)
            )
            domain_records = domain_records.select(
                domain_records["person_id"],
                domain_records[concept_id_field].alias("standard_concept_id"),
                domain_records["date"].cast("date"),
                domain_records["datetime"].cast(T.TimestampType()),
                domain_records["visit_occurrence_id"],
                F.lit(domain_table_name).alias("domain"),
                F.lit(None).cast("string").alias("event_group_id"),
                F.lit(None).cast("float").alias("number_as_value"),
                F.lit(None).cast("string").alias("concept_as_value"),
                F.col("unit") if domain_has_unit(domain_records) else F.lit(NA).alias("unit"),
            ).distinct()

            # Remove "Patient Died" from condition_occurrence
            if domain_table_name == "condition_occurrence":
                domain_records = domain_records.where("condition_concept_id != 4216643")

        if ehr_events is None:
            ehr_events = domain_records
        else:
            ehr_events = ehr_events.unionByName(domain_records)

    return ehr_events


def preprocess_domain_table(
        spark,
        input_folder,
        domain_table_name,
        with_diagnosis_rollup=False,
        with_drug_rollup=True,
):
    domain_table = spark.read.parquet(os.path.join(input_folder, domain_table_name))
    if "concept" in domain_table_name.lower():
        return domain_table

    # lowercase the schema fields
    domain_table = domain_table.select([F.col(f_n).alias(f_n.lower()) for f_n in domain_table.schema.fieldNames()])

    for f_n in domain_table.schema.fieldNames():
        if "date" in f_n and "datetime" not in f_n:
            # convert date columns to the date type
            domain_table = domain_table.withColumn(f_n, F.to_date(f_n))
        elif "datetime" in f_n:
            # convert date columns to the datetime type
            domain_table = domain_table.withColumn(f_n, F.to_timestamp(f_n))

    if domain_table_name == "visit_occurrence":
        # This is CDM 5.2, we need to rename this column to be CDM 5.3 compatible
        if "discharge_to_concept_id" in domain_table.schema.fieldNames():
            domain_table = domain_table.withColumnRenamed("discharge_to_concept_id", "discharged_to_concept_id")

    if with_drug_rollup:
        if (
                domain_table_name == "drug_exposure"
                and path.exists(os.path.join(input_folder, "concept"))
                and path.exists(os.path.join(input_folder, "concept_ancestor"))
        ):
            concept = spark.read.parquet(os.path.join(input_folder, "concept"))
            concept_ancestor = spark.read.parquet(os.path.join(input_folder, "concept_ancestor"))
            domain_table = roll_up_to_drug_ingredients(domain_table, concept, concept_ancestor)

    if with_diagnosis_rollup:
        if (
                domain_table_name == "condition_occurrence"
                and path.exists(os.path.join(input_folder, "concept"))
                and path.exists(os.path.join(input_folder, "concept_relationship"))
        ):
            concept = spark.read.parquet(os.path.join(input_folder, "concept"))
            concept_relationship = spark.read.parquet(os.path.join(input_folder, "concept_relationship"))
            domain_table = roll_up_diagnosis(domain_table, concept, concept_relationship)

        if (
                domain_table_name == "procedure_occurrence"
                and path.exists(os.path.join(input_folder, "concept"))
                and path.exists(os.path.join(input_folder, "concept_ancestor"))
        ):
            concept = spark.read.parquet(os.path.join(input_folder, "concept"))
            concept_ancestor = spark.read.parquet(os.path.join(input_folder, "concept_ancestor"))
            domain_table = roll_up_procedure(domain_table, concept, concept_ancestor)

    return domain_table


def create_sequence_data(patient_event, date_filter=None, include_visit_type=False, classic_bert_seq=False):
    """
    Create a sequence of the events associated with one patient in a chronological order.

    :param patient_event:
    :param date_filter:
    :param include_visit_type:
    :param classic_bert_seq:
    :return:
    """

    if date_filter:
        patient_event = patient_event.where(F.col("date") >= date_filter)

    # Define a list of custom UDFs for creating custom columns
    date_conversion_udf = (F.unix_timestamp("date") / F.lit(24 * 60 * 60 * 7)).cast("int")
    earliest_visit_date_udf = F.min("date_in_week").over(W.partitionBy("visit_occurrence_id"))

    visit_rank_udf = F.dense_rank().over(W.partitionBy("cohort_member_id", "person_id").orderBy("earliest_visit_date"))
    visit_segment_udf = F.col("visit_rank_order") % F.lit(2) + 1

    # Derive columns
    patient_event = (
        patient_event.where("visit_occurrence_id IS NOT NULL")
        .withColumn("date_in_week", date_conversion_udf)
        .withColumn("earliest_visit_date", earliest_visit_date_udf)
        .withColumn("visit_rank_order", visit_rank_udf)
        .withColumn("visit_segment", visit_segment_udf)
        .withColumn("priority", F.lit(0))
    )

    if classic_bert_seq:
        # Udf for identifying the earliest date associated with a visit_occurrence_id
        visit_start_date_udf = F.first("date").over(
            W.partitionBy("cohort_member_id", "person_id", "visit_occurrence_id").orderBy("date")
        )

        # Udf for identifying the previous visit_occurrence_id
        prev_visit_occurrence_id_udf = F.lag("visit_occurrence_id").over(
            W.partitionBy("cohort_member_id", "person_id").orderBy("visit_start_date", "visit_occurrence_id")
        )

        # We can achieve this by overwriting the record with the earliest time stamp
        separator_events = (
            patient_event.withColumn("visit_start_date", visit_start_date_udf)
            .withColumn("prev_visit_occurrence_id", prev_visit_occurrence_id_udf)
            .where("prev_visit_occurrence_id IS NOT NULL")
            .where("visit_occurrence_id <> prev_visit_occurrence_id")
            .withColumn("domain", F.lit("Separator"))
            .withColumn("standard_concept_id", F.lit("SEP"))
            .withColumn("priority", F.lit(-1))
            .withColumn("visit_segment", F.lit(0))
            .select(patient_event.schema.fieldNames())
        )

        # Combine this artificial token SEP with the original data
        patient_event = patient_event.union(separator_events)

    order_udf = F.row_number().over(
        W.partitionBy("cohort_member_id", "person_id").orderBy(
            "earliest_visit_date",
            "visit_occurrence_id",
            "priority",
            "date_in_week",
            "standard_concept_id",
        )
    )
    # Group the data into sequences
    output_columns = [
        "order",
        "date_in_week",
        "standard_concept_id",
        "visit_segment",
        "age",
        "visit_rank_order",
    ]

    if include_visit_type:
        output_columns.append("visit_concept_id")

    # Group by data by person_id and put all the events into a list
    # The order of the list is determined by the order column
    patient_grouped_events = (
        patient_event.withColumn("order", order_udf)
        .withColumn("date_concept_id_period", F.struct(output_columns))
        .groupBy("person_id", "cohort_member_id")
        .agg(
            F.sort_array(F.collect_set("date_concept_id_period")).alias("date_concept_id_period"),
            F.min("earliest_visit_date").alias("earliest_visit_date"),
            F.max("date").alias("max_event_date"),
            F.max("visit_rank_order").alias("num_of_visits"),
            F.count("standard_concept_id").alias("num_of_concepts"),
        )
        .withColumn(
            "orders",
            F.col("date_concept_id_period.order").cast(T.ArrayType(T.IntegerType())),
        )
        .withColumn("dates", F.col("date_concept_id_period.date_in_week"))
        .withColumn("concept_ids", F.col("date_concept_id_period.standard_concept_id"))
        .withColumn("visit_segments", F.col("date_concept_id_period.visit_segment"))
        .withColumn("ages", F.col("date_concept_id_period.age"))
        .withColumn("visit_concept_orders", F.col("date_concept_id_period.visit_rank_order"))
    )

    # Default columns in the output dataframe
    columns_for_output = [
        "cohort_member_id",
        "person_id",
        "earliest_visit_date",
        "max_event_date",
        "orders",
        "dates",
        "ages",
        "concept_ids",
        "visit_segments",
        "visit_concept_orders",
        "num_of_visits",
        "num_of_concepts",
    ]

    if include_visit_type:
        patient_grouped_events = patient_grouped_events.withColumn(
            "visit_concept_ids", F.col("date_concept_id_period.visit_concept_id")
        )
        columns_for_output.append("visit_concept_ids")

    return patient_grouped_events.select(columns_for_output)


def create_sequence_data_with_att(
        patient_events,
        visit_occurrence,
        date_filter=None,
        include_visit_type=False,
        exclude_visit_tokens=False,
        patient_demographic=None,
        death=None,
        att_type: AttType = AttType.CEHR_BERT,
        inpatient_att_type: AttType = AttType.MIX,
        exclude_demographic: bool = True,
        use_age_group: bool = False,
        include_inpatient_hour_token: bool = False,
        cohort_index: DataFrame = None,
        spark: SparkSession = None,
        persistence_folder: str = None,
):
    """
    Create a sequence of the events associated with one patient in a chronological order.

    :param patient_events:
    :param visit_occurrence:
    :param date_filter:
    :param include_visit_type:
    :param exclude_visit_tokens:
    :param patient_demographic:
    :param death:
    :param att_type:
    :param inpatient_att_type:
    :param exclude_demographic:
    :param use_age_group:
    :param include_inpatient_hour_token:
    :param cohort_index:
    :param spark: SparkSession
    :param persistence_folder: persistence folder for the temp data frames

    :return:
    """
    if date_filter:
        patient_events = patient_events.where(F.col("date").cast("date") >= date_filter)

    if cohort_index is not None:
        if "cohort_member_id" in visit_occurrence.columns:
            joined_expr = ["person_id", "cohort_member_id"]
        else:
            joined_expr = ["person_id"]

        # There could be outpatient visits, where the visit_start_date occurs
        visit_start_datetime_udf = F.when(
            F.col("visit_start_datetime") > F.col("index_date"),
            F.expr(f"index_date - INTERVAL {1} DAY")
        ).otherwise(
            F.col("visit_start_datetime")
        )
        # Remove the visits that are not used in the patient events
        visit_occurrence = visit_occurrence.join(
            patient_events.select("visit_occurrence_id").distinct(),
            "visit_occurrence_id"
        ).join(
            cohort_index,
            joined_expr
        ).withColumn(
            "visit_start_datetime", visit_start_datetime_udf
        ).withColumn(
            "visit_start_date", F.to_date("visit_start_datetime")
        )

    decorators = [
        ClinicalEventDecorator(visit_occurrence, spark=spark, persistence_folder=persistence_folder),
        AttEventDecorator(
            visit_occurrence,
            include_visit_type,
            exclude_visit_tokens,
            att_type,
            inpatient_att_type,
            include_inpatient_hour_token,
            spark=spark,
            persistence_folder=persistence_folder
        ),
        DeathEventDecorator(death, att_type, spark=spark, persistence_folder=persistence_folder),
        # PredictionEventDecorator(cohort_index, spark=spark, persistence_folder=persistence_folder),
    ]

    if not exclude_demographic:
        decorators.append(
            DemographicEventDecorator(
                patient_demographic,
                use_age_group,
                spark=spark,
                persistence_folder=persistence_folder
            )
        )

    for decorator in decorators:
        patient_events = decorator.decorate(patient_events)

    # Make sure to ONLY keep the events before the index datetime if this is a prediction task
    if cohort_index is not None:
        patient_events = patient_events.join(
            cohort_index,
            ["person_id", "cohort_member_id"]
        ).where(
            (patient_events["datetime"] <= cohort_index["index_date"]) |
            (patient_events["standard_concept_id"] == "[END]")
        ).drop(
            "index_date"
        )

    # add randomness to the order of the concepts that have the same time stamp
    order_udf = F.row_number().over(
        W.partitionBy("cohort_member_id", "person_id").orderBy(
            "visit_rank_order",
            "concept_order",
            "priority",
            "datetime",
            "standard_concept_id",
        )
    )

    dense_rank_udf = F.dense_rank().over(
        W.partitionBy("cohort_member_id", "person_id").orderBy(
            "visit_rank_order", "concept_order", "priority", "datetime"
        )
    )

    # Those columns are derived from the previous decorators
    struct_columns = [
        "order",
        "record_rank",
        "date_in_week",
        "standard_concept_id",
        "visit_segment",
        "age",
        "visit_rank_order",
        "concept_value_mask",
        "number_as_value",
        "concept_as_value",
        "is_numeric_type",
        "mlm_skip_value",
        "visit_concept_id",
        "visit_concept_order",
        "concept_order",
        "priority",
        "unit",
    ]
    output_columns = [
        "cohort_member_id",
        "person_id",
        "concept_ids",
        "visit_segments",
        "orders",
        "dates",
        "ages",
        "visit_concept_orders",
        "num_of_visits",
        "num_of_concepts",
        "concept_value_masks",
        "number_as_values",
        "concept_as_values",
        "is_numeric_types",
        "mlm_skip_values",
        "priorities",
        "visit_concept_ids",
        "visit_rank_orders",
        "concept_orders",
        "record_ranks",
        "units",
    ]

    patient_grouped_events = (
        patient_events.withColumn("order", order_udf)
        .withColumn("record_rank", dense_rank_udf)
        .withColumn("data_for_sorting", F.struct(struct_columns))
        .groupBy("cohort_member_id", "person_id")
        .agg(
            F.sort_array(F.collect_set("data_for_sorting")).alias("data_for_sorting"),
            F.max("visit_rank_order").alias("num_of_visits"),
            F.count("standard_concept_id").alias("num_of_concepts"),
        )
        .withColumn("orders", F.col("data_for_sorting.order").cast(T.ArrayType(T.IntegerType())))
        .withColumn(
            "record_ranks",
            F.col("data_for_sorting.record_rank").cast(T.ArrayType(T.IntegerType())),
        )
        .withColumn("dates", F.col("data_for_sorting.date_in_week"))
        .withColumn("concept_ids", F.col("data_for_sorting.standard_concept_id"))
        .withColumn("visit_segments", F.col("data_for_sorting.visit_segment"))
        .withColumn("ages", F.col("data_for_sorting.age"))
        .withColumn("visit_rank_orders", F.col("data_for_sorting.visit_rank_order"))
        .withColumn("visit_concept_orders", F.col("data_for_sorting.visit_concept_order"))
        .withColumn("concept_orders", F.col("data_for_sorting.concept_order"))
        .withColumn("priorities", F.col("data_for_sorting.priority"))
        .withColumn("concept_value_masks", F.col("data_for_sorting.concept_value_mask"))
        .withColumn("number_as_values", F.col("data_for_sorting.number_as_value"))
        .withColumn("concept_as_values", F.col("data_for_sorting.concept_as_value"))
        .withColumn("is_numeric_types", F.col("data_for_sorting.is_numeric_type"))
        .withColumn("mlm_skip_values", F.col("data_for_sorting.mlm_skip_value"))
        .withColumn("visit_concept_ids", F.col("data_for_sorting.visit_concept_id"))
        .withColumn("units", F.col("data_for_sorting.unit"))
    )
    return patient_grouped_events.select(output_columns)


def create_concept_frequency_data(patient_event, date_filter=None):
    if date_filter:
        patient_event = patient_event.where(F.col("date") >= date_filter)

    take_concept_ids_udf = F.udf(lambda rows: [row[0] for row in rows], T.ArrayType(T.StringType()))
    take_freqs_udf = F.udf(lambda rows: [row[1] for row in rows], T.ArrayType(T.IntegerType()))

    num_of_visits_concepts = patient_event.groupBy("cohort_member_id", "person_id").agg(
        F.countDistinct("visit_occurrence_id").alias("num_of_visits"),
        F.count("standard_concept_id").alias("num_of_concepts"),
    )

    patient_event = (
        patient_event.groupBy("cohort_member_id", "person_id", "standard_concept_id")
        .count()
        .withColumn("concept_id_freq", F.struct("standard_concept_id", "count"))
        .groupBy("cohort_member_id", "person_id")
        .agg(F.collect_list("concept_id_freq").alias("sequence"))
        .withColumn("concept_ids", take_concept_ids_udf("sequence"))
        .withColumn("frequencies", take_freqs_udf("sequence"))
        .select("cohort_member_id", "person_id", "concept_ids", "frequencies")
        .join(num_of_visits_concepts, ["person_id", "cohort_member_id"])
    )

    return patient_event


def construct_artificial_visits(
        patient_events: DataFrame,
        visit_occurrence: DataFrame,
        spark: SparkSession = None,
        persistence_folder: str = None,
        duplicate_records: bool = False,
        disconnect_problem_list_records: bool = False,
) -> Tuple[DataFrame, DataFrame]:
    """
    Fix visit_occurrence_id of

    :param patient_events:
    :param visit_occurrence:
    :param spark:
    :param persistence_folder:
    :param duplicate_records:
    :param disconnect_problem_list_records:
    :return:
    """

    visit = visit_occurrence.select(
        F.col("person_id"),
        F.col("visit_occurrence_id"),
        F.col("visit_concept_id"),
        F.coalesce("visit_start_datetime", F.to_timestamp("visit_start_date")).alias("visit_start_datetime"),
        F.coalesce("visit_end_datetime", F.to_timestamp(F.date_add(F.col("visit_end_date"), 1))).alias(
            "visit_end_datetime"),
    ).withColumn(
        "visit_start_lower_bound", F.expr("visit_start_datetime - INTERVAL 1 DAYS")
    ).withColumn(
        "visit_end_upper_bound", F.expr("visit_end_datetime + INTERVAL 1 DAYS")
    )

    if disconnect_problem_list_records:
        # Set visit_occurrence_id to None if the event datetime is outside the visit start and visit end
        updated_patient_events = patient_events.join(
            visit.select("visit_occurrence_id", "visit_start_lower_bound", "visit_end_upper_bound"),
            "visit_occurrence_id",
            "left_outer"
        ).withColumn(
            "visit_occurrence_id",
            F.when(
                F.col("datetime").between(F.col("visit_start_lower_bound"), F.col("visit_end_upper_bound")),
                F.col("visit_occurrence_id")
            ).otherwise(
                F.lit(None).cast(T.IntegerType())
            )
        ).withColumn(
            "visit_concept_id",
            F.when(
                F.col("visit_occurrence_id").isNotNull(),
                F.col("visit_concept_id")
            ).otherwise(
                F.lit(0).cast(T.IntegerType())
            )
        ).drop(
            "visit_start_lower_bound", "visit_end_upper_bound"
        )

        if duplicate_records:
            patient_events = updated_patient_events.where(F.col("visit_occurrence_id").isNull()).unionByName(patient_events)
        else:
            patient_events = updated_patient_events

    # Try to connect to the existing visit
    events_to_fix = patient_events.where(
        F.col("visit_occurrence_id").isNull()
    ).withColumn(
        "record_id", F.monotonically_increasing_id()
    )

    if spark is not None and persistence_folder is not None:
        raw_events_dir = os.path.join(persistence_folder, "events_to_fix", "raw_events")
        events_to_fix.write.mode("overwrite").parquet(
            raw_events_dir
        )
        events_to_fix = spark.read.parquet(raw_events_dir)

    events_to_fix_with_visit = events_to_fix.drop("visit_occurrence_id").alias("event").join(
        visit.alias("visit"),
        (F.col("event.person_id") == F.col("visit.person_id"))
        & F.col("event.datetime").between(
            F.col("visit.visit_start_datetime").cast(T.DateType()).cast(T.TimestampType()),
            F.expr("visit.visit_end_datetime + INTERVAL 1 DAY - INTERVAL 1 SECOND")
        ),
        "left_outer"
    ).withColumn(
        "matching_rank",
        F.row_number().over(W.partitionBy("event.record_id").orderBy("visit.visit_start_datetime"))
    ).where(
        F.col("matching_rank") == 1
    ).select(
        [F.col("event." + _).alias(_) for _ in events_to_fix.schema.fieldNames()
         if _ not in ["visit_occurrence_id", "visit_concept_id"]] +
        [F.col("visit.visit_occurrence_id").alias("visit_occurrence_id"),
         F.col("visit.visit_concept_id").alias("visit_concept_id")]
    )

    linked_events = events_to_fix_with_visit.where(F.col("visit_occurrence_id").isNotNull())
    if spark is not None and persistence_folder is not None:
        linked_events_dir = os.path.join(persistence_folder, "events_to_fix", "linked_events")
        linked_events.write.mode("overwrite").parquet(
            linked_events_dir
        )
        linked_events = spark.read.parquet(linked_events_dir)

    events_artificial_visits = events_to_fix_with_visit.where(F.col("visit_occurrence_id").isNull())
    max_visit_id_value = visit.select(F.max("visit_occurrence_id")).collect()[0][0]
    # Generate the new visit_occurrence_id for (person_id, data) pairs
    new_visit_ids = events_artificial_visits.select(
        "person_id", "date"
    ).distinct().withColumn(
        "visit_occurrence_id", F.lit(max_visit_id_value) + F.rank().over(W.orderBy("person_id", "date"))
    )
    events_artificial_visits = events_artificial_visits.drop("visit_occurrence_id").join(
        new_visit_ids, ["person_id", "date"]
    )
    if spark is not None and persistence_folder is not None:
        events_artificial_visits_dir = os.path.join(persistence_folder, "events_to_fix", "events_artificial_visits")
        events_artificial_visits.write.mode("overwrite").parquet(
            events_artificial_visits_dir
        )
        events_artificial_visits = spark.read.parquet(events_artificial_visits_dir)

    artificial_visits_agg = events_artificial_visits.groupby(
        "visit_occurrence_id",
        "person_id"
    ).agg(
        F.min("datetime").alias("visit_start_datetime"),
        F.max("datetime").alias("visit_end_datetime")
    ).select(
        F.col("visit_occurrence_id"),
        F.col("person_id"),
        F.lit(0).alias("visit_concept_id"),
        F.to_date("visit_start_datetime").alias("visit_start_date"),
        F.col("visit_start_datetime"),
        F.to_date("visit_end_datetime").alias("visit_end_date"),
        F.col("visit_end_datetime")
    )
    existing_columns = artificial_visits_agg.columns
    additional_columns = [
        F.lit(None).cast(field.dataType).alias(field.name)
        for field in visit_occurrence.schema
        if field.name not in existing_columns
    ]
    artificial_visits = artificial_visits_agg.select(existing_columns + additional_columns)
    if spark is not None and persistence_folder is not None:
        artificial_visits_dir = os.path.join(persistence_folder, "events_to_fix", "artificial_visits")
        artificial_visits.write.mode("overwrite").parquet(
            artificial_visits_dir
        )
        artificial_visits = spark.read.parquet(artificial_visits_dir)

    refreshed_patient_events = patient_events.where(
        F.col("visit_occurrence_id").isNotNull()
    ).unionByName(
        linked_events.drop("record_id")
    ).unionByName(
        events_artificial_visits.drop("record_id")
    )

    visit_occurrence = visit_occurrence.unionByName(artificial_visits)

    return refreshed_patient_events, visit_occurrence


def extract_ehr_records(
        spark: SparkSession,
        input_folder: str,
        domain_table_list: List[str],
        include_visit_type: bool = False,
        with_diagnosis_rollup: bool = False,
        with_drug_rollup: bool = False,
        include_concept_list: bool = False,
        refresh_measurement: bool = False,
        aggregate_by_hour: bool = False,
        keep_orphan_records: bool = False,
):
    """
    Extract the ehr records for domain_table_list from input_folder.

    :param spark:
    :param input_folder:
    :param domain_table_list:
    :param include_visit_type: whether or not to include the visit type to the ehr records
    :param with_diagnosis_rollup: whether ot not to roll up the diagnosis concepts to the parent levels
    :param with_drug_rollup: whether ot not to roll up the drug concepts to the parent levels
    :param include_concept_list:
    :param refresh_measurement:
    :param aggregate_by_hour:
    :param keep_orphan_records:
    :return:
    """
    concept = preprocess_domain_table(spark, input_folder, CONCEPT)
    patient_ehr_records = None
    for domain_table_name in domain_table_list:
        domain_table = preprocess_domain_table(
            spark=spark,
            input_folder=input_folder,
            domain_table_name=domain_table_name,
            with_diagnosis_rollup=with_diagnosis_rollup,
            with_drug_rollup=with_drug_rollup
        )
        ehr_events = extract_events_by_domain(
            domain_table,
            spark=spark,
            concept=concept,
            aggregate_by_hour=aggregate_by_hour,
            refresh=refresh_measurement,
            persistence_folder=input_folder
        )
        if patient_ehr_records is None:
            patient_ehr_records = ehr_events
        else:
            patient_ehr_records = patient_ehr_records.unionByName(ehr_events)

    if include_concept_list and patient_ehr_records:
        # Filter out concepts
        qualified_concepts = preprocess_domain_table(spark, input_folder, QUALIFIED_CONCEPT_LIST_PATH).select(
            "standard_concept_id"
        )
        patient_ehr_records = patient_ehr_records.join(qualified_concepts, "standard_concept_id")

    if not keep_orphan_records:
        patient_ehr_records = patient_ehr_records.where(F.col("visit_occurrence_id").isNotNull()).distinct()
    person = preprocess_domain_table(spark, input_folder, PERSON)
    person = person.withColumn(
        "birth_datetime",
        F.coalesce(
            "birth_datetime",
            F.concat("year_of_birth", F.lit("-01-01")).cast("timestamp"),
        ),
    )
    patient_ehr_records = patient_ehr_records.join(person, "person_id").withColumn(
        "age",
        F.ceil(F.months_between(F.col("date"), F.col("birth_datetime")) / F.lit(12)),
    )
    if include_visit_type:
        visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
        patient_ehr_records = patient_ehr_records.join(
            visit_occurrence,
            "visit_occurrence_id"
        ).select(
            patient_ehr_records["person_id"],
            patient_ehr_records["standard_concept_id"],
            patient_ehr_records["date"],
            patient_ehr_records["datetime"],
            patient_ehr_records["visit_occurrence_id"],
            patient_ehr_records["domain"],
            patient_ehr_records["unit"],
            patient_ehr_records["number_as_value"],
            patient_ehr_records["concept_as_value"],
            patient_ehr_records["event_group_id"],
            visit_occurrence["visit_concept_id"],
            patient_ehr_records["age"],
        )
    return patient_ehr_records


def create_visit_person_join(person, visit_occurrence, include_incomplete_visit=True):
    """
    Create a new spark data frame based on person and visit_occurrence.

    :param person:
    :param visit_occurrence:
    :param include_incomplete_visit:
    :return:
    """

    # Create a pandas udf for generating the att token between two neighboring visits
    @pandas_udf("string")
    def pandas_udf_to_att(time_intervals: pd.Series) -> pd.Series:
        return time_intervals.apply(time_token_func)

    visit_rank_udf = F.row_number().over(
        W.partitionBy("person_id").orderBy("visit_start_date", "visit_end_date", "visit_occurrence_id")
    )
    visit_segment_udf = F.col("visit_rank_order") % F.lit(2) + 1
    visit_windowing = W.partitionBy("person_id").orderBy("visit_start_date", "visit_end_date", "visit_occurrence_id")
    # Check whehter or not the visit is either an inpatient visit or E-I visit
    is_inpatient_logic = F.col("visit_concept_id").isin([9201, 262]).cast("integer")
    # Construct the logic for readmission, which is defined as inpatient visit occurred within 30
    # days of the discharge
    readmission_logic = F.coalesce(
        (
                (F.col("time_interval") <= 30)
                & (F.col("visit_concept_id").isin([9201, 262]))
                & (F.col("prev_visit_concept_id").isin([9201, 262]))
        ).cast("integer"),
        F.lit(0),
    )

    # Create prolonged inpatient stay
    # For the incomplete visit, we set prolonged_length_stay_logic to 0
    prolonged_length_stay_logic = F.coalesce(
        (F.datediff("visit_end_date", "visit_start_date") >= 7).cast("integer"),
        F.lit(0),
    )

    visit_filter = "visit_start_date IS NOT NULL"
    if not include_incomplete_visit:
        visit_filter = f"{visit_filter} AND visit_end_date IS NOT NULL"

    # Select the subset of columns and create derived columns using the UDF or spark sql
    # functions. In addition, we allow visits where visit_end_date IS NOT NULL, indicating the
    # visit is still on-going
    visit_occurrence = (
        visit_occurrence.select(
            "visit_occurrence_id",
            "person_id",
            "visit_concept_id",
            "visit_start_date",
            "visit_end_date",
        )
        .where(visit_filter)
        .withColumn("visit_rank_order", visit_rank_udf)
        .withColumn("visit_segment", visit_segment_udf)
        .withColumn(
            "prev_visit_occurrence_id",
            F.lag("visit_occurrence_id").over(visit_windowing),
        )
        .withColumn("prev_visit_concept_id", F.lag("visit_concept_id").over(visit_windowing))
        .withColumn("prev_visit_start_date", F.lag("visit_start_date").over(visit_windowing))
        .withColumn("prev_visit_end_date", F.lag("visit_end_date").over(visit_windowing))
        .withColumn("time_interval", F.datediff("visit_start_date", "prev_visit_end_date"))
        .withColumn(
            "time_interval",
            F.when(F.col("time_interval") < 0, F.lit(0)).otherwise(F.col("time_interval")),
        )
        .withColumn("time_interval_att", pandas_udf_to_att("time_interval"))
        .withColumn("is_inpatient", is_inpatient_logic)
        .withColumn("is_readmission", readmission_logic)
    )

    visit_occurrence = visit_occurrence.withColumn("prolonged_stay", prolonged_length_stay_logic).select(
        "visit_occurrence_id",
        "visit_concept_id",
        "person_id",
        "prolonged_stay",
        "is_readmission",
        "is_inpatient",
        "time_interval_att",
        "visit_rank_order",
        "visit_start_date",
        "visit_segment",
    )
    # Assume the birthday to be the first day of the birth year if birth_datetime is missing
    person = person.select(
        "person_id",
        F.coalesce(
            "birth_datetime",
            F.concat("year_of_birth", F.lit("-01-01")).cast("timestamp"),
        ).alias("birth_datetime"),
    )
    return visit_occurrence.join(person, "person_id")


def clean_up_unit(dataframe: DataFrame) -> DataFrame:
    return dataframe.withColumn(
        "unit",
        F.regexp_replace(F.col("unit"), r"\{.*?\}", "")
    ).withColumn(
        "unit",
        F.regexp_replace(F.col("unit"), r"^/", "1/")
    )


def get_measurement_events(
        measurement: DataFrame,
        concept: DataFrame,
        aggregate_by_hour: bool = False,
        refresh: bool = False,
        spark: SparkSession = None,
        persistence_folder: str = None,
) -> DataFrame:
    """
    Extract medical events from the measurement table

    spark: :param
    measurement: :param
    concept:

    :return:
    """

    if persistence_folder and spark:
        measurement_events_data_path = os.path.join(persistence_folder, PROCESSED_MEASUREMENT)
        if os.path.exists(measurement_events_data_path) and not refresh:
            return preprocess_domain_table(spark, persistence_folder, PROCESSED_MEASUREMENT)

    # Register the tables in spark context
    concept.createOrReplaceTempView(CONCEPT)
    measurement.createOrReplaceTempView(MEASUREMENT)
    measurement_events = spark.sql(
        """
        SELECT DISTINCT
            m.person_id,
            m.measurement_concept_id AS standard_concept_id,
            CAST(m.measurement_date AS DATE) AS date,
            CAST(COALESCE(m.measurement_datetime, m.measurement_date) AS TIMESTAMP) AS datetime,
            m.visit_occurrence_id AS visit_occurrence_id,
            'measurement' AS domain,
            CAST(NULL AS STRING) AS event_group_id,
            m.value_as_number AS number_as_value,
            CAST(m.value_as_concept_id AS STRING) AS concept_as_value,
            COALESCE(c.concept_code, m.unit_source_value, 'N/A') AS unit
        FROM measurement AS m
        LEFT JOIN concept AS c
            ON m.unit_concept_id = c.concept_id
        """
    )
    numeric_events = measurement_events.where(F.col("number_as_value").isNotNull())
    numeric_events = clean_up_unit(numeric_events)
    non_numeric_events = measurement_events.where(F.col("number_as_value").isNull())

    if aggregate_by_hour:
        numeric_events = numeric_events.withColumn("lab_hour", F.hour("datetime"))
        numeric_events = numeric_events.groupby(
            "person_id", "visit_occurrence_id", "standard_concept_id", "unit", "date", "lab_hour"
        ).agg(
            F.min("datetime").alias("datetime"),
            F.avg("number_as_value").alias("number_as_value"),
        ).withColumn(
            "domain", F.lit("measurement").cast("string")
        ).withColumn(
            "concept_as_value", F.lit(None).cast("string")
        ).withColumn(
            "event_group_id", F.lit(None).cast("string")
        ).drop("lab_hour")

    measurement_events = numeric_events.unionByName(non_numeric_events)
    if spark and persistence_folder:
        measurement_events_data_path = os.path.join(persistence_folder, PROCESSED_MEASUREMENT)
        measurement_events.write.mode("overwrite").parquet(measurement_events_data_path)
        measurement_events = spark.read.parquet(measurement_events_data_path)
    return measurement_events


def get_observation_events(
        observation: DataFrame,
        concept: DataFrame,
        aggregate_by_hour: bool = False,
        refresh: bool = False,
        spark: SparkSession = None,
        persistence_folder: str = None,
):
    """
    Extract medical events from the observation table

    spark: :param
    measurement: :param
    required_measurement:
    concept:

    :return:
    """

    if spark and persistence_folder:
        observation_events_data_path = os.path.join(persistence_folder, PROCESSED_OBSERVATION)
        if os.path.exists(observation_events_data_path) and not refresh:
            return preprocess_domain_table(spark, persistence_folder, PROCESSED_OBSERVATION)

    # Register the tables in spark context
    concept.createOrReplaceTempView(CONCEPT)
    observation.createOrReplaceTempView(OBSERVATION)
    observation_events = spark.sql(
        """
        SELECT DISTINCT
            o.person_id,
            o.observation_concept_id AS standard_concept_id,
            CAST(o.observation_date AS DATE) AS date,
            CAST(COALESCE(o.observation_datetime, o.observation_date) AS TIMESTAMP) AS datetime,
            o.visit_occurrence_id AS visit_occurrence_id,
            'observation' AS domain,
            CAST(NULL AS STRING) AS event_group_id,
            o.value_as_number AS number_as_value,
            CAST(o.value_as_concept_id AS STRING) AS concept_as_value,
            COALESCE(c.concept_code, o.unit_source_value, 'N/A') AS unit
        FROM observation AS o
        LEFT JOIN concept AS c
            ON o.unit_concept_id = c.concept_id
    """
    )
    numeric_events = observation_events.where(F.col("number_as_value").isNotNull())
    numeric_events = clean_up_unit(numeric_events)
    non_numeric_events = observation_events.where(F.col("number_as_value").isNull())
    if aggregate_by_hour:
        numeric_events = numeric_events.withColumn("lab_hour", F.hour("datetime"))
        numeric_events = numeric_events.groupby(
            "person_id", "visit_occurrence_id", "standard_concept_id", "unit", "date", "lab_hour"
        ).agg(
            F.min("datetime").alias("datetime"),
            F.avg("number_as_value").alias("number_as_value"),
        ).withColumn(
            "domain", F.lit("observation").cast("string")
        ).withColumn(
            "concept_as_value", F.lit(None).cast("string")
        ).withColumn(
            "event_group_id", F.lit(None).cast("string")
        ).drop("lab_hour")

    observation_events = numeric_events.unionByName(non_numeric_events)
    if spark and persistence_folder:
        observation_events_data_path = os.path.join(persistence_folder, PROCESSED_OBSERVATION)
        observation_events.write.mode("overwrite").parquet(observation_events_data_path)
        observation_events = spark.read.parquet(observation_events_data_path)
    return observation_events


def get_device_events(
        device_exposure: DataFrame,
        concept: DataFrame,
        aggregate_by_hour: bool = False,
        refresh: bool = False,
        spark: SparkSession = None,
        persistence_folder: str = None,
) -> DataFrame:
    """
    Extract medical events from the measurement table

    spark: :param
    measurement: :param
    concept:

    :return:
    """

    if persistence_folder and spark:
        device_events_data_path = os.path.join(persistence_folder, PROCESSED_DEVICE)
        if os.path.exists(device_events_data_path) and not refresh:
            return preprocess_domain_table(spark, persistence_folder, PROCESSED_DEVICE)

    # Register the tables in spark context
    concept.createOrReplaceTempView(CONCEPT)
    device_exposure.createOrReplaceTempView(DEVICE_EXPOSURE)
    device_events = spark.sql(
        """
        SELECT DISTINCT
            d.person_id,
            d.device_concept_id AS standard_concept_id,
            CAST(d.device_exposure_start_date AS DATE) AS date,
            CAST(COALESCE(d.device_exposure_start_datetime, d.device_exposure_start_date) AS TIMESTAMP) AS datetime,
            d.visit_occurrence_id AS visit_occurrence_id,
            'device' AS domain,
            CAST(NULL AS STRING) AS event_group_id,
            d.quantity AS number_as_value,
            CAST(NULL AS STRING) AS concept_as_value,
            COALESCE(c.concept_code, d.unit_source_value, 'N/A') AS unit
        FROM device_exposure AS d
        LEFT JOIN concept AS c
            ON d.unit_concept_id = c.concept_id
        """
    )
    numeric_events = device_events.where(F.col("number_as_value").isNotNull())
    numeric_events = clean_up_unit(numeric_events)
    non_numeric_events = device_events.where(F.col("number_as_value").isNull())

    if aggregate_by_hour:
        numeric_events = numeric_events.withColumn("lab_hour", F.hour("datetime"))
        numeric_events = numeric_events.groupby(
            "person_id", "visit_occurrence_id", "standard_concept_id", "unit", "date", "lab_hour"
        ).agg(
            F.min("datetime").alias("datetime"),
            F.avg("number_as_value").alias("number_as_value"),
        ).withColumn(
            "domain", F.lit("device").cast("string")
        ).withColumn(
            "concept_as_value", F.lit(None).cast("string")
        ).withColumn(
            "event_group_id", F.lit(None).cast("string")
        ).drop("lab_hour")

    device_events = numeric_events.unionByName(non_numeric_events)
    if spark and persistence_folder:
        device_events_data_path = os.path.join(persistence_folder, PROCESSED_DEVICE)
        device_events.write.mode("overwrite").parquet(device_events_data_path)
        device_events = spark.read.parquet(device_events_data_path)
    return device_events


def get_mlm_skip_domains(spark, input_folder, mlm_skip_table_list):
    """
    Translate the domain_table_name to the domain name.

    :param spark:
    :param input_folder:
    :param mlm_skip_table_list:
    :return:
    """
    domain_tables = [
        preprocess_domain_table(spark, input_folder, domain_table_name) for domain_table_name in mlm_skip_table_list
    ]

    return list(map(get_domain_field, domain_tables))


def validate_table_names(domain_names):
    for domain_name in domain_names.split(" "):
        if domain_name not in CDM_TABLES:
            raise argparse.ArgumentTypeError(f"{domain_name} is an invalid CDM table name")
    return domain_names

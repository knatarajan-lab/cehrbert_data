import argparse
import logging
import os.path
from os import path
from typing import List, Tuple

import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import Window as W
from pyspark.sql.functions import broadcast
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql import DataFrame, SparkSession

from cehrbert_data.config.output_names import QUALIFIED_CONCEPT_LIST_PATH
from cehrbert_data.const.common import (
    MEASUREMENT,
    PROCESSED_MEASUREMENT,
    REQUIRED_MEASUREMENT,
    NUMERIC_MEASUREMENT_STATS,
    CATEGORICAL_MEASUREMENT,
    CDM_TABLES,
    PERSON,
    UNKNOWN_CONCEPT,
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


def create_file_path(input_folder: str, table_name: str):
    if input_folder[-1] == "/":
        file_path = input_folder + table_name
    else:
        file_path = input_folder + "/" + table_name
    return file_path


def join_domain_tables(domain_tables: List[DataFrame]) -> DataFrame:
    """Standardize the format of OMOP domain tables using a time frame.
    Keyword arguments:
    domain_tables -- the array containing the OMOP domain tables except visit_occurrence
        except measurement

    The output columns of the domain table is converted to the same standard format as the following
    (person_id, standard_concept_id, date, lower_bound, upper_bound, domain).
    In this case, co-occurrence is defined as those concept ids that have co-occurred
    within the same time window of a patient.
    """
    patient_event = None

    for domain_table in domain_tables:
        # extract the domain concept_id from the table fields. E.g. condition_concept_id from
        # condition_occurrence extract the domain start_date column extract the name of the table
        for (
                concept_id_field,
                date_field,
                datetime_field,
                table_domain_field
        ) in get_key_fields(domain_table):
            # Remove records that don't have a date or standard_concept_id
            filtered_domain_table = domain_table.where(F.col(date_field).isNotNull()).where(
                F.col(concept_id_field).isNotNull()
            )
            datetime_field_udf = F.to_timestamp(F.coalesce(datetime_field, date_field), "yyyy-MM-dd HH:mm:ss")
            filtered_domain_table = (
                filtered_domain_table.where(F.col(concept_id_field).cast("string") != "0")
                .withColumn("date", F.to_date(F.col(date_field)))
                .withColumn("datetime", datetime_field_udf)
            )

            filtered_domain_table = filtered_domain_table.select(
                filtered_domain_table["person_id"],
                filtered_domain_table[concept_id_field].alias("standard_concept_id"),
                filtered_domain_table["date"].cast("date"),
                filtered_domain_table["datetime"].cast(T.TimestampType()),
                filtered_domain_table["visit_occurrence_id"],
                F.lit(table_domain_field).alias("domain"),
                F.lit(None).cast("string").alias("event_group_id"),
                F.lit(None).cast("float").alias("number_as_value"),
                F.lit(None).cast("string").alias("concept_as_value"),
                F.col("unit") if domain_has_unit(filtered_domain_table) else F.lit(NA).alias("unit"),
            ).distinct()

            # Remove "Patient Died" from condition_occurrence
            if filtered_domain_table == "condition_occurrence":
                filtered_domain_table = filtered_domain_table.where("condition_concept_id != 4216643")

            if patient_event is None:
                patient_event = filtered_domain_table
            else:
                patient_event = patient_event.unionByName(filtered_domain_table)

    return patient_event


def preprocess_domain_table(
        spark,
        input_folder,
        domain_table_name,
        with_diagnosis_rollup=False,
        with_drug_rollup=True,
):
    domain_table = spark.read.parquet(create_file_path(input_folder, domain_table_name))

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
                and path.exists(create_file_path(input_folder, "concept"))
                and path.exists(create_file_path(input_folder, "concept_ancestor"))
        ):
            concept = spark.read.parquet(create_file_path(input_folder, "concept"))
            concept_ancestor = spark.read.parquet(create_file_path(input_folder, "concept_ancestor"))
            domain_table = roll_up_to_drug_ingredients(domain_table, concept, concept_ancestor)

    if with_diagnosis_rollup:
        if (
                domain_table_name == "condition_occurrence"
                and path.exists(create_file_path(input_folder, "concept"))
                and path.exists(create_file_path(input_folder, "concept_relationship"))
        ):
            concept = spark.read.parquet(create_file_path(input_folder, "concept"))
            concept_relationship = spark.read.parquet(create_file_path(input_folder, "concept_relationship"))
            domain_table = roll_up_diagnosis(domain_table, concept, concept_relationship)

        if (
                domain_table_name == "procedure_occurrence"
                and path.exists(create_file_path(input_folder, "concept"))
                and path.exists(create_file_path(input_folder, "concept_ancestor"))
        ):
            concept = spark.read.parquet(create_file_path(input_folder, "concept"))
            concept_ancestor = spark.read.parquet(create_file_path(input_folder, "concept_ancestor"))
            domain_table = roll_up_procedure(domain_table, concept, concept_ancestor)

    return domain_table


def roll_up_to_drug_ingredients(drug_exposure, concept, concept_ancestor):
    # lowercase the schema fields
    drug_exposure = drug_exposure.select([F.col(f_n).alias(f_n.lower()) for f_n in drug_exposure.schema.fieldNames()])

    drug_ingredient = (
        drug_exposure.select("drug_concept_id")
        .distinct()
        .join(concept_ancestor, F.col("drug_concept_id") == F.col("descendant_concept_id"))
        .join(concept, F.col("ancestor_concept_id") == F.col("concept_id"))
        .where(concept["concept_class_id"] == "Ingredient")
        .select(F.col("drug_concept_id"), F.col("concept_id").alias("ingredient_concept_id"))
    )

    drug_ingredient_fields = [
        F.coalesce(F.col("ingredient_concept_id"), F.col("drug_concept_id")).alias("drug_concept_id")
    ]
    drug_ingredient_fields.extend(
        [F.col(field_name) for field_name in drug_exposure.schema.fieldNames() if field_name != "drug_concept_id"]
    )

    drug_exposure = drug_exposure.join(drug_ingredient, "drug_concept_id", "left_outer").select(drug_ingredient_fields)

    return drug_exposure


def roll_up_diagnosis(condition_occurrence, concept, concept_relationship):
    list_3dig_code = [
        "3-char nonbill code",
        "3-dig nonbill code",
        "3-char billing code",
        "3-dig billing code",
        "3-dig billing E code",
        "3-dig billing V code",
        "3-dig nonbill E code",
        "3-dig nonbill V code",
    ]

    condition_occurrence = condition_occurrence.select(
        [F.col(f_n).alias(f_n.lower()) for f_n in condition_occurrence.schema.fieldNames()]
    )

    condition_icd = (
        condition_occurrence.select("condition_source_concept_id")
        .distinct()
        .join(concept, (F.col("condition_source_concept_id") == F.col("concept_id")))
        .where(concept["domain_id"] == "Condition")
        .where(concept["vocabulary_id"] != "SNOMED")
        .select(
            F.col("condition_source_concept_id"),
            F.col("vocabulary_id").alias("child_vocabulary_id"),
            F.col("concept_class_id").alias("child_concept_class_id"),
        )
    )

    condition_icd_hierarchy = (
        condition_icd.join(
            concept_relationship,
            F.col("condition_source_concept_id") == F.col("concept_id_1"),
        )
        .join(
            concept,
            (F.col("concept_id_2") == F.col("concept_id")) & (F.col("concept_class_id").isin(list_3dig_code)),
            how="left",
        )
        .select(
            F.col("condition_source_concept_id").alias("source_concept_id"),
            F.col("child_concept_class_id"),
            F.col("concept_id").alias("parent_concept_id"),
            F.col("concept_name").alias("parent_concept_name"),
            F.col("vocabulary_id").alias("parent_vocabulary_id"),
            F.col("concept_class_id").alias("parent_concept_class_id"),
        )
        .distinct()
    )

    condition_icd_hierarchy = condition_icd_hierarchy.withColumn(
        "ancestor_concept_id",
        F.when(
            F.col("child_concept_class_id").isin(list_3dig_code),
            F.col("source_concept_id"),
        ).otherwise(F.col("parent_concept_id")),
    ).dropna(subset="ancestor_concept_id")

    condition_occurrence_fields = [
        F.col(f_n).alias(f_n.lower())
        for f_n in condition_occurrence.schema.fieldNames()
        if f_n != "condition_source_concept_id"
    ]
    condition_occurrence_fields.append(
        F.coalesce(F.col("ancestor_concept_id"), F.col("condition_source_concept_id")).alias(
            "condition_source_concept_id"
        )
    )

    condition_occurrence = (
        condition_occurrence.join(
            condition_icd_hierarchy,
            condition_occurrence["condition_source_concept_id"] == condition_icd_hierarchy["source_concept_id"],
            how="left",
        )
        .select(condition_occurrence_fields)
        .withColumn("condition_concept_id", F.col("condition_source_concept_id"))
    )
    return condition_occurrence


def roll_up_procedure(procedure_occurrence, concept, concept_ancestor):
    def extract_parent_code(concept_code):
        return concept_code.split(".")[0]

    parent_code_udf = F.udf(extract_parent_code, T.StringType())

    procedure_code = (
        procedure_occurrence.select("procedure_source_concept_id")
        .distinct()
        .join(concept, F.col("procedure_source_concept_id") == F.col("concept_id"))
        .where(concept["domain_id"] == "Procedure")
        .select(
            F.col("procedure_source_concept_id").alias("source_concept_id"),
            F.col("vocabulary_id").alias("child_vocabulary_id"),
            F.col("concept_class_id").alias("child_concept_class_id"),
            F.col("concept_code").alias("child_concept_code"),
        )
    )

    # cpt code rollup
    cpt_code = procedure_code.where(F.col("child_vocabulary_id") == "CPT4")

    cpt_hierarchy = (
        cpt_code.join(
            concept_ancestor,
            cpt_code["source_concept_id"] == concept_ancestor["descendant_concept_id"],
        )
        .join(concept, concept_ancestor["ancestor_concept_id"] == concept["concept_id"])
        .where(concept["vocabulary_id"] == "CPT4")
        .select(
            F.col("source_concept_id"),
            F.col("child_concept_class_id"),
            F.col("ancestor_concept_id").alias("parent_concept_id"),
            F.col("min_levels_of_separation"),
            F.col("concept_class_id").alias("parent_concept_class_id"),
        )
    )

    cpt_hierarchy_level_1 = (
        cpt_hierarchy.where(F.col("min_levels_of_separation") == 1)
        .where(F.col("child_concept_class_id") == "CPT4")
        .where(F.col("parent_concept_class_id") == "CPT4 Hierarchy")
        .select(F.col("source_concept_id"), F.col("parent_concept_id"))
    )

    cpt_hierarchy_level_1 = cpt_hierarchy_level_1.join(
        concept_ancestor,
        (cpt_hierarchy_level_1["source_concept_id"] == concept_ancestor["descendant_concept_id"])
        & (concept_ancestor["min_levels_of_separation"] == 1),
        how="left",
    ).select(
        F.col("source_concept_id"),
        F.col("parent_concept_id"),
        F.col("ancestor_concept_id").alias("root_concept_id"),
    )

    cpt_hierarchy_level_1 = cpt_hierarchy_level_1.withColumn(
        "isroot",
        F.when(
            cpt_hierarchy_level_1["root_concept_id"] == 45889197,
            cpt_hierarchy_level_1["source_concept_id"],
        ).otherwise(cpt_hierarchy_level_1["parent_concept_id"]),
    ).select(F.col("source_concept_id"), F.col("isroot").alias("ancestor_concept_id"))

    cpt_hierarchy_level_0 = (
        cpt_hierarchy.groupby("source_concept_id")
        .max()
        .where(F.col("max(min_levels_of_separation)") == 0)
        .select(F.col("source_concept_id").alias("cpt_level_0_concept_id"))
    )

    cpt_hierarchy_level_0 = cpt_hierarchy.join(
        cpt_hierarchy_level_0,
        cpt_hierarchy["source_concept_id"] == cpt_hierarchy_level_0["cpt_level_0_concept_id"],
    ).select(
        F.col("source_concept_id"),
        F.col("parent_concept_id").alias("ancestor_concept_id"),
    )

    cpt_hierarchy_rollup_all = cpt_hierarchy_level_1.union(cpt_hierarchy_level_0).drop_duplicates()

    # ICD code rollup
    icd_list = ["ICD9CM", "ICD9Proc", "ICD10CM"]

    procedure_icd = procedure_code.where(F.col("vocabulary_id").isin(icd_list))

    procedure_icd = (
        procedure_icd.withColumn("parent_concept_code", parent_code_udf(F.col("child_concept_code")))
        .withColumnRenamed("procedure_source_concept_id", "source_concept_id")
        .withColumnRenamed("concept_name", "child_concept_name")
        .withColumnRenamed("vocabulary_id", "child_vocabulary_id")
        .withColumnRenamed("concept_code", "child_concept_code")
        .withColumnRenamed("concept_class_id", "child_concept_class_id")
    )

    procedure_icd_map = (
        procedure_icd.join(
            concept,
            (procedure_icd["parent_concept_code"] == concept["concept_code"])
            & (procedure_icd["child_vocabulary_id"] == concept["vocabulary_id"]),
            how="left",
        )
        .select("source_concept_id", F.col("concept_id").alias("ancestor_concept_id"))
        .distinct()
    )

    # ICD10PCS rollup
    procedure_10pcs = procedure_code.where(F.col("vocabulary_id") == "ICD10PCS")

    procedure_10pcs = (
        procedure_10pcs.withColumn("parent_concept_code", F.substring(F.col("child_concept_code"), 1, 3))
        .withColumnRenamed("procedure_source_concept_id", "source_concept_id")
        .withColumnRenamed("concept_name", "child_concept_name")
        .withColumnRenamed("vocabulary_id", "child_vocabulary_id")
        .withColumnRenamed("concept_code", "child_concept_code")
        .withColumnRenamed("concept_class_id", "child_concept_class_id")
    )

    procedure_10pcs_map = (
        procedure_10pcs.join(
            concept,
            (procedure_10pcs["parent_concept_code"] == concept["concept_code"])
            & (procedure_10pcs["child_vocabulary_id"] == concept["vocabulary_id"]),
            how="left",
        )
        .select("source_concept_id", F.col("concept_id").alias("ancestor_concept_id"))
        .distinct()
    )

    # HCPCS rollup --- keep the concept_id itself
    procedure_hcpcs = procedure_code.where(F.col("child_vocabulary_id") == "HCPCS")
    procedure_hcpcs_map = (
        procedure_hcpcs.withColumn("ancestor_concept_id", F.col("source_concept_id"))
        .select("source_concept_id", "ancestor_concept_id")
        .distinct()
    )

    procedure_hierarchy = (
        cpt_hierarchy_rollup_all.union(procedure_icd_map)
        .union(procedure_10pcs_map)
        .union(procedure_hcpcs_map)
        .distinct()
    )
    procedure_occurrence_fields = [
        F.col(f_n).alias(f_n.lower())
        for f_n in procedure_occurrence.schema.fieldNames()
        if f_n != "procedure_source_concept_id"
    ]
    procedure_occurrence_fields.append(
        F.coalesce(F.col("ancestor_concept_id"), F.col("procedure_source_concept_id")).alias(
            "procedure_source_concept_id"
        )
    )

    procedure_occurrence = (
        procedure_occurrence.join(
            procedure_hierarchy,
            procedure_occurrence["procedure_source_concept_id"] == procedure_hierarchy["source_concept_id"],
            how="left",
        )
        .select(procedure_occurrence_fields)
        .withColumn("procedure_concept_id", F.col("procedure_source_concept_id"))
    )
    return procedure_occurrence


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
        PredictionEventDecorator(cohort_index, spark=spark, persistence_folder=persistence_folder),
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
) -> Tuple[DataFrame, DataFrame]:
    """
    Fix visit_occurrence_id of

    :param patient_events:
    :param visit_occurrence:
    :param spark:
    :param persistence_folder:
    :return:
    """
    visit = visit_occurrence.select(
        F.col("person_id"),
        F.col("visit_occurrence_id"),
        F.col("visit_concept_id"),
        F.coalesce("visit_start_datetime", F.to_timestamp("visit_start_date")).alias("visit_start_datetime"),
        F.coalesce("visit_end_datetime", F.to_timestamp(F.date_add(F.col("visit_end_date"), 1))).alias("visit_end_datetime"),
    ).withColumn(
        "visit_start_lower_bound", F.expr("visit_start_datetime - INTERVAL 1 DAYS")
    ).withColumn(
        "visit_end_upper_bound", F.expr("visit_end_datetime + INTERVAL 1 DAYS")
    )

    # Set visit_occurrence_id to None if the event datetime is outside the visit start and visit end
    patient_events = patient_events.join(
        visit.select("visit_occurrence_id", "visit_start_lower_bound", "visit_end_upper_bound"),
        "visit_occurrence_id"
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
        & F.col("event.datetime").between(F.col("visit.visit_start_datetime"), F.col("visit.visit_end_datetime")),
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
        with_drug_rollup: bool = True,
        include_concept_list: bool = False,
        refresh_measurement: bool = False,
        aggregate_by_hour: bool = False,
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
    :return:
    """
    domain_tables = []
    for domain_table_name in domain_table_list:
        if domain_table_name != MEASUREMENT:
            domain_tables.append(
                preprocess_domain_table(
                    spark=spark,
                    input_folder=input_folder,
                    domain_table_name=domain_table_name,
                    with_diagnosis_rollup=with_diagnosis_rollup,
                    with_drug_rollup=with_drug_rollup
                )
            )
    patient_ehr_records = join_domain_tables(domain_tables)

    if include_concept_list and patient_ehr_records:
        # Filter out concepts
        qualified_concepts = preprocess_domain_table(spark, input_folder, QUALIFIED_CONCEPT_LIST_PATH).select(
            "standard_concept_id"
        )
        patient_ehr_records = patient_ehr_records.join(qualified_concepts, "standard_concept_id")

    # Process the measurement table if exists
    if MEASUREMENT in domain_table_list:
        processed_measurement = get_measurement_table(
            spark,
            input_folder,
            refresh=refresh_measurement,
            aggregate_by_hour=aggregate_by_hour,
        )
        if patient_ehr_records:
            # Union all measurement records together with other domain records
            patient_ehr_records = patient_ehr_records.unionByName(processed_measurement)
        else:
            patient_ehr_records = processed_measurement

    patient_ehr_records = patient_ehr_records.where("visit_occurrence_id IS NOT NULL").distinct()
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


def build_ancestry_table_for(spark, concept_ids):
    initial_query = """
    SELECT
        cr.concept_id_1 AS ancestor_concept_id,
        cr.concept_id_2 AS descendant_concept_id,
        1 AS distance
    FROM global_temp.concept_relationship AS cr
    WHERE cr.concept_id_1 in ({concept_ids}) AND cr.relationship_id = 'Subsumes'
    """

    recurring_query = """
    SELECT
        i.ancestor_concept_id AS ancestor_concept_id,
        cr.concept_id_2 AS descendant_concept_id,
        i.distance + 1 AS distance
    FROM global_temp.ancestry_table AS i
    JOIN global_temp.concept_relationship AS cr
        ON i.descendant_concept_id = cr.concept_id_1 AND cr.relationship_id = 'Subsumes'
    LEFT JOIN global_temp.ancestry_table AS i2
        ON cr.concept_id_2 = i2.descendant_concept_id
    WHERE i2.descendant_concept_id IS NULL
    """

    union_query = """
    SELECT
        *
    FROM global_temp.ancestry_table

    UNION

    SELECT
        *
    FROM global_temp.candidate
    """

    ancestry_table = spark.sql(initial_query.format(concept_ids=",".join([str(c) for c in concept_ids])))
    ancestry_table.createOrReplaceGlobalTempView("ancestry_table")

    candidate_set = spark.sql(recurring_query)
    candidate_set.createOrReplaceGlobalTempView("candidate")

    while candidate_set.count() != 0:
        spark.sql(union_query).createOrReplaceGlobalTempView("ancestry_table")
        candidate_set = spark.sql(recurring_query)
        candidate_set.createOrReplaceGlobalTempView("candidate")

    ancestry_table = spark.sql(
        """
    SELECT
        *
    FROM global_temp.ancestry_table
    """
    )

    spark.sql(
        """
    DROP VIEW global_temp.ancestry_table
    """
    )

    return ancestry_table


def get_descendant_concept_ids(spark, concept_ids):
    """
    Query concept_ancestor table to get all descendant_concept_ids for the given list of concept_ids.

    :param spark:
    :param concept_ids:
    :return:
    """
    sanitized_concept_ids = [int(c) for c in concept_ids]
    # Join the sanitized IDs into a string for the query
    concept_ids_str = ",".join(map(str, sanitized_concept_ids))
    # Construct and execute the SQL query using the sanitized string
    descendant_concept_ids = spark.sql(
        f"""
        SELECT DISTINCT
            c.*
        FROM global_temp.concept_ancestor AS ca
        JOIN global_temp.concept AS c
            ON ca.descendant_concept_id = c.concept_id
        WHERE ca.ancestor_concept_id IN ({concept_ids_str})
    """
    )
    return descendant_concept_ids


def get_standard_concept_ids(spark, concept_ids):
    standard_concept_ids = spark.sql(
        """
            SELECT DISTINCT
                c.*
            FROM global_temp.concept_relationship AS cr
            JOIN global_temp.concept AS c
                ON ca.concept_id_2 = c.concept_id AND cr.relationship_id = 'Maps to'
            WHERE ca.concept_id_1 IN ({concept_ids})
        """.format(
            concept_ids=",".join([str(c) for c in concept_ids])
        )
    )
    return standard_concept_ids


def get_table_column_refs(dataframe):
    return [dataframe[fieldName] for fieldName in dataframe.schema.fieldNames()]


def create_hierarchical_sequence_data(
        person,
        visit_occurrence,
        patient_events,
        date_filter=None,
        max_num_of_visits_per_person=None,
        include_incomplete_visit=True,
        allow_measurement_only=False,
):
    """
    This creates a hierarchical data frame for the hierarchical bert model.

    :param person:
    :param visit_occurrence:
    :param patient_events:
    :param date_filter:
    :param max_num_of_visits_per_person:
    :param include_incomplete_visit:
    :param allow_measurement_only:
    :return:
    """

    if date_filter:
        visit_occurrence = visit_occurrence.where(F.col("visit_start_date").cast("date") >= date_filter)

    # Construct visit information with the person demographic
    visit_occurrence_person = create_visit_person_join(person, visit_occurrence, include_incomplete_visit)

    # Retrieve all visit column references
    visit_column_refs = get_table_column_refs(visit_occurrence_person)

    # Construct the patient event column references
    pat_col_refs = [
        F.coalesce(patient_events["cohort_member_id"], visit_occurrence["person_id"]).alias("cohort_member_id"),
        F.coalesce(patient_events["standard_concept_id"], F.lit(UNKNOWN_CONCEPT)).alias("standard_concept_id"),
        F.coalesce(patient_events["date"], visit_occurrence["visit_start_date"]).alias("date"),
        F.coalesce(patient_events["domain"], F.lit("unknown")).alias("domain"),
        F.coalesce(patient_events["number_as_value"], F.lit(-1.0)).alias("number_as_value"),
    ]

    # Convert standard_concept_id to string type, this is needed for the tokenization
    # Calculate the age w.r.t to the event
    patient_events = (
        visit_occurrence_person.join(patient_events, "visit_occurrence_id", "left_outer")
        .select(visit_column_refs + pat_col_refs)
        .withColumn("standard_concept_id", F.col("standard_concept_id").cast("string"))
        .withColumn(
            "age",
            F.ceil(F.months_between(F.col("date"), F.col("birth_datetime")) / F.lit(12)),
        )
        .withColumn("concept_value_mask", (F.col("domain") == MEASUREMENT).cast("int"))
        .withColumn(
            "mlm_skip",
            (F.col("domain").isin([MEASUREMENT, CATEGORICAL_MEASUREMENT])).cast("int"),
        )
        .withColumn("condition_mask", (F.col("domain") == "condition").cast("int"))
    )

    if not allow_measurement_only:
        # We only allow persons that have a non measurement record in the dataset
        qualified_person_df = (
            patient_events.where(~F.col("domain").isin([MEASUREMENT, CATEGORICAL_MEASUREMENT]))
            .where(F.col("standard_concept_id") != UNKNOWN_CONCEPT)
            .select("person_id")
            .distinct()
        )

        patient_events = patient_events.join(qualified_person_df, "person_id")

    # Create the udf for calculating the weeks since the epoch time 1970-01-01
    weeks_since_epoch_udf = (F.unix_timestamp("date") / F.lit(24 * 60 * 60 * 7)).cast("int")

    # UDF for creating the concept orders within each visit
    visit_concept_order_udf = F.row_number().over(
        W.partitionBy("cohort_member_id", "person_id", "visit_occurrence_id").orderBy("date", "standard_concept_id")
    )

    patient_events = (
        patient_events.withColumn("date", F.col("date").cast("date"))
        .withColumn("date_in_week", weeks_since_epoch_udf)
        .withColumn("visit_concept_order", visit_concept_order_udf)
    )

    # Insert a CLS token at the beginning of each visit, this CLS token will be used as the visit
    # summary in pre-training / fine-tuning. We basically make a copy of the first concept of
    # each visit and change it to CLS, and set the concept order to 0 to make sure this is always
    # the first token of each visit
    insert_cls_tokens = (
        patient_events.where("visit_concept_order == 1")
        .withColumn("standard_concept_id", F.lit("CLS"))
        .withColumn("domain", F.lit("CLS"))
        .withColumn("visit_concept_order", F.lit(0))
        .withColumn("date", F.col("visit_start_date"))
        .withColumn("concept_value_mask", F.lit(0))
        .withColumn("number_as_value", F.lit(-1.0))
        .withColumn("mlm_skip", F.lit(1))
        .withColumn("condition_mask", F.lit(0))
    )

    # Declare a list of columns that need to be collected per each visit
    struct_columns = [
        "visit_concept_order",
        "standard_concept_id",
        "date_in_week",
        "age",
        "concept_value_mask",
        "number_as_value",
        "mlm_skip",
        "condition_mask",
    ]

    # Merge the first CLS tokens into patient sequence and collect events for each visit
    patent_visit_sequence = (
        patient_events.union(insert_cls_tokens)
        .withColumn("visit_struct_data", F.struct(struct_columns))
        .groupBy("cohort_member_id", "person_id", "visit_occurrence_id")
        .agg(
            F.sort_array(F.collect_set("visit_struct_data")).alias("visit_struct_data"),
            F.first("visit_start_date").alias("visit_start_date"),
            F.first("visit_rank_order").alias("visit_rank_order"),
            F.first("visit_concept_id").alias("visit_concept_id"),
            F.first("is_readmission").alias("is_readmission"),
            F.first("is_inpatient").alias("is_inpatient"),
            F.first("visit_segment").alias("visit_segment"),
            F.first("time_interval_att").alias("time_interval_att"),
            F.first("prolonged_stay").alias("prolonged_stay"),
            F.count("standard_concept_id").alias("num_of_concepts"),
        )
        .orderBy(["person_id", "visit_rank_order"])
    )

    patient_visit_sequence = (
        patent_visit_sequence.withColumn("visit_concept_orders", F.col("visit_struct_data.visit_concept_order"))
        .withColumn("visit_concept_ids", F.col("visit_struct_data.standard_concept_id"))
        .withColumn("visit_concept_dates", F.col("visit_struct_data.date_in_week"))
        .withColumn("visit_concept_ages", F.col("visit_struct_data.age"))
        .withColumn("concept_value_masks", F.col("visit_struct_data.concept_value_mask"))
        .withColumn("concept_values", F.col("visit_struct_data.concept_value"))
        .withColumn("mlm_skip_values", F.col("visit_struct_data.mlm_skip"))
        .withColumn("condition_masks", F.col("visit_struct_data.condition_mask"))
        .withColumn("visit_mask", F.lit(0))
        .drop("visit_struct_data")
    )

    visit_struct_data_columns = [
        "visit_rank_order",
        "visit_occurrence_id",
        "visit_start_date",
        "visit_concept_id",
        "prolonged_stay",
        "visit_mask",
        "visit_segment",
        "num_of_concepts",
        "is_readmission",
        "is_inpatient",
        "time_interval_att",
        "visit_concept_orders",
        "visit_concept_ids",
        "visit_concept_dates",
        "visit_concept_ages",
        "concept_values",
        "concept_value_masks",
        "mlm_skip_values",
        "condition_masks",
    ]

    visit_weeks_since_epoch_udf = (
            F.unix_timestamp(F.col("visit_start_date").cast("date")) / F.lit(24 * 60 * 60 * 7)
    ).cast("int")

    patient_sequence = (
        patient_visit_sequence.withColumn("visit_start_date", visit_weeks_since_epoch_udf)
        .withColumn(
            "visit_struct_data",
            F.struct(visit_struct_data_columns).alias("visit_struct_data"),
        )
        .groupBy("cohort_member_id", "person_id")
        .agg(
            F.sort_array(F.collect_list("visit_struct_data")).alias("patient_list"),
            F.sum(F.lit(1) - F.col("visit_mask")).alias("num_of_visits"),
            F.sum("num_of_concepts").alias("num_of_concepts"),
        )
    )

    if max_num_of_visits_per_person:
        patient_sequence = patient_sequence.where(F.col("num_of_visits") <= max_num_of_visits_per_person)

    patient_sequence = (
        patient_sequence.withColumn("visit_rank_orders", F.col("patient_list.visit_rank_order"))
        .withColumn("concept_orders", F.col("patient_list.visit_concept_orders"))
        .withColumn("concept_ids", F.col("patient_list.visit_concept_ids"))
        .withColumn("dates", F.col("patient_list.visit_concept_dates"))
        .withColumn("ages", F.col("patient_list.visit_concept_ages"))
        .withColumn("visit_dates", F.col("patient_list.visit_start_date"))
        .withColumn("visit_segments", F.col("patient_list.visit_segment"))
        .withColumn("visit_masks", F.col("patient_list.visit_mask"))
        .withColumn(
            "visit_concept_ids",
            F.col("patient_list.visit_concept_id").cast(T.ArrayType(T.StringType())),
        )
        .withColumn("time_interval_atts", F.col("patient_list.time_interval_att"))
        .withColumn("concept_values", F.col("patient_list.concept_values"))
        .withColumn("concept_value_masks", F.col("patient_list.concept_value_masks"))
        .withColumn("mlm_skip_values", F.col("patient_list.mlm_skip_values"))
        .withColumn("condition_masks", F.col("patient_list.condition_masks"))
        .withColumn(
            "is_readmissions",
            F.col("patient_list.is_readmission").cast(T.ArrayType(T.IntegerType())),
        )
        .withColumn(
            "is_inpatients",
            F.col("patient_list.is_inpatient").cast(T.ArrayType(T.IntegerType())),
        )
        .withColumn(
            "visit_prolonged_stays",
            F.col("patient_list.prolonged_stay").cast(T.ArrayType(T.IntegerType())),
        )
        .drop("patient_list")
    )

    return patient_sequence


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


def get_measurement_table(
        spark: SparkSession,
        input_folder: str,
        refresh: bool = False,
        aggregate_by_hour: bool = False,
) -> DataFrame:
    """
    A helper function to process and create the measurement table

    :param spark:
    :param input_folder:
    :param refresh:
    :param aggregate_by_hour:

    :return:
    """

    measurement = preprocess_domain_table(spark, input_folder, MEASUREMENT)
    # The required_measurement table must exist
    if not os.path.exists(os.path.join(input_folder, REQUIRED_MEASUREMENT)):
        raise RuntimeError(f"{REQUIRED_MEASUREMENT} needs to be provided when measurement is included!")
    required_measurement = preprocess_domain_table(spark, input_folder, REQUIRED_MEASUREMENT)
    # The measurement_stats table must exist
    if not os.path.exists(os.path.join(input_folder, NUMERIC_MEASUREMENT_STATS)):
        raise RuntimeError(f"{NUMERIC_MEASUREMENT_STATS} needs to be provided when measurement is included!")
    measurement_stats = preprocess_domain_table(spark, input_folder, NUMERIC_MEASUREMENT_STATS)
    # The concept table must exist
    if not os.path.exists(os.path.join(input_folder, CONCEPT)):
        raise RuntimeError(f"{CONCEPT} needs to be provided when measurement is included!")
    concept = preprocess_domain_table(spark, input_folder, CONCEPT)
    # The select is necessary to make sure the order of the columns is the same as the
    # original dataframe, otherwise the union might use the wrong columns
    if os.path.exists(os.path.join(input_folder, PROCESSED_MEASUREMENT)) and not refresh:
        processed_measurement = preprocess_domain_table(spark, input_folder, PROCESSED_MEASUREMENT)
    else:
        processed_measurement = process_measurement(
            spark, measurement, required_measurement, measurement_stats, concept, aggregate_by_hour
        )
        processed_measurement.write.mode("overwrite").parquet(os.path.join(input_folder, PROCESSED_MEASUREMENT))

    return processed_measurement


def process_measurement(
        spark,
        measurement: DataFrame,
        required_measurement: DataFrame,
        measurement_stats: DataFrame,
        concept: DataFrame,
        aggregate_by_hour: bool = False
):
    """
    Preprocess the measurement table and only include the measurements whose measurement_concept_ids are specified
    in required_measurement. Add the standard unit as an additional column

    spark: :param
    measurement: :param
    required_measurement:
    concept:

    :return:
    """
    # Register the tables in spark context
    concept.createOrReplaceTempView(CONCEPT)
    measurement.createOrReplaceTempView(MEASUREMENT)
    required_measurement.createOrReplaceTempView(REQUIRED_MEASUREMENT)
    # Broadcast df to local executors
    broadcast(measurement_stats)
    # Create the temp view for this dataframe
    measurement_stats.createOrReplaceTempView("measurement_unit_stats")
    numeric_lab = spark.sql(
        """
        SELECT
            m.person_id,
            m.measurement_concept_id AS standard_concept_id,
            CAST(m.measurement_date AS DATE) AS date,
            CAST(COALESCE(m.measurement_datetime, m.measurement_date) AS TIMESTAMP) AS datetime,
            m.visit_occurrence_id AS visit_occurrence_id,
            'measurement' AS domain,
            CAST(NULL AS STRING) AS event_group_id,
            m.value_as_number AS number_as_value,
            CAST(NULL AS STRING) AS concept_as_value,
            c.concept_code AS unit
        FROM measurement AS m
        JOIN measurement_unit_stats AS s
            ON s.measurement_concept_id = m.measurement_concept_id AND s.unit_concept_id = m.unit_concept_id
        JOIN concept AS c
            ON m.unit_concept_id = c.concept_id
        WHERE m.visit_occurrence_id IS NOT NULL
            AND m.value_as_number IS NOT NULL
            AND m.value_as_number BETWEEN s.lower_bound AND s.upper_bound
    """
    )
    if aggregate_by_hour:
        numeric_lab = numeric_lab.withColumn("lab_hour", F.hour("datetime"))
        numeric_lab = numeric_lab.groupby(
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

    numeric_lab = clean_up_unit(numeric_lab)
    # For categorical measurements in required_measurement, we concatenate measurement_concept_id
    # with value_as_concept_id to construct a new standard_concept_id
    categorical_lab = spark.sql(
        """
        SELECT
            m.person_id,
            measurement_concept_id AS standard_concept_id,
            CAST(m.measurement_date AS DATE) AS date,
            CAST(COALESCE(m.measurement_datetime, m.measurement_date) AS TIMESTAMP) AS datetime,
            m.visit_occurrence_id AS visit_occurrence_id,
            'categorical_measurement' AS domain,
            CONCAT('mea-', CAST(m.measurement_id AS STRING)) AS event_group_id,
            CAST(NULL AS FLOAT) AS number_as_value,
            CAST(COALESCE(value_as_concept_id, 0) AS STRING) AS concept_as_value,
            'N/A' AS unit
        FROM measurement AS m
        WHERE EXISTS (
            SELECT
                1
            FROM required_measurement AS r
            WHERE r.measurement_concept_id = m.measurement_concept_id
            AND r.is_numeric = false
        )
    """
    )
    return numeric_lab.unionByName(categorical_lab)


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

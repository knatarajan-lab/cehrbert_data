import os
import logging
import argparse
import shutil

from cehrbert_data.utils.logging_utils import add_console_logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import types as t
from pyspark.sql import functions as f
from pyspark.sql.window import Window

# Enable logging
add_console_logging()
logger = logging.getLogger(__name__)

VOCABULARY_TABLES = ["concept", "concept_relationship", "concept_ancestor"]

visit_occurrence_mapping = {
    "patient_id": "person_id",
    "start": "visit_start_datetime",
    "end": "visit_end_datetime",
    "code": "visit_source_value",
    "visit_id": "visit_occurrence_id"
}

condition_occurrence_mapping = {
    "patient_id": "person_id",
    "start": "condition_start_datetime",
    "end": "condition_end_datetime",
    "code": "condition_source_value",
    "visit_id": "visit_occurrence_id"
}

procedure_occurrence_mapping = {
    "patient_id": "person_id",
    "start": "procedure_datetime",
    "end": "procedure_end_datetime",
    "code": "procedure_source_value",
    "visit_id": "visit_occurrence_id"
}

drug_exposure_mapping = {
    "patient_id": "person_id",
    "start": "drug_exposure_start_datetime",
    "end": "drug_exposure_end_datetime",
    "code": "drug_source_value",
    "visit_id": "visit_occurrence_id"
}

measurement_mapping = {
    "patient_id": "person_id",
    "start": "measurement_datetime",
    "code": "measurement_source_value",
    "visit_id": "visit_occurrence_id",
}

observation_mapping = {
    "patient_id": "person_id",
    "start": "observation_datetime",
    "code": "observation_source_value",
    "visit_id": "visit_occurrence_id",
}

death_mapping = {
    "patient_id": "person_id",
    "start": "death_datetime",
    "code": "death_source_value",
}

table_mapping = {
    "visit_occurrence": visit_occurrence_mapping,
    "condition_occurrence": condition_occurrence_mapping,
    "procedure_occurrence": procedure_occurrence_mapping,
    "drug_exposure": drug_exposure_mapping,
    "measurement": measurement_mapping,
    "observation": observation_mapping,
    "death": death_mapping
}

concept_id_mapping = {
    "visit_occurrence": "visit_concept_id",
    "condition_occurrence": "condition_concept_id",
    "procedure_occurrence": "procedure_concept_id",
    "drug_exposure": "drug_concept_id",
    "measurement": "measurement_concept_id",
    "observation": "observation_concept_id",
    "death": "death_type_concept_id"
}


def get_schema() -> t.StructType:
    # Define the modified schema
    return t.StructType([
        t.StructField("_c0", t.StringType(), True),
        t.StructField("patient_id", t.IntegerType(), True),
        t.StructField("start", t.TimestampType(), True),  # Converted to TimestampType
        t.StructField("end", t.TimestampType(), True),  # Converted to TimestampType
        t.StructField("code", t.StringType(), True),
        t.StructField("value", t.StringType(), True),
        t.StructField("unit", t.StringType(), True),
        t.StructField("visit_id", t.StringType(), True),  # Converted to IntegerType
        t.StructField("omop_table", t.StringType(), True)
    ])


def create_omop_person(
        ehr_shot_data: DataFrame,
        concept: DataFrame,
) -> DataFrame:
    """
    Transforms EHR data into the OMOP-compliant `person` table format.

    This function extracts and transforms specific attributes from the input
    `ehr_shot_data` to match the OMOP `person` table schema, incorporating
    concepts for `birth_datetime`, `gender`, `ethnicity`, and `race`. The function
    filters for relevant information and performs necessary joins to return a unified
    `person` DataFrame in OMOP format.

    Parameters
    ----------
    ehr_shot_data : DataFrame
        Source DataFrame containing raw EHR data, with columns that include `omop_table`,
        `code`, `start`, and `patient_id`, from which the function filters records for
        the `person` table.

    concept : DataFrame
        DataFrame containing mappings of codes to OMOP concepts, used by helper functions
        to convert source-specific codes (for gender, ethnicity, and race) into OMOP-compatible
        concept IDs.

    Returns
    -------
    DataFrame
        OMOP-compliant `person` DataFrame with the following columns:
        - `person_id`: Unique identifier for each individual.
        - `birth_datetime`: Date and time of birth.
        - `year_of_birth`, `month_of_birth`, `day_of_birth`: Year, month, and day extracted from `birth_datetime`.
        - `gender_concept_id`, `gender_source_value`: OMOP concept ID and original code for gender.
        - `ethnicity_concept_id`, `ethnicity_source_value`: OMOP concept ID and original code for ethnicity.
        - `race_concept_id`, `race_source_value`: OMOP concept ID and original code for race.

    Notes
    -----
    - `birth_datetime` is identified by the code `SNOMED/3950001` in the `ehr_shot_data`.
    - `gender`, `ethnicity`, and `race` fields are filtered based on code prefixes
      "Gender", "Ethnicity", and "Race" respectively.
    - Joins are performed to ensure all demographic attributes are associated
      with each `person_id` from `birth_datetime`, with left outer joins to retain
      records even when some fields are missing.

    """
    omop_person = ehr_shot_data.where(f.col("omop_table") == "person")
    birth_datetime = omop_person.where(f.col("code") == "SNOMED/3950001").select(
        f.col("patient_id").alias("person_id"),
        f.col("start").alias("birth_datetime"),
        f.year("start").alias("year_of_birth"),
        f.month("start").alias("month_of_birth"),
        f.dayofmonth("start").alias("day_of_birth"),
    )
    gender = convert_code_to_omop_concept(
        omop_person.where(f.col("code").startswith("Gender")),
        concept,
        "code"
    ).select(
        f.col("patient_id").alias("person_id"),
        f.col("concept_id").cast(t.IntegerType()).alias("gender_concept_id"),
        f.col("code").alias("gender_source_value"),
    )
    ethnicity = convert_code_to_omop_concept(
        omop_person.where(f.col("code").startswith("Ethnicity")),
        concept,
        "code"
    ).select(
        f.col("patient_id").alias("person_id"),
        f.col("concept_id").cast(t.IntegerType()).alias("ethnicity_concept_id"),
        f.col("code").alias("ethnicity_source_value"),
    )
    race = convert_code_to_omop_concept(
        omop_person.where(f.col("code").startswith("Race")),
        concept,
        "code"
    ).select(
        f.col("patient_id").alias("person_id"),
        f.col("concept_id").cast(t.IntegerType()).alias("race_concept_id"),
        f.col("code").alias("race_source_value"),
    )
    return birth_datetime.join(
        gender, "person_id", "left_outer"
    ).join(
        ethnicity, "person_id", "left_outer"
    ).join(
        race, "person_id", "left_outer"
    )


def map_unit(
        data: DataFrame,
        concept: DataFrame
) -> DataFrame:
    """
    Maps unit values in a DataFrame to OMOP unit concept IDs.

    This function processes distinct `unit` entries in the `data` DataFrame, joining them with
    the `concept` DataFrame to find corresponding OMOP concept IDs where `domain_id` is `"Unit"`.
    It adds a `unit_concept_id` column with the mapped concept ID or assigns a default of `0` when
    no match is found. Only the first matched concept ID is retained for each unique unit value.

    Parameters
    ----------
    data : DataFrame
        Source DataFrame containing a `unit` column with units to be mapped to OMOP concept IDs.

    concept : DataFrame
        DataFrame containing OMOP concepts with columns `domain_id`, `concept_code`, and `concept_id`,
        used for mapping units in `data` to OMOP unit concept IDs.

    Returns
    -------
    DataFrame
        The input `data` DataFrame with an additional `unit_concept_id` column representing the
        mapped OMOP concept ID for each `unit`. If no matching concept is found, `unit_concept_id`
        is set to `0`.
    """
    # Find the unit mapping from the concept table
    unit_df = data.select("unit").distinct().join(
        concept.where(f.col("domain_id") == "Unit"),
        on=data["unit"] == concept["concept_code"],
        how="left_outer"
    ).select(
        data["unit"],
        f.coalesce(concept["concept_id"], f.lit(0)).alias("unit_concept_id")
    )
    unit_df = unit_df.withColumn(
        "order",
        f.row_number().over(Window.partitionBy(f.col("unit")).orderBy(f.col("unit_concept_id")))
    ).where(f.col("order") == 1).drop("order")
    return data.join(unit_df, "unit", "left_outer")


def map_answer(
        data: DataFrame,
        concept: DataFrame
) -> DataFrame:
    """
    Maps categorical answer values in a DataFrame to OMOP concept IDs.

    This function processes distinct `value` entries in the `data` DataFrame, joining them with
    the `concept` DataFrame to find OMOP concept IDs where `domain_id` is `"Meas Value"`. It
    adds a `value_as_concept_id` column with the mapped concept ID or assigns a default of `0`
    when no match is found. Only the first matched concept ID is retained for each unique value.

    Parameters
    ----------
    data : DataFrame
        Source DataFrame containing the `value` column, which holds categorical answer values
        that will be mapped to OMOP concept IDs.

    concept : DataFrame
        DataFrame containing OMOP concepts, including columns `domain_id`, `concept_name`,
        and `concept_id`, used for matching answer values to OMOP concept IDs.

    Returns
    -------
    DataFrame
        The input `data` DataFrame with an additional `value_as_concept_id` column, representing
        the mapped OMOP concept ID for each `value`. If no matching concept is found, `value_as_concept_id`
        is set to `0`.
    """
    answer_df = data.select("value").distinct().join(
        concept.where(f.col("domain_id") == "Meas Value"),
        data["value"] == concept["concept_name"],
        how="left_outer"
    ).select(
        data["value"],
        f.coalesce(concept["concept_id"], f.lit(0)).alias("value_as_concept_id")
    )

    answer_df = answer_df.withColumn(
        "order",
        f.row_number().over(Window.partitionBy(f.col("value")).orderBy(f.col("value_as_concept_id")))
    ).where(f.col("order") == 1).drop(
        "order"
    )
    return data.join(answer_df, "value", "left_outer")


def extract_value(
        data: DataFrame,
        concept: DataFrame
):
    """
    Transforms and maps values in a DataFrame to OMOP-compatible numeric, categorical, or unclassified representations.

    This function processes the `value` column in the `data` DataFrame to categorize each entry as numeric,
    categorical, or unmatched. Numeric values are cast to floats and joined with OMOP units in `concept`.
    Categorical values are matched to OMOP measurement values in `concept`, while unmatched values are returned
    as null mappings. The output includes mapped `unit_concept_id` and `value_as_concept_id` columns, as well as
    renamed source columns for OMOP compatibility.

    Parameters
    ----------
    data : DataFrame
        Source DataFrame containing the columns `value` (mixed numeric or categorical entries) and `unit`
        (for numeric values). Both columns are used for mapping to OMOP concepts.

    concept : DataFrame
        DataFrame containing OMOP concepts, used to map numeric units and categorical values with columns
        `domain_id`, `concept_name`, `concept_code`, and `concept_id`.

    Returns
    -------
    DataFrame
        A DataFrame where:
        - `value_source_value`: Original `value` column renamed for OMOP compatibility.
        - `unit_source_value`: Original `unit` column renamed for OMOP compatibility.
        - `value_as_number`: Contains numeric values where applicable, otherwise null.
        - `value_as_concept_id`: Concept ID for categorical values, null for numeric.
        - `unit_concept_id`: Concept ID for units associated with numeric values, null for categorical values.
    """
    numeric_pattern = "^[+-]?\\d*\\.?\\d+$"
    # Add a new column 'is_numeric' to check if 'value' is numeric
    df = data.withColumn(
        "is_numeric",
        f.regexp_extract(f.col("value"), numeric_pattern, 0) != ""
    )

    numeric_df = map_unit(
        df.where(
            f.col("is_numeric")
        ).withColumn(
            "value_as_number",
            f.col("value").cast(t.FloatType())
        ).withColumn(
            "value_as_concept_id", f.lit(None).cast(t.IntegerType())
        ),
        concept
    )

    categorical_df = map_answer(
        df.where(
            ~f.col("is_numeric")
        ).withColumn(
            "unit_concept_id", f.lit(None).cast(t.IntegerType())
        ).withColumn(
            "value_as_number", f.lit(None).cast(t.FloatType())
        ),
        concept
    )

    other_df = df.where(f.col("is_numeric").isNull()).withColumn(
        "unit_concept_id", f.lit(None).cast(t.IntegerType())
    ).withColumn(
        "value_as_number", f.lit(None).cast(t.FloatType())
    ).withColumn(
        "value_as_concept_id", f.lit(None).cast(t.IntegerType())
    )

    return numeric_df.unionByName(categorical_df).unionByName(other_df).withColumnRenamed(
        "value", "value_source_value"
    ).withColumnRenamed(
        "unit", "unit_source_value"
    ).drop("is_numeric")


def convert_code_to_omop_concept(
        data: DataFrame,
        concept: DataFrame,
        field: str
) -> DataFrame:
    """
    Maps source-specific codes in a DataFrame to OMOP concept IDs based on vocabulary and concept codes.

    This function extracts `vocabulary_id` and `concept_code` from the specified `field` column in the
    `data` DataFrame by splitting it at the "/" character. It then performs a left join with the `concept`
    DataFrame on both `vocabulary_id` and `concept_code` to retrieve the corresponding OMOP `concept_id`.
    If a match is not found, `concept_id` defaults to 0.

    Parameters
    ----------
    data : DataFrame
        The source DataFrame containing at least the specified `field` column to be mapped, which includes
        vocabulary and concept codes separated by "/".

    concept : DataFrame
        A DataFrame containing `vocabulary_id`, `concept_code`, and `concept_id` columns that provide
        mappings to OMOP concept IDs.

    field : str
        The name of the column in `data` that contains the vocabulary and concept codes to be split
        and mapped.

    Returns
    -------
    DataFrame
        A DataFrame with all columns from `data`, along with an added `concept_id` column mapped from `concept`.
        If no mapping is found in the `concept` DataFrame, `concept_id` is set to 0.

    Example
    -------
    Assuming `data` contains a column `code` with values such as "ICD10/1234" and `concept` contains rows
    mapping `ICD10` vocabulary and `1234` code to a specific OMOP concept ID, this function will add
    `concept_id` to `data` where values are mapped as per the `concept` DataFrame.
    """
    output_columns = [data[_] for _ in data.schema.fieldNames()] + [
        f.coalesce(concept["concept_id"], f.lit(0)).alias("concept_id")
    ]
    data = data.withColumn(
        "vocabulary_id",
        f.split(field, "/")[0]
    ).withColumn(
        "concept_code",
        f.split(field, "/")[1]
    )
    return data.join(
        concept,
        on=(data["vocabulary_id"] == concept["vocabulary_id"]) & (data["concept_code"] == concept["concept_code"]),
        how="left_outer",
    ).select(output_columns)


def generate_visit_id(
        data: DataFrame,
        spark: SparkSession,
        cache_folder: str,
        day_cutoff: int = 1
) -> DataFrame:
    """
     Generates unique `visit_id`s for each visit based on distinct patient event records.

     This function identifies records associated with actual visits (`visit_occurrence` table) and assigns
     `visit_id`s to those records. For other event records without a `visit_id`, it attempts to link them to
     existing visits based on overlapping date ranges. If no matching visit is found, it generates new `visit_id`s
     for these orphan records and creates artificial visits.

     Parameters
     ----------
     data : DataFrame
         A PySpark DataFrame containing at least the following columns:
         - `patient_id`: Identifier for each patient.
         - `start`: Start timestamp of the event.
         - `end`: (Optional) End timestamp of the event.
         - `omop_table`: String specifying the type of event (e.g., "visit_occurrence" for real visits).
         - `visit_id`: (Optional) Identifier for visits. May be missing in some records.
     spark: SparkSession
        The current spark session
     cache_folder: str
        The cache folder for saving the intermediate dataframes
     day_cutoff: int
        Day cutoff to disconnect the records with their associated visit

     Returns
     -------
     DataFrame
         A DataFrame with a `visit_id` assigned to each event record, including both real visits and artificial visits.
         The returned DataFrame includes both the original records and any generated artificial visit records,
         with each record grouped according to the identified visit.

     Steps
     -----
     1. **Identify Real Visits**: Filters out records from `visit_occurrence` and sets start and end dates for each visit.
     2. **Assign `visit_id`s to Other Records**: Attempts to link non-visit records (from other tables) to real visits
        based on matching `patient_id` and date ranges.
     3. **Handle Orphan Records**: For records without a matching visit, assigns new `visit_id`s by grouping
        records by patient and start date.
     4. **Create Artificial Visits**: Generates artificial visit records for orphan `visit_id`s.
     5. **Merge and Validate**: Combines the original records with artificial visits and validates the uniqueness of each `visit_id`.
     """
    visit_reconstruction_folder = os.path.join(cache_folder, "visit_reconstruction")
    data = data.repartition(16)
    real_visits = data.where(
        f.col("omop_table") == "visit_occurrence"
    ).withColumn(
        "visit_start_date",
        f.col("start").cast(t.DateType())
    ).withColumn(
        "visit_end_date",
        f.coalesce(f.col("end").cast(t.DateType()), f.col("visit_start_date"))
    )
    real_visits_folder = os.path.join(visit_reconstruction_folder, "real_visits")
    real_visits.write.mode("overwrite").parquet(real_visits_folder)
    real_visits = spark.read.parquet(real_visits_folder)
    # Getting the records that do not have a visit_id
    domain_records = data.where(
        f.col("omop_table") != "visit_occurrence"
    ).withColumn(
        "record_id",
        f.row_number().over(Window.orderBy(f.monotonically_increasing_id()))
    )

    # This is important to have a deterministic behavior for generating record_id
    temp_domain_records_folder = os.path.join(visit_reconstruction_folder, "temp_domain_records")
    domain_records.write.mode("overwrite").parquet(temp_domain_records_folder)
    domain_records = spark.read.parquet(temp_domain_records_folder)

    # Join the records to the nearest visits if they occur within the visit span with aliasing
    domain_records = domain_records.alias("domain").join(
        real_visits.where(f.col("code").isin(['Visit/IP', 'Visit/ERIP'])).alias("visit"),
        (f.col("domain.patient_id") == f.col("visit.patient_id")) &
        (f.col("domain.start").between(f.col("visit.start"), f.col("visit.end"))),
        "left_outer"
    ).withColumn(
        "ranking",
        f.row_number().over(
            Window.partitionBy("domain.record_id").orderBy(
                f.abs(f.unix_timestamp("visit.start") - f.unix_timestamp("domain.start"))
            )
        )
    ).where(
        f.col("ranking") == 1
    ).select(
        [f.col("domain." + _).alias(_) for _ in domain_records.schema.fieldNames() if _ != "visit_id"] +
        [f.coalesce(f.col("visit.visit_id"), f.col("domain.visit_id")).alias("visit_id")]
    )

    max_visit_id_df = real_visits.select(f.max("visit_id").alias("max_visit_id"))
    orphan_records = domain_records.where(
        f.col("visit_id").isNull()
    ).where(
        f.col("omop_table") != "person"
    ).crossJoin(
        max_visit_id_df
    ).withColumn(
        "new_visit_id",
        f.dense_rank().over(
            Window.orderBy(f.col("patient_id"), f.col("start").cast(t.DateType()))
        ).cast(t.LongType()) + f.col("max_visit_id").cast(t.LongType())
    ).drop(
        "visit_id"
    )
    orphan_records.groupby("new_visit_id").agg(
        f.countDistinct("patient_id").alias("pat_count")
    ).select(
        f.assert_true(f.col("pat_count") == 1)
    ).collect()

    # Link the artificial visit_ids back to the domain_records
    domain_records = domain_records.alias("domain").join(
        orphan_records.alias("orphan").select(
            f.col("orphan.record_id"),
            f.col("orphan.new_visit_id"),
        ),
        f.col("domain.record_id") == f.col("orphan.record_id"),
        "left_outer"
    ).withColumn(
        "update_visit_id",
        f.coalesce(f.col("orphan.new_visit_id"), f.col("domain.visit_id"))
    ).select(
        [
            f.col("domain." + field).alias(field)
            for field in domain_records.schema.fieldNames() if not field.endswith("visit_id")
        ] + [f.col("update_visit_id").alias("visit_id")]
    ).drop(
        "record_id"
    )

    # Generate the artificial visits
    artificial_visits = orphan_records.groupBy("new_visit_id", "patient_id").agg(
        f.min("start").alias("start"),
        f.max("start").alias("end")
    ).withColumn(
        "code",
        f.lit(0)
    ).withColumn(
        "value",
        f.lit(None).cast(t.StringType())
    ).withColumn(
        "unit",
        f.lit(None).cast(t.StringType())
    ).withColumn(
        "omop_table",
        f.lit("visit_occurrence")
    ).withColumnRenamed(
        "new_visit_id", "visit_id"
    ).drop("record_id")

    artificial_visits_folder = os.path.join(visit_reconstruction_folder, "artificial_visits")
    artificial_visits.write.mode("overwrite").parquet(artificial_visits_folder)
    artificial_visits = spark.read.parquet(artificial_visits_folder)

    # Drop visit_start_date and visit_end_date
    real_visits = real_visits.drop("visit_start_date", "visit_end_date")

    # Validate the uniqueness of visit_id
    artificial_visits.groupby("visit_id").count().select(f.assert_true(f.col("count") == 1)).collect()

    return domain_records.unionByName(
        real_visits
    ).unionByName(
        artificial_visits
    )

def disconnect_visit_id(
        data: DataFrame,
        spark: SparkSession,
        cache_folder: str,
        day_cutoff: int = 1
):
    # There are records that fall outside the corresponding visits, the time difference could be days
    # and evens years apart, this is likely that the timestamps of the lab events are the time when the
    # lab results came back instead of when the labs were sent out, therefore creating the time discrepancy.
    # In this case, we will disassociate such records with the visits and will try to connect them to the
    # other visits.
    visit_reconstruction_folder = os.path.join(cache_folder, "visit_reconstruction")
    domain_records = data.where(f.col("omop_table") != "visit_occurrence")
    visit_records = data.where(f.col("omop_table") == "visit_occurrence")
    visit_inferred_start_end = domain_records.alias("domain").join(
        visit_records.alias("visit"),
        f.col("domain.visit_id") == f.col("visit.visit_id"),
    ).groupby("domain.visit_id").agg(
        f.min("domain.start").alias("start"),
        f.max("domain.start").alias("end")
    )
    visit_to_fix = visit_inferred_start_end.alias("d_visit").join(
        visit_records.alias("visit"),
        f.col("d_visit.visit_id") == f.col("visit.visit_id"),
    ).where(
        # If the record is 24 * day_cutoff hours before the visit_start or
        # if the record is 24 * day_cutoff hours after the visit_end
        ((f.unix_timestamp("visit.start") - f.unix_timestamp("d_visit.start")) / 3600 > day_cutoff * 24) |
        ((f.unix_timestamp("d_visit.end") - f.unix_timestamp("visit.end")) / 3600 > day_cutoff * 24)
    ).select(
        f.col("visit.visit_id").alias("visit_id"),
        f.col("visit.start").alias("start"),
        f.col("visit.end").alias("end"),
        f.col("d_visit.start").alias("inferred_start"),
        f.col("d_visit.end").alias("inferred_end"),
    )
    visit_to_fix_folder = os.path.join(visit_reconstruction_folder, "visit_to_fix")
    visit_to_fix.write.mode("overwrite").parquet(visit_to_fix_folder)
    visit_to_fix = spark.read.parquet(visit_to_fix_folder)

    # Identify the unique visit_id/start pairs, we will identify the boundary of the visit
    distinct_visit_date_mapping = domain_records.alias("domain").join(
        visit_to_fix.alias("visit"),
        f.col("domain.visit_id") == f.col("visit.visit_id"),
    ).select(
        f.col("domain.visit_id").alias("visit_id"),
        f.col("domain.start").alias("start"),
        f.col("domain.code").alias("code"),
    ).distinct().withColumn(
        "visit_order",
        f.row_number().over(
            Window.partitionBy("visit_id").orderBy("start")
        )
    ).withColumn(
        "prev_start",
        f.lag("start").over(
            Window.partitionBy("visit_id").orderBy("visit_order")
        )
    ).withColumn(
        "hour_diff",
        f.coalesce(
            (f.unix_timestamp("start") - f.unix_timestamp("prev_start")) / 3600,
            f.lit(0)
        )
    ).withColumn(
        "visit_partition",
        f.sum((f.col("hour_diff") > 24).cast("int")).over(
            Window.partitionBy("visit_id").orderBy("visit_order")
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
    ).withColumn(
        "visit_partition_rank",
        f.dense_rank().over(Window.orderBy(f.col("visit_id"), f.col("visit_partition")))
    ).crossJoin(
        visit_records.select(f.max("visit_id").alias("max_visit_id"))
    ).withColumn(
        "new_visit_id",
        f.col("max_visit_id") + f.col("visit_partition_rank")
    ).drop(
        "max_visit_id", "row_number"
    )

    # Connect visit partitions in chronological order
    distinct_visit_date_pair_folder = os.path.join(visit_reconstruction_folder, "distinct_visit_date_mapping")
    distinct_visit_date_mapping.write.mode("overwrite").parquet(distinct_visit_date_pair_folder)
    distinct_visit_date_mapping = spark.read.parquet(distinct_visit_date_pair_folder)

    fix_visit_records = data.alias("ehr").join(
        distinct_visit_date_mapping.alias("visit"),
        f.col("ehr.visit_id") == f.col("visit.visit_id"),
    ).where(
        f.col("ehr.omop_table") == "visit_occurrence"
    ).groupby(
        f.col("visit.visit_id").alias("original_visit_id"),
        f.col("visit.new_visit_id").alias("visit_id"),
        f.col("ehr.patient_id").alias("patient_id"),
        f.col("ehr.code").alias("code"),
        f.col("ehr.value").alias("value"),
        f.col("ehr.unit").alias("unit"),
        f.col("ehr.omop_table").alias("omop_table"),
    ).agg(
        f.min("visit.start").alias("start"),
        f.max("visit.start").alias("end"),
    ).withColumn(
        "code",
        f.when(
            (f.col("code").isin(['Visit/IP', 'Visit/ERIP']))
            & ((f.unix_timestamp("end") - f.unix_timestamp("start")) / 3600 <= 24),
            f.lit("Visit/OP")
        ).otherwise(f.col("code"))
    )

    # Fix visit records
    fix_visit_records_folder = os.path.join(visit_reconstruction_folder, "fix_visit_records")
    fix_visit_records.write.mode("overwrite").parquet(fix_visit_records_folder)
    fix_visit_records = spark.read.parquet(fix_visit_records_folder)

    fix_domain_records = data.alias("ehr").join(
        distinct_visit_date_mapping.alias("visit"),
        (f.col("ehr.visit_id") == f.col("visit.visit_id"))
        & (f.col("ehr.start") == f.col("visit.start"))
        & (f.col("ehr.code") == f.col("visit.code")),
    ).where(
        f.col("ehr.omop_table") != "visit_occurrence"
    ).select(
        [
            f.coalesce(f.col("visit.new_visit_id"), f.col("ehr.visit_id")).alias("visit_id"),
            f.coalesce(f.col("visit.visit_id"), f.col("ehr.visit_id")).alias("original_visit_id")
        ]
        +
        [
            f.col(f"ehr.{column}").alias(column) for column in data.columns if column != "visit_id"
        ]
    )

    # Fix domain records
    fix_domain_records_folder = os.path.join(visit_reconstruction_folder, "fix_domain_records")
    fix_domain_records.write.mode("overwrite").parquet(fix_domain_records_folder)
    fix_domain_records = spark.read.parquet(fix_domain_records_folder)

    # Retrieve other records that do not require fixing
    other_events = data.join(
        distinct_visit_date_mapping.select("visit_id").distinct(),
        "visit_id",
        "left_anti"
    ).withColumn("original_visit_id", f.col("visit_id"))

    return other_events.unionByName(fix_domain_records).unionByName(fix_visit_records)


def drop_duplicate_visits(data: DataFrame) -> DataFrame:
    """
    Removes duplicate visits based on visit priority, retaining a single record per `visit_id`.

    This function identifies duplicate visits by `visit_id` and assigns a priority to each visit type.
    Visits with the highest priority (lowest priority value) are retained, while others are dropped.
    Priority is assigned based on the `code` column:
    - "Visit/IP" and "Visit/ERIP" have the highest priority (1),
    - "Visit/ER" has medium priority (2),
    - All other visit types have the lowest priority (3).

    The function returns a DataFrame with only the highest-priority visit per `visit_id`.

    Parameters
    ----------
    data : DataFrame
        A PySpark DataFrame containing the following columns:
        - `visit_id`: Unique identifier for each visit.
        - `code`: String code indicating the type of visit, which determines visit priority.

    Returns
    -------
    DataFrame
        The input DataFrame with duplicates removed based on `visit_id` and `code` priority.
        Only the highest-priority visit is retained for each `visit_id`.
    """
    data = data.withColumn(
        "priority",
        f.when(f.col("code").isin(["Visit/IP", "Visit/ERIP"]), 1).otherwise(
            f.when(f.col("code") == "Visit/ER", 2).otherwise(3)
        )
    ).withColumn(
        "visit_rank",
        f.row_number().over(Window.partitionBy("visit_id").orderBy(f.col("priority")))
    ).where(
        f.col("visit_rank") == 1
    ).drop(
        "visit_rank",
        "priority"
    )
    return data


def main(args):
    spark = SparkSession.builder.appName("Convert EHRShot Data").getOrCreate()

    logger.info(
        f"ehr_shot_file: {args.ehr_shot_file}\n"
        f"output_folder: {args.output_folder}\n"
    )
    ehr_shot_path = os.path.join(args.output_folder, "ehr_shot")
    if args.refresh_ehrshot or not os.path.exists(ehr_shot_path):
        ehr_shot_data = spark.read.option("header", "true").schema(get_schema()).csv(
            args.ehr_shot_file
        ).withColumn(
            "visit_id",
            f.col("visit_id").cast(t.LongType())
        ).drop("_c0")
        # Add visit_id based on the time intervals between neighboring events
        ehr_shot_data = generate_visit_id(
            ehr_shot_data,
            spark,
            args.output_folder,
        )
        # Disconnect domain records whose timestamps fall outside of the corresponding visit ranges
        ehr_shot_data = disconnect_visit_id(
            ehr_shot_data,
            spark,
            args.output_folder,
            args.day_cutoff,
        )
        outpatient_visits = ehr_shot_data.where(
            ~f.col("code").isin(["Visit/IP", "Visit/ERIP"])
        ).where(f.col("omop_table") == "visit_occurrence")
        # We don't use the end column to get the max end because some end datetime could be years apart from the start date
        outpatient_visit_start_end = ehr_shot_data.join(outpatient_visits.select("visit_id"), "visit_id").where(
            f.col("omop_table").isin(
                ["condition_occurrence", "procedure_occurrence", "drug_exposure", "measurement", "observation", "death"]
            )
        ).groupby("visit_id").agg(f.min("start").alias("start"), f.max("start").alias("end")).withColumn(
            "hour_diff", (f.unix_timestamp("end") - f.unix_timestamp("start")) / 3600
        ).withColumn(
            "inpatient_indicator",
            (f.col("hour_diff") > 24).cast("int")
        )
        # Reload it from the disk to update the dataframe
        outpatient_visit_start_end_folder = os.path.join(args.output_folder, "outpatient_visit_start_end")
        outpatient_visit_start_end.write.mode("overwrite").parquet(
            outpatient_visit_start_end_folder
        )
        outpatient_visit_start_end = spark.read.parquet(outpatient_visit_start_end_folder)
        inferred_inpatient_visits = outpatient_visit_start_end.where("inpatient_indicator = 1").select(
            "visit_id", "start", "end", f.lit("Visit/IP").alias("code"),
        )
        ehr_shot_data = ehr_shot_data.alias("ehr").join(
            inferred_inpatient_visits.alias("visits"), "visit_id", "left_outer"
        ).select(
            f.col("ehr.patient_id").alias("patient_id"),
            f.when(
                f.col("ehr.omop_table") == "visit_occurrence",
                f.coalesce(f.col("visits.start"), f.col("ehr.start")),
            ).otherwise(f.col("ehr.start")).alias("start"),
            f.when(
                f.col("ehr.omop_table") == "visit_occurrence",
                f.coalesce(f.col("visits.end"), f.col("ehr.end")),
            ).otherwise(f.col("ehr.end")).alias("end"),
            f.when(
                f.col("ehr.omop_table") == "visit_occurrence",
                f.coalesce(f.col("visits.code"), f.col("ehr.code")),
            ).otherwise(f.col("ehr.code")).alias("code"),
            f.col("ehr.value").alias("value"),
            f.col("ehr.unit").alias("unit"),
            f.col("ehr.omop_table").alias("omop_table"),
            f.col("ehr.visit_id").alias("visit_id"),
            f.col("ehr.original_visit_id").alias("original_visit_id"),
        )
        ehr_shot_data.write.mode("overwrite").parquet(ehr_shot_path)

    ehr_shot_data = spark.read.parquet(ehr_shot_path)
    concept = spark.read.parquet(os.path.join(args.vocabulary_folder, "concept"))

    person = create_omop_person(ehr_shot_data, concept)
    person.write.mode("overwrite").parquet(os.path.join(args.output_folder, "person"))

    for domain_table_name, mappings in table_mapping.items():
        domain_table = ehr_shot_data.where(f.col("omop_table") == domain_table_name)
        original_columns = domain_table.schema.fieldNames()
        for column, omop_column in mappings.items():
            if omop_column.endswith("datetime"):
                domain_table = domain_table.withColumn(omop_column, f.col(column).cast(t.TimestampType()))
                domain_table = domain_table.withColumn(
                    omop_column[:-4], f.col(omop_column).cast(t.DateType())
                )
            else:
                domain_table = domain_table.withColumn(omop_column, f.col(column))

        if domain_table_name in ["measurement", "observation"]:
            domain_table = extract_value(domain_table, concept)

        domain_table = convert_code_to_omop_concept(
            domain_table, concept, "code"
        ).withColumnRenamed("concept_id", concept_id_mapping[domain_table_name])

        # There could be multiple visit
        if domain_table_name == "visit_occurrence":
            # The ehrshot dataset did not document where the patients got discharged to, so let's set everything to 0
            domain_table = drop_duplicate_visits(domain_table).withColumn(
                "discharged_to_concept_id",
                f.when(
                    f.col("visit_concept_id").isin([9201, 262, 8971, 8920]),
                    f.lit(0).cast(t.IntegerType())
                ).otherwise(f.lit(None).cast(t.IntegerType()))
            )
        else:
            # Adding the domain table id
            domain_table = domain_table.withColumn(
                domain_table_name + "_id",
                f.row_number().over(Window.orderBy(f.monotonically_increasing_id()))
            )

        domain_table.drop(
            *original_columns
        ).write.mode("overwrite").parquet(
            os.path.join(args.output_folder, domain_table_name)
        )

    for vocabulary_table in VOCABULARY_TABLES:
        if not os.path.exists(os.path.join(args.output_folder, vocabulary_table)):
            shutil.copytree(
                os.path.join(args.vocabulary_folder, vocabulary_table),
                os.path.join(args.output_folder, vocabulary_table),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for Converting external data to CEHR-BERT/GPT datasets")
    parser.add_argument(
        "--ehr_shot_file",
        dest="ehr_shot_file",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--vocabulary_folder",
        dest="vocabulary_folder",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--refresh_ehrshot",
        dest="refresh_ehrshot",
        action="store_true",
    )
    parser.add_argument(
        "--day_cutoff",
        dest="day_cutoff",
        action="store",
        type=int,
        default=1,
        required=False,
    )
    main(
        parser.parse_args()
    )

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
        t.StructField("visit_id", t.LongType(), True),  # Converted to IntegerType
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
        f.col("concept_id").alias("gender_concept_id"),
        f.col("code").alias("gender_source_value"),
    )
    ethnicity = convert_code_to_omop_concept(
        omop_person.where(f.col("code").startswith("Ethnicity")),
        concept,
        "code"
    ).select(
        f.col("patient_id").alias("person_id"),
        f.col("concept_id").alias("ethnicity_concept_id"),
        f.col("code").alias("ethnicity_source_value"),
    )
    race = convert_code_to_omop_concept(
        omop_person.where(f.col("code").startswith("Race")),
        concept,
        "code"
    ).select(
        f.col("patient_id").alias("person_id"),
        f.col("concept_id").alias("race_concept_id"),
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


def generate_visit_id(data: DataFrame, time_interval: int = 12) -> DataFrame:
    """
    Generates unique `visit_id`s for each visit based on time intervals between events per patient.

    This function identifies distinct visits within a patient's event history by analyzing time gaps
    between consecutive events. Events with gaps exceeding the specified `time_interval` (in hours)
    are considered separate visits. A unique integer `visit_id` is then assigned to each visit
    using a hash of the patient ID and visit order.

    Parameters
    ----------
    data : DataFrame
        A PySpark DataFrame containing at least the following columns:
        - `patient_id`: Identifier for each patient.
        - `start`: Start timestamp of the event.
        - `end`: (Optional) End timestamp of the event.

    time_interval : int, optional, default=12
        The maximum time gap in hours between consecutive events within the same visit. If the time
        difference between two events exceeds this interval, a new visit is assigned.

    Returns
    -------
    DataFrame
        The input DataFrame with an additional `visit_id` column, which is a unique integer identifier
        for each visit, and each `patient_id`'s events are grouped according to visit.
    """
    data = data.repartition(16)
    order_window = Window.partitionBy("patient_id").orderBy(f.col("start"))

    data = data.withColumn(
        "patient_event_order", f.row_number().over(order_window)
    ).withColumn(
        "time", f.coalesce(f.col("end"), f.col("start"))
    ).withColumn(
        "prev_time", f.coalesce(f.lag(f.col("time")).over(order_window), f.col("time"))
    ).withColumn(
        "hour_diff", (f.unix_timestamp("start") - f.unix_timestamp("prev_time")) / 3600
    ).withColumn(
        "is_gap", (f.col("hour_diff") > time_interval).cast(t.IntegerType())
    ).drop(
        "time", "prev_time", "hour_diff"
    )

    cumulative_window = Window.partitionBy("patient_id").orderBy("patient_event_order").rowsBetween(
        Window.unboundedPreceding,
        Window.currentRow
    )

    data = data.withColumn(
        "visit_order",
        f.sum("is_gap").over(cumulative_window)
    ).drop(
        "is_gap"
    )

    visit = data.select("patient_id", "visit_order").distinct().withColumn(
        "new_visit_id",
        f.abs(
            f.hash(f.concat(f.col("patient_id").cast("string"), f.col("visit_order").cast("string")))
        ).cast("bigint")
    )

    # Validate the uniqueness of visit_id
    visit.groupby("new_visit_id").count().select(f.assert_true(f.col("count") == 1))
    # Join the generated visit_id back to data
    return data.join(
        visit,
        on=["patient_id", "visit_order"]
    ).withColumn(
        "visit_id", f.coalesce(f.col("new_visit_id"), f.col("visit_id"))
    ).drop("visit_order", "patient_event_order", "new_visit_id")


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

    ehr_shot_data = spark.read.option("header", "true").schema(get_schema()).csv(
        args.ehr_shot_file
    )
    # Add visit_id based on the time intervals between neighboring events
    ehr_shot_data = generate_visit_id(
        ehr_shot_data
    )
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
            domain_table = drop_duplicate_visits(domain_table)

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
    main(
        parser.parse_args()
    )

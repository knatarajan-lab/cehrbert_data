import os
import logging
import argparse
from cehrbert_data.utils.logging_utils import add_console_logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import types as t
from pyspark.sql import functions as f
from pyspark.sql.window import Window

# Enable logging
add_console_logging()
logger = logging.getLogger(__name__)

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
    "code": "observation_source_value",
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


def extract_value(data: DataFrame, concept: DataFrame):
    numeric_pattern = "^[+-]?\\d*\\.?\\d+$"
    # Add a new column 'is_numeric' to check if 'value' is numeric
    df = data.withColumn("is_numeric", f.regexp_extract(f.col("value"), numeric_pattern, 0) != "")

    numeric_df = df.where(
        f.col("is_numeric")
    ).withColumn(
        "value_as_number",
        f.col("value").cast(t.FloatType())
    )
    # Find the unit mapping from the concept table
    unit_df = numeric_df.select("unit").distinct().join(
        concept.where(f.col("domain_id") == "Unit"), numeric_df["unit"] == concept["concept_code"],
        "left_outer"
    ).select(
        numeric_df["unit"],
        f.coalesce(concept["concept_id"], f.lit(0)).alias("unit_concept_id")
    )
    unit_df = unit_df.withColumn(
        "order",
        f.row_number().over(Window.partitionBy(f.col("unit")).orderBy(f.col("unit_concept_id")))
    ).where(
        f.col("order") == 1
    ).drop("order")

    numeric_df = numeric_df.join(
        unit_df, "unit"
    ).withColumn(
        "value_as_concept", f.lit(None).cast(t.IntegerType())
    )

    categorical_df = df.where(
        ~f.col("is_numeric")
    )
    answer_df = categorical_df.select("value").distinct().join(
        concept.where(f.col("domain_id") == "Meas Value"),
        categorical_df["value"] == concept["concept_name"]
    ).select(
        categorical_df["value"],
        f.coalesce(concept["concept_id"], f.lit(0)).alias("value_as_concept_id")
    ).withColumn(
        "order",
        f.row_number().over(Window.partitionBy(f.col("value")).orderBy(f.col("value_as_concept_id")))
    ).where(
        f.col("order") == 1
    ).drop("order")

    categorical_df = categorical_df.join(
        answer_df, "value"
    ).withColumn(
        "unit_concept_id", f.lit(None).cast(t.IntegerType())
    )

    other_df = df.where(f.col("is_numeric").isNull()).withColumn(
        "unit_concept_id", f.lit(None).cast(t.IntegerType())
    ).withColumn(
        "value_as_concept", f.lit(None).cast(t.IntegerType())
    )

    return numeric_df.unionByName(categorical_df).unionByName(other_df).withColumnRenamed(
        "value", "value_source_value"
    ).withColumnRenamed(
        "unit", "unit_source_value"
    )


def convert_code_to_omop_concept(data: DataFrame, concept: DataFrame, field: str) -> DataFrame:
    data = data.withColumn(
        "vocabulary_id",
        f.split(field, "/")[0]
    ).withColumn(
        "concept_code",
        f.split(field, "/")[1]
    )
    output_columns = [data[_] for _ in data.schema.fieldNames()] + [f.coalesce(concept["concept_id"], f.lit(0))]
    return data.join(
        concept,
        on=(data["vocabulary_id"] == concept["vocabulary_id"]) & (data["concept_code"] == concept["concept_code"]),
        how="left_outer",
    ).select(output_columns)


def main(args):
    spark = SparkSession.builder.appName("Convert EHRShot Data").getOrCreate()

    logger.info(
        f"input_folder: {args.ehr_shot_path}\n"
        f"output_folder: {args.output_folder}\n"
    )

    ehr_shot_data = spark.read.option("header", "true").schema(get_schema()).csv(
        args.ehr_shot_path
    )
    concept = spark.read.parquet(os.path.join(args.vocabulary_folder, "concept"))

    person = create_omop_person(ehr_shot_data, concept)
    person.write.mode("overwrite").parquet(os.path.join(args.output_folder, "person"))

    for domain_table_name, mappings in table_mapping.items():
        domain_table = ehr_shot_data.where(f.col("omop_table") == domain_table_name)
        original_columns = domain_table.schema.fieldNames()
        for column, omop_column in mappings.items():
            domain_table = domain_table.withColumn(omop_column, f.col(column))
        if "value" in mappings:
            domain_table = extract_value(domain_table, concept)
        domain_table.drop(original_columns)
        domain_table.write.mode("overwrite").parquet(os.path.join(args.output_folder, domain_table_name))


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

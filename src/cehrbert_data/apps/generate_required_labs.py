import argparse
import os

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from cehrbert_data.const.common import (
    CONCEPT,
    MEASUREMENT,
    REQUIRED_MEASUREMENT,
    NUMERIC_MEASUREMENT_STATS
)
from cehrbert_data.utils.spark_utils import preprocess_domain_table
from cehrbert_data.queries.measurement_queries import (
    LAB_PREVALENCE_QUERY,
    MEASUREMENT_UNIT_STATS_QUERY
)


def main(
        input_folder,
        output_folder,
        num_of_numeric_labs,
        num_of_categorical_labs,
        min_num_of_patients
):
    spark = SparkSession.builder.appName("Generate required labs").getOrCreate()

    # Load measurement as a dataframe in pyspark
    measurement = preprocess_domain_table(spark, input_folder, MEASUREMENT)
    # Load concept as a dataframe in pyspark
    concept = preprocess_domain_table(spark, input_folder, CONCEPT)
    # Create the local measurement view
    measurement.createOrReplaceTempView(MEASUREMENT)
    # Create the local concept view
    concept.createOrReplaceTempView(CONCEPT)
    # Create the
    required_lab_dataframe = generate_required_labs(
        spark, num_of_numeric_labs, num_of_categorical_labs, min_num_of_patients
    )
    required_lab_dataframe.write.mode("overwrite").parquet(
        os.path.join(output_folder, REQUIRED_MEASUREMENT)
    )
    # Reload the dataframe from the disk
    required_lab_dataframe = spark.read.parquet(
        os.path.join(output_folder, REQUIRED_MEASUREMENT)
    )
    required_lab_dataframe.createOrReplaceTempView(REQUIRED_MEASUREMENT)
    numeric_measurement_stats_dataframe = spark.sql(MEASUREMENT_UNIT_STATS_QUERY)
    numeric_measurement_stats_dataframe.write.mode("overwrite").parquet(
        os.path.join(output_folder, NUMERIC_MEASUREMENT_STATS)
    )


def generate_required_labs(
        spark: SparkSession,
        num_of_numeric_labs: int,
        num_of_categorical_labs: int,
        min_num_of_patients: int
) -> DataFrame:

    prevalent_labs = spark.sql(LAB_PREVALENCE_QUERY)
    prevalent_labs = prevalent_labs.where(F.col("person_count") >= min_num_of_patients)
    # Cache the dataframe for faster computation in the below transformations
    prevalent_labs.cache()
    prevalent_numeric_labs = (
        prevalent_labs.withColumn("is_numeric", F.col("numeric_percentage") >= 0.5)
        .where("is_numeric")
        .withColumn("rn", F.row_number().over(Window.orderBy(F.desc("freq"))))
        .where(F.col("rn") <= num_of_numeric_labs)
        .drop("rn")
    )
    prevalent_categorical_labs = (
        prevalent_labs.withColumn("is_categorical", F.col("categorical_percentage") >= 0.5)
        .where("is_categorical")
        .withColumn("is_numeric", ~F.col("is_categorical"))
        .withColumn("rn", F.row_number().over(Window.orderBy(F.desc("freq"))))
        .where(F.col("rn") <= num_of_categorical_labs)
        .drop("is_categorical")
        .drop("rn")
    )
    return prevalent_numeric_labs.unionAll(prevalent_categorical_labs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for generate " "required labs to be included")
    parser.add_argument(
        "-i",
        "--input_folder",
        dest="input_folder",
        action="store",
        help="The path for your input_folder where the raw data is",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The path for your output_folder",
        required=True,
    )
    parser.add_argument(
        "--num_of_numeric_labs",
        dest="num_of_numeric_labs",
        action="store",
        type=int,
        default=100,
        help="The top most prevalent numeric labs to be included",
        required=False,
    )
    parser.add_argument(
        "--num_of_categorical_labs",
        dest="num_of_categorical_labs",
        action="store",
        type=int,
        default=100,
        help="The top most prevalent categorical labs to be included",
        required=False,
    )
    parser.add_argument(
        "--min_num_of_patients",
        dest="min_num_of_patients",
        action="store",
        type=int,
        default=0,
        help="Min no.of patients linked to concepts to be included",
        required=False,
    )

    ARGS = parser.parse_args()

    main(
        ARGS.input_folder,
        ARGS.output_folder,
        ARGS.num_of_numeric_labs,
        ARGS.num_of_categorical_labs,
        ARGS.min_num_of_patients
    )

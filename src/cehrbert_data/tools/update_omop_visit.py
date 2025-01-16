import os
import argparse
import shutil

from cehrbert_data.tools.ehrshot_to_omop import table_mapping, VOCABULARY_TABLES
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

def main(args):
    spark = SparkSession.builder.appName("Clean up visit_occurrence").getOrCreate()
    visit_mapping = spark.read.parquet(
        os.path.join(args.output_folder, "visit_mapping")
    )
    omop_table_name: str
    for omop_table_name in table_mapping.keys():
        if omop_table_name not in ["visit_occurrence", "death"]:
            omop_table = spark.read.parquet(os.path.join(args.input_folder, omop_table_name))
            omop_table.alias("domain").join(
                visit_mapping.alias("visit"),
                on=f.col("domain.visit_occurrence_id") == f.col("visit.visit_occurrence_id"),
                how="left"
            ).select(
                [
                    f.coalesce(
                        f.col("visit.master_visit_occurrence_id"),
                        f.col("domain.visit_occurrence_id")
                    ).alias("visit_occurrence_id")
                ] +
                [
                    f.col(f"domain.{column}").alias(column)
                    for column in omop_table.columns if column != "visit_occurrence_id"
                ]
            )
            omop_table.write.mode("overwrite").parquet(os.path.join(args.output_folder, omop_table_name))

    vocabulary_table: str
    for vocabulary_table in VOCABULARY_TABLES + ["person"]:
        if not os.path.exists(os.path.join(args.output_folder, vocabulary_table)):
            shutil.copytree(
                os.path.join(args.vocabulary_folder, vocabulary_table),
                os.path.join(args.output_folder, vocabulary_table),
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for connecting OMOP visits in chronological order")
    parser.add_argument(
        "--input_folder",
        dest="input_folder",
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

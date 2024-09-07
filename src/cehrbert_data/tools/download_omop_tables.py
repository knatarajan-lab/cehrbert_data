import argparse
import configparser
import os.path

from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from cehrbert_data.const.common import (
    CONDITION_OCCURRENCE,
    DRUG_EXPOSURE,
    MEASUREMENT,
    OBSERVATION,
    PERSON,
    PROCEDURE_OCCURRENCE,
    VISIT_OCCURRENCE,
)

omop_table_dict = {
    PERSON: "person_id",
    CONDITION_OCCURRENCE: "condition_occurrence_id",
    MEASUREMENT: "measurement_id",
    DRUG_EXPOSURE: "drug_exposure_id",
    PROCEDURE_OCCURRENCE: "procedure_occurrence_id",
    OBSERVATION: "observation_id",
    VISIT_OCCURRENCE: "visit_occurrence_id",
}


def find_num_of_records(domain_table_name, db_properties, column_name, spark_session):
    table_max_id = (
        spark_session.read.format("jdbc")
        .option("driver", db_properties["driver"])
        .option("url", db_properties["base_url"])
        .option(
            "dbtable",
            "(SELECT MAX({}) AS {} FROM {}) as {}".format(column_name, column_name, domain_table_name, column_name),
        )
        .option("user", db_properties["user"])
        .option("password", db_properties["password"])
        .load()
        .select("{}".format(column_name))
        .collect()[0]["{}".format(column_name)]
    )
    return table_max_id


def download_omop_tables_with_partitions(domain_table, column_name, db_properties, output_folder, spark_session):
    table = (
        spark_session.read.format("jdbc")
        .option("url", db_properties["base_url"])
        .option("dbtable", "%s" % domain_table)
        .option("user", db_properties["user"])
        .option("password", db_properties["password"])
        .option("numPartitions", 16)
        .option("partitionColumn", column_name)
        .option("lowerBound", 1)
        .option(
            "upperBound",
            find_num_of_records(domain_table, db_properties, column_name, spark_session),
        )
        .load()
    )
    table.write.mode("overwrite").parquet(output_folder + "/" + str(domain_table) + "/")


def download_omop_tables(domain_table, db_properties, output_folder, spark_session):
    table = (
        spark_session.read.format("jdbc")
        .option("url", db_properties["base_url"])
        .option("dbtable", "%s" % domain_table)
        .option("user", db_properties["user"])
        .option("password", db_properties["password"])
        .load()
    )
    table.write.mode("overwrite").parquet(output_folder + "/" + str(domain_table) + "/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for downloading OMOP tables")

    parser.add_argument(
        "-c",
        "--credential_path",
        dest="credential_path",
        action="store",
        help="The path for your database credentials",
        required=True,
    )

    parser.add_argument(
        "-tc",
        "--domain_table_list",
        dest="domain_table_list",
        nargs="+",
        action="store",
        help="The list of domain tables you want to download",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The output folder that stores the domain tables download destination",
        required=True,
    )

    ARGS = parser.parse_args()
    spark = SparkSession.builder.appName("Download OMOP tables").getOrCreate()
    domain_table_list = ARGS.domain_table_list
    credential_path = ARGS.credential_path
    download_folder = ARGS.output_folder
    config = configparser.ConfigParser()
    config.read(credential_path)
    properties = config.defaults()
    downloaded_tables = []

    for item in domain_table_list:
        try:
            if item in omop_table_dict:
                download_omop_tables_with_partitions(
                    item, omop_table_dict.get(item), properties, download_folder, spark
                )
            else:
                download_omop_tables(item, properties, download_folder, spark)
            downloaded_tables.append(item)
            print("table: " + str(item) + " is downloaded")
        except Exception as e:
            print(str(e))

    print("The following tables were downloaded:" + str(downloaded_tables))
    patient_splits_folder = os.path.join(download_folder, "patient_splits")
    if not os.path.exists(patient_splits_folder):
        print("Creating the patient splits")
        person = spark.read.parquet(os.path.join(download_folder, "person"))
        train_split, test_split = person.select("person_id").randomSplit([0.8, 0.2], seed=42)
        train_split = train_split.withColumn("split", f.lit("train"))
        test_split = test_split.withColumn("split", f.lit("test"))
        patient_splits = train_split.unionByName(test_split)
        patient_splits.write.parquet(os.path.join(download_folder, "patient_splits"))

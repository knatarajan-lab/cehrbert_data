"""
This module generates a qualified concept list by processing patient event data across various.

domain tables (e.g., condition_occurrence, procedure_occurrence, drug_exposure) and applying a
patient frequency filter to retain concepts linked to a minimum number of patients.

Key Functions:
    - preprocess_domain_table: Preprocesses domain tables to prepare for event extraction.
    - join_domain_tables: Joins multiple domain tables into a unified DataFrame.
    - main: Coordinates the entire process of reading domain tables, applying frequency filters,
      and saving the qualified concept list.

Command-line Arguments:
    - input_folder: Directory containing the input data.
    - output_folder: Directory where the qualified concept list will be saved.
    - min_num_of_patients: Minimum number of patients linked to a concept for it to be included.
    - with_drug_rollup: Boolean flag indicating whether drug concept rollups should be applied.
"""

import os
from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from cehrbert_data.config.output_names import QUALIFIED_CONCEPT_LIST_PATH
from cehrbert_data.const.common import CONCEPT
from cehrbert_data.utils.spark_utils import extract_events_by_domain, preprocess_domain_table

DOMAIN_TABLE_LIST = ["condition_occurrence", "procedure_occurrence", "drug_exposure"]


def main(
        input_folder: str,
        output_folder: str,
        min_num_of_patients: int,
        with_drug_rollup: bool = True,
        ehr_table_list: List[str] = None
):
    """
    Main function to generate a qualified concept list based on patient event data from multiple.

    domain tables.

    Args:
        input_folder (str): The directory where the input data is stored.
        output_folder (str): The directory where the output (qualified concept list) will be saved.
        min_num_of_patients (int): Minimum number of patients that a concept must be linked to for
        inclusion.
        with_drug_rollup (bool): If True, applies drug rollup logic to the drug_exposure domain.
        ehr_table_list (List[str]): List of patient event tables to include in the concept list.

    The function processes patient event data across various domain tables, excludes low-frequency
    concepts, and saves the filtered concepts to a specified output folder.
    """
    spark = SparkSession.builder.appName("Generate concept list").getOrCreate()

    # Exclude measurement from domain_table_list if exists because we need to process measurement
    # in a different way
    domain_table_list = ehr_table_list if ehr_table_list else DOMAIN_TABLE_LIST
    concept = preprocess_domain_table(spark, input_folder, CONCEPT)
    patient_ehr_events = None
    for domain_table_name in domain_table_list:
        domain_table = preprocess_domain_table(
            spark=spark,
            input_folder=input_folder,
            domain_table_name=domain_table_name,
            with_drug_rollup=with_drug_rollup
        )
        ehr_events = extract_events_by_domain(
            domain_table,
            spark=spark,
            concept=concept,
            aggregate_by_hour=False,
            refresh=False,
            persistence_folder=input_folder
        )
        if patient_ehr_events is None:
            patient_ehr_events = ehr_events
        else:
            patient_ehr_events = patient_ehr_events.unionByName(ehr_events)

    # Filter out concepts that are linked to less than 100 patients
    qualified_concepts = (
        patient_ehr_events.where("visit_occurrence_id IS NOT NULL")
        .groupBy("standard_concept_id")
        .agg(F.countDistinct("person_id").alias("freq"))
        .where(F.col("freq") >= min_num_of_patients)
    )
    qualified_concepts.write.mode("overwrite").parquet(os.path.join(output_folder, QUALIFIED_CONCEPT_LIST_PATH))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arguments for generate concept list to be included")
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
        "--min_num_of_patients",
        dest="min_num_of_patients",
        action="store",
        type=int,
        default=0,
        help="Min no.of patients linked to concepts to be included",
        required=False,
    )
    parser.add_argument("--with_drug_rollup", dest="with_drug_rollup", action="store_true")
    parser.add_argument(
        "--ehr_table_list",
        dest="ehr_table_list",
        nargs="+",
        action="store",
        help="The list of domain tables you want to include for feature extraction",
        required=False,
    )

    ARGS = parser.parse_args()

    main(
        ARGS.input_folder,
        ARGS.output_folder,
        ARGS.min_num_of_patients,
        ARGS.with_drug_rollup,
        ARGS.ehr_table_list,
    )

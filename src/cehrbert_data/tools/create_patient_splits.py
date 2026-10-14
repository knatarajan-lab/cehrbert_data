"""Create the patient_splits table, which assigns every patient of the person table to the train or test split.

The table has the columns (person_id, split) with split in ("train", "test"), and it is written as parquet to
<input_folder>/patient_splits, where generate_training_data picks it up to write patient_sequence/train and
patient_sequence/test.

Example:
    python -m cehrbert_data.tools.create_patient_splits --input_folder /path/to/omop --train_ratio 0.8 --seed 42
"""

import argparse
import logging
import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from cehrbert_data.const.common import PERSON
from cehrbert_data.utils.omop_tables import omop_table_exists, read_omop_table

PATIENT_SPLITS = "patient_splits"
TRAIN = "train"
TEST = "test"

LOG = logging.getLogger(__name__)


def create_patient_splits(person: DataFrame, train_ratio: float = 0.8, seed: int = 42) -> DataFrame:
    """
    Split the patients randomly (but reproducibly) into the train and test sets.

    :param person: the person table
    :param train_ratio: the proportion of the patients in the train set, between 0 and 1 (exclusive)
    :param seed: the seed of the random split
    :return: the table (person_id, split)
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    # The split is a hash of the person_id and the seed, not Spark's randomSplit whose result depends on how the person
    # table is partitioned. The same seed gives the same split for parquet or CSV files, in any order and partitioning,
    # and the ratio holds for a large number of patients.
    buckets = 1_000_000
    bucket = F.expr(f"pmod(xxhash64(person_id, {int(seed)}), {buckets})")
    return (
        person.select("person_id")
        .distinct()
        .withColumn("split", F.when(bucket < F.lit(int(train_ratio * buckets)), F.lit(TRAIN)).otherwise(F.lit(TEST)))
    )


def main(
        input_folder: str,
        output_folder: str = None,
        train_ratio: float = 0.8,
        seed: int = 42,
        overwrite: bool = False,
):
    spark = (
        SparkSession.builder.appName("Create patient splits")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    if not omop_table_exists(input_folder, PERSON):
        raise FileNotFoundError(f"The person table is not in {input_folder}")
    output_path = os.path.join(output_folder or input_folder, PATIENT_SPLITS)
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"{output_path} already exists, use --overwrite to replace it")

    patient_splits = create_patient_splits(read_omop_table(spark, input_folder, PERSON), train_ratio, seed)
    patient_splits.write.mode("overwrite").parquet(output_path)

    counts = {row["split"]: row["count"] for row in spark.read.parquet(output_path).groupBy("split").count().collect()}
    LOG.info("Wrote the patient splits to %s: %s", output_path, counts)
    print(f"Wrote the patient splits to {output_path}: {counts}")


def create_argparser():
    parser = argparse.ArgumentParser(description="Create the patient splits (train/test) from the person table")
    parser.add_argument(
        "--input_folder",
        dest="input_folder",
        action="store",
        required=True,
        help="The folder with the OMOP tables (parquet or CSV), the person table is read from it",
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        action="store",
        default=None,
        help="The folder where patient_splits is written, the input folder by default",
    )
    parser.add_argument(
        "--train_ratio",
        dest="train_ratio",
        action="store",
        type=float,
        default=0.8,
        help="The proportion of the patients in the train set, 0.8 by default",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        action="store",
        type=int,
        default=42,
        help="The seed of the random split, 42 by default",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Replace the patient_splits that already exists",
    )
    return parser


if __name__ == "__main__":
    ARGS = create_argparser().parse_args()
    main(ARGS.input_folder, ARGS.output_folder, ARGS.train_ratio, ARGS.seed, ARGS.overwrite)

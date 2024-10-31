import os
from pathlib import Path
import shutil
from enum import Enum

from pyspark.sql import SparkSession
from pyspark.sql import types as t
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from cehrbert_data.utils.spark_parse_args import create_spark_args
from cehrbert_data.utils.spark_utils import extract_ehr_records, create_sequence_data_with_att


class PredictionType(Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


def create_feature_extraction_args():
    spark_args = create_spark_args(
        parse=False
    )
    spark_args.add_argument(
        '--cohort_dir',
        required=True
    )
    spark_args.add_argument(
        '--person_id_column',
        required=True
    )
    spark_args.add_argument(
        '--index_date_column',
        required=True
    )
    spark_args.add_argument(
        '--label_column',
        required=True
    )
    spark_args.add_argument(
        '--prediction_type',
        choices=[e.value for e in PredictionType],
        required=False,
        default=PredictionType.BINARY
    )
    return spark_args.parse_args()


def main(args):
    cohort_path = Path(args.cohort_dir)
    if not cohort_path.exists():
        raise ValueError(f"{args.cohort_dir} does not exist!")
    spark = SparkSession.builder.appName(
        f"Extract Features for existing cohort {args.cohort_name}"
    ).getOrCreate()

    cohort = spark.read. \
        option("header", "true"). \
        option("inferSchema", "true"). \
        csv(args.cohort_dir). \
        withColumnRenamed(args.person_id_column, "person_id"). \
        withColumnRenamed(args.index_date_column, "index_date"). \
        withColumnRenamed(args.label_column, "label")

    if PredictionType.REGRESSION:
        cohort = cohort.withColumn("label", f.col("label").cast(t.FloatType()))
    else:
        cohort = cohort.withColumn("label", f.col("label").cast(t.IntegerType()))

    cohort_member_id_udf = f.row_number().over(Window.orderBy("person_id", "index_date"))
    cohort = cohort.withColumn("cohort_member_id", cohort_member_id_udf)

    ehr_records = extract_ehr_records(
        spark,
        input_folder=args.input_folder,
        domain_table_list=args.ehr_table_list,
        include_visit_type=args.include_visit_type,
        with_diagnosis_rollup=args.is_roll_up_concept,
        with_drug_rollup=args.is_drug_roll_up_concept,
        include_concept_list=args.include_concept_list,
        refresh_measurement=args.refresh_measurement,
    )

    # Drop index_date because create_sequence_data_with_att does not expect this column
    ehr_records = cohort.select("person_id", "cohort_member_id", "index_date").join(
        ehr_records,
        "person_id"
    ).where(ehr_records["date"] <= cohort["index_date"]).drop("index_date")

    birthdate_udf = f.coalesce(
        "birth_datetime",
        f.concat("year_of_birth", f.lit("-01-01")).cast("timestamp"),
    )
    person = spark.read.parquet(os.path.join(args.input_folder, "person"))
    patient_demographic = person.select(
        "person_id",
        birthdate_udf.alias("birth_datetime"),
        "race_concept_id",
        "gender_concept_id",
    )
    visit_occurrence = spark.read.parquet(os.path.join(args.input_folder, "visit_occurrence"))
    age_udf = f.ceil(f.months_between(f.col("visit_start_date"), f.col("birth_datetime")) / f.lit(12))
    visit_occurrence_person = (
        visit_occurrence
        .join(patient_demographic, "person_id")
        .withColumn("age", age_udf)
        .drop("birth_datetime")
    )

    ehr_records = create_sequence_data_with_att(
        ehr_records,
        visit_occurrence=visit_occurrence_person,
        include_visit_type=args.include_visit_type,
        exclude_visit_tokens=args.exclude_visit_tokens,
        patient_demographic=(
            patient_demographic if args.gpt_patient_sequence else None
        ),
        att_type=args.att_type,
        exclude_demographic=args.exclude_demographic,
        use_age_group=args.use_age_group
    )

    cohort = ehr_records.join(
        cohort,
        (ehr_records.person_id == cohort.person_id) & (ehr_records.cohort_member_id == cohort.cohort_member_id),
    ).select([ehr_records[_] for _ in ehr_records.schema.fieldNames()] + [cohort["index_date"], cohort["label"]])

    if args.patient_splits_folder:
        patient_splits = spark.read.parquet(args.patient_splits_folder)
        cohort.join(patient_splits, "person_id").write.mode(
            "overwrite"
        ).parquet(os.path.join(args.output_folder, "temp"))
        # Reload the data from the disk
        cohort = spark.read.parquet(os.path.join(args.output_folder, "temp"))
        cohort.where('split="train"').write.mode("overwrite").parquet(
            os.path.join(args.output_folder, "train")
        )
        cohort.where('split="test"').write.mode("overwrite").parquet(os.path.join(args.output_folder, "test"))
        shutil.rmtree(os.path.join(args.output_folder, "temp"))
    else:
        cohort.write.mode("overwrite").parquet(args.output_folder)


if __name__ == "__main__":
    main(
        create_feature_extraction_args()
    )

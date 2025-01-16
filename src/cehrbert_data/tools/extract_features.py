import os
from pathlib import Path
import shutil
from enum import Enum

from pyspark.sql import SparkSession
from pyspark.sql import types as t
from pyspark.sql import functions as f
from pyspark.sql.window import Window

from cehrbert_data.decorators import AttType
from cehrbert_data.utils.spark_parse_args import create_spark_args
from cehrbert_data.utils.spark_utils import (
    extract_ehr_records, create_sequence_data_with_att, create_concept_frequency_data
)


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
    spark_args.add_argument(
        '--bound_visit_end_date',
        action='store_true',
    )
    spark_args.add_argument(
        "--include_inpatient_hour_token",
        dest="include_inpatient_hour_token",
        action="store_true",
    )
    return spark_args.parse_args()


def main(args):
    cohort_path = Path(args.cohort_dir)
    if not cohort_path.exists():
        raise ValueError(f"{args.cohort_dir} does not exist!")
    spark = SparkSession.builder.appName(
        f"Extract Features for existing cohort {args.cohort_name}"
    ).getOrCreate()

    cohort_csv = spark.read. \
        option("header", "true"). \
        option("inferSchema", "true"). \
        csv(args.cohort_dir). \
        withColumnRenamed(args.person_id_column, "person_id"). \
        withColumnRenamed(args.index_date_column, "index_date"). \
        withColumnRenamed(args.label_column, "label"). \
        withColumn("index_date", f.col("index_date").cast(t.TimestampType()))

    if PredictionType.REGRESSION:
        cohort_csv = cohort_csv.withColumn("label", f.col("label").cast(t.FloatType()))
    else:
        cohort_csv = cohort_csv.withColumn("label", f.col("label").cast(t.IntegerType()))

    cohort_member_id_udf = f.row_number().over(Window.orderBy("person_id", "index_date"))
    cohort_csv = cohort_csv.withColumn("cohort_member_id", cohort_member_id_udf)

    # Save cohort as parquet files
    cohort_temp_folder = os.path.join(
        args.output_folder, args.cohort_name, "cohort"
    )
    cohort_csv.write.mode("overwrite").parquet(cohort_temp_folder)
    cohort = spark.read.parquet(cohort_temp_folder)

    ehr_records = extract_ehr_records(
        spark,
        input_folder=args.input_folder,
        domain_table_list=args.ehr_table_list,
        include_visit_type=args.include_visit_type,
        with_diagnosis_rollup=args.is_roll_up_concept,
        with_drug_rollup=args.is_drug_roll_up_concept,
        include_concept_list=args.include_concept_list,
        refresh_measurement=args.refresh_measurement,
        aggregate_by_hour=args.aggregate_by_hour,
    )

    # Drop index_date because create_sequence_data_with_att does not expect this column
    ehr_records = cohort.select("person_id", "cohort_member_id", "index_date").join(
        ehr_records,
        "person_id"
    ).where(ehr_records["datetime"] <= cohort["index_date"])
    ehr_records_temp_folder = os.path.join(
        args.output_folder, args.cohort_name, "ehr_records"
    )
    ehr_records.write.mode("overwrite").parquet(ehr_records_temp_folder)
    ehr_records = spark.read.parquet(ehr_records_temp_folder)

    visit_occurrence = spark.read.parquet(os.path.join(args.input_folder, "visit_occurrence"))
    cohort_visit_occurrence = visit_occurrence.join(
        cohort.select("person_id", "cohort_member_id", "index_date"),
        "person_id"
    ).withColumn(
        "visit_start_date",
        f.col("visit_start_date").cast(t.DateType())
    ).withColumn(
        "visit_end_date",
        f.coalesce(f.col("visit_end_date"), f.col("visit_start_date")).cast(t.DateType())
    ).withColumn(
        "visit_start_datetime",
        f.col("visit_start_datetime").cast(t.TimestampType())
    ).withColumn(
        "visit_end_datetime",
        f.coalesce(
            f.col("visit_end_datetime"),
            f.col("visit_end_date").cast(t.TimestampType()),
            f.col("visit_start_datetime")
        ).cast(t.TimestampType())
    ).where(
        f.col("visit_start_datetime") <= f.col("index_date")
    )

    # For each patient/index_date pair, we get the last record before the index_date
    # we get the corresponding visit_occurrence_id and index_date
    if args.bound_visit_end_date:
        cohort_visit_occurrence = cohort_visit_occurrence.withColumn(
            "time_diff_from_index_date",
            f.abs(f.unix_timestamp("index_date") - f.unix_timestamp("visit_start_datetime"))
        ).withColumn(
            "visit_rank",
            f.row_number().over(
                Window.partitionBy("person_id", "cohort_member_id").orderBy("time_diff_from_index_date")
            )
        ).drop("time_diff_from_index_date")

        # We create placeholder tokens for those inpatient visits, where the first token occurs after the index_date
        placeholder_tokens = cohort_visit_occurrence.where(
            f.col("visit_rank") == 1
        ).select(
            "person_id",
            "cohort_member_id",
            "index_date",
            "visit_occurrence_id",
            f.lit("0").alias("standard_concept_id"),
            f.col("index_date").cast(t.DateType()).alias("date"),
            f.col("index_date").alias("datetime"),
            f.lit("unknown").alias("domain"),
            f.lit(None).cast(t.StringType()).alias("unit"),
            f.lit(None).cast(t.FloatType()).alias("number_as_value"),
            f.lit(None).cast(t.StringType()).alias("concept_as_value"),
            f.lit(None).cast(t.StringType()).alias("event_group_id"),
            "visit_concept_id",
            f.lit(-1).alias("age")
        ).join(
            ehr_records.select("cohort_member_id", "visit_occurrence_id"),
            ["cohort_member_id", "visit_occurrence_id"],
            "left_anti",
        )

        placeholder_tokens.write.mode("overwrite").parquet(
            os.path.join(args.output_folder, args.cohort_name, "placeholder_tokens")
        )
        # Add an artificial token for the visit in which the prediction is made
        ehr_records = ehr_records.unionByName(
            placeholder_tokens
        )

        # Bound the visit_end_date and visit_end_datetime
        cohort_visit_occurrence = cohort_visit_occurrence.withColumn(
            "visit_end_datetime",
            f.when(
                f.col("visit_end_datetime") > f.col("index_date"),
                f.col("index_date")
            ).otherwise(f.col("visit_end_datetime"))
        ).withColumn(
            "visit_end_date",
            f.col("visit_end_datetime").cast(t.DateType())
        )
        cohort_member_visit_folder = os.path.join(
            args.output_folder, args.cohort_name, "cohort_member_visit_occurrence"
        )
        cohort_visit_occurrence.write.mode("overwrite").parquet(
            cohort_member_visit_folder
        )
        cohort_visit_occurrence = spark.read.parquet(cohort_member_visit_folder).drop("visit_rank")

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

    age_udf = f.ceil(f.months_between(f.col("visit_start_date"), f.col("birth_datetime")) / f.lit(12))
    visit_occurrence_person = (
        cohort_visit_occurrence
        .join(patient_demographic, "person_id")
        .withColumn("age", age_udf)
        .drop("birth_datetime")
    )

    if args.is_new_patient_representation:
        ehr_records = create_sequence_data_with_att(
            ehr_records.drop("index_date") if "index_date" in ehr_records.schema.fieldNames() else ehr_records,
            visit_occurrence=visit_occurrence_person,
            include_visit_type=args.include_visit_type,
            exclude_visit_tokens=args.exclude_visit_tokens,
            patient_demographic=(
                patient_demographic if args.gpt_patient_sequence else None
            ),
            att_type=AttType(args.att_type),
            inpatient_att_type=AttType(args.inpatient_att_type),
            exclude_demographic=args.exclude_demographic,
            use_age_group=args.use_age_group,
            include_inpatient_hour_token=args.include_inpatient_hour_token,
            spark=spark,
            persistence_folder=str(os.path.join(args.output_folder, args.cohort_name)),
        )
    elif args.is_feature_concept_frequency:
        ehr_records = create_concept_frequency_data(
            ehr_records
        )
    else:
        raise RuntimeError(
            "Can not extract features, use --is_new_patient_representation or --is_feature_concept_frequency"
        )

    cohort = cohort.join(
        person.select(
            "person_id",
            "year_of_birth",
            f.coalesce(f.col("race_concept_id"), f.lit(0)).cast(t.IntegerType()).alias("race_concept_id"),
            "gender_concept_id"
        ),
        "person_id"
    ).withColumn(
        "age",
        f.year("index_date") - f.col("year_of_birth")
    ).drop("year_of_birth")

    # Alias ehr_records and cohort to avoid column ambiguity
    cohort = ehr_records.alias("ehr").join(
        cohort.alias("cohort"),
        (f.col("ehr.person_id") == f.col("cohort.person_id")) &
        (f.col("ehr.cohort_member_id") == f.col("cohort.cohort_member_id")),
    ).select(
        [f.col("ehr." + col) for col in ehr_records.columns] +
        [
            f.col("cohort.age"),
            f.col("cohort.race_concept_id"),
            f.col("cohort.gender_concept_id"),
            f.col("cohort.index_date"),
            f.col("cohort.label")
        ]
    )

    cohort_folder = str(os.path.join(args.output_folder, args.cohort_name))
    if args.patient_splits_folder:
        patient_splits = spark.read.parquet(args.patient_splits_folder)
        cohort.join(patient_splits, "person_id").write.mode(
            "overwrite"
        ).parquet(os.path.join(cohort_folder, "temp"))
        # Reload the data from the disk
        cohort = spark.read.parquet(os.path.join(cohort_folder, "temp"))
        cohort.where('split="train"').write.mode("overwrite").parquet(
            os.path.join(cohort_folder, "train")
        )
        cohort.where('split="test"').write.mode("overwrite").parquet(os.path.join(cohort_folder, "test"))
        shutil.rmtree(os.path.join(cohort_folder, "temp"))
    else:
        cohort.write.mode("overwrite").parquet(cohort_folder)

    spark.stop()


if __name__ == "__main__":
    main(
        create_feature_extraction_args()
    )

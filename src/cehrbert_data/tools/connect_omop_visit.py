import os
import argparse
from typing import Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import types as t
from pyspark.sql import functions as f
from pyspark.sql.window import Window


def connect_visits_in_chronological_order(
        spark: SparkSession,
        visit_to_fix: DataFrame,
        visit_occurrence: DataFrame,
        hour_diff_threshold: int,
        workspace_folder: str,
        visit_name: str
):
    visit_to_fix = visit_to_fix.withColumn(
        "visit_end_datetime",
        f.coalesce("visit_end_datetime", f.col("visit_end_date").cast(t.TimestampType()))
    ).withColumn(
        "visit_end_datetime",
        f.when(
            f.col("visit_end_datetime") > f.col("visit_start_datetime"), f.col("visit_end_datetime")
        ).otherwise(f.col("visit_start_datetime"))
    ).withColumn(
        "visit_order",
        f.row_number().over(
            Window.partitionBy("person_id").orderBy("visit_start_datetime", "visit_occurrence_id")
        )
    ).withColumn(
        "prev_visit_end_datetime",
        f.lag("visit_end_datetime").over(
            Window.partitionBy("person_id").orderBy("visit_order")
        )
    ).withColumn(
        "hour_diff",
        f.coalesce(
            (f.unix_timestamp("visit_start_datetime") - f.unix_timestamp("prev_visit_end_datetime")) / 3600,
            f.lit(0)
        )
    ).withColumn(
        "visit_partition",
        f.sum((f.col("hour_diff") > hour_diff_threshold).cast("int")).over(
            Window.partitionBy("person_id").orderBy("visit_order")
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
    ).withColumn(
        "is_master_visit",
        f.row_number().over(Window.partitionBy("person_id", "visit_partition").orderBy("visit_order")) == 1
    )
    visit_to_fix_folder = os.path.join(workspace_folder, f"{visit_name}_visit_to_fix")
    visit_to_fix.write.mode("overwrite").parquet(visit_to_fix_folder)
    visit_to_fix = spark.read.parquet(visit_to_fix_folder)
    # Connect all the individual visits
    master_visit = visit_to_fix.alias("visit").join(
        visit_to_fix.where(
            f.col("is_master_visit")
        ).alias("master"),
        (f.col("visit.person_id") == f.col("master.person_id"))
        & (f.col("visit.visit_partition") == f.col("master.visit_partition")),
    ).groupby(
        f.col("master.person_id").alias("person_id"),
        f.col("master.visit_partition").alias("visit_partition"),
        f.col("master.visit_occurrence_id").alias("visit_occurrence_id"),
    ).agg(
        f.min("visit.visit_start_date").alias("visit_start_date"),
        f.min("visit.visit_start_datetime").alias("visit_start_datetime"),
        f.max("visit.visit_end_date").alias("visit_end_date"),
        f.max("visit.visit_end_datetime").alias("visit_end_datetime"),
    )
    master_visit_folder = os.path.join(workspace_folder, f"{visit_name}_master_visit")
    master_visit.write.mode("overwrite").parquet(master_visit_folder)
    master_visit = spark.read.parquet(master_visit_folder)
    visit_mapping = master_visit.alias("master").join(
        visit_to_fix.alias("visit"),
        (f.col("master.person_id") == f.col("visit.person_id"))
        & (f.col("master.visit_partition") == f.col("visit.visit_partition")),
    ).where(
        f.col("master.visit_occurrence_id") != f.col("visit.visit_occurrence_id")
    ).select(
        f.col("master.person_id").alias("person_id"),
        f.col("master.visit_partition").alias("visit_partition"),
        f.col("master.visit_occurrence_id").alias("master_visit_occurrence_id"),
        f.col("visit.visit_occurrence_id").alias("visit_occurrence_id"),
    )
    visit_mapping_folder = os.path.join(workspace_folder, f"{visit_name}_visit_mapping")
    visit_mapping.write.mode("overwrite").parquet(visit_mapping_folder)
    visit_mapping = spark.read.parquet(visit_mapping_folder)
    # Update the visit_start_date(time) and visit_end_date(time)
    columns_to_update = [
        "visit_occurrence_id", "visit_start_date", "visit_end_date", "visit_start_datetime", "visit_end_datetime"
    ]
    other_columns = [column for column in visit_occurrence.columns if column not in columns_to_update]
    visit_occurrence_fixed = visit_occurrence.alias("visit").join(
        master_visit.alias("master"),
        (f.col("master.visit_occurrence_id") == f.col("visit.visit_occurrence_id")),
        "left_outer"
    ).select(
        [
            f.coalesce(f.col(f"master.{column}"), f.col(f"visit.{column}")).alias(column)
            for column in columns_to_update
        ] + [
            f.col(f"visit.{column}").alias(column)
            for column in other_columns
        ]
    )
    visit_occurrence_fixed = visit_occurrence_fixed.join(
        visit_mapping.select("visit_occurrence_id"),
        on="visit_occurrence_id",
        how="left_anti"
    )
    visit_occurrence_fixed.write.mode("overwrite").parquet(
        os.path.join(workspace_folder, f"{visit_name}_visit_occurrence_fixed")
    )
    return visit_occurrence_fixed, visit_mapping


def step_3_consolidate_outpatient_visits(
        spark: SparkSession,
        visit_occurrence: DataFrame,
        output_folder: str,
        outpatient_hour_diff_threshold: int
) -> Tuple[DataFrame, DataFrame]:
    # We need to connect the visits together
    workspace_folder = os.path.join(output_folder, "outpatient_visit_workspace")
    outpatient_visit = visit_occurrence.where(
        ~f.col("visit_concept_id").isin(9201, 262)
    ).select(
        "person_id", "visit_occurrence_id",
        "visit_start_date", "visit_start_datetime",
        "visit_end_date", "visit_end_datetime"
    )
    visit_occurrence_outpatient_visit_fixed, outpatient_visit_mapping = connect_visits_in_chronological_order(
        spark=spark,
        visit_to_fix=outpatient_visit,
        visit_occurrence=visit_occurrence,
        hour_diff_threshold=outpatient_hour_diff_threshold,
        workspace_folder=workspace_folder,
        visit_name="outpatient",
    )
    return visit_occurrence_outpatient_visit_fixed, outpatient_visit_mapping


def step_1_consolidate_inpatient_visits(
        spark: SparkSession,
        visit_occurrence: DataFrame,
        output_folder: str,
        inpatient_hour_diff_threshold: int
) -> Tuple[DataFrame, DataFrame]:
    # We need to connect the visits together
    workspace_folder = os.path.join(output_folder, "inpatient_visit_workspace")
    inpatient_visits = visit_occurrence.where(
        f.col("visit_concept_id").isin(9201, 262)
    ).select(
        "person_id", "visit_occurrence_id",
        "visit_start_date", "visit_start_datetime",
        "visit_end_date", "visit_end_datetime"
    )
    visit_occurrence_inpatient_visit_fixed, inpatient_visit_mapping = connect_visits_in_chronological_order(
        spark=spark,
        visit_to_fix=inpatient_visits,
        visit_occurrence=visit_occurrence,
        hour_diff_threshold=inpatient_hour_diff_threshold,
        workspace_folder=workspace_folder,
        visit_name="inpatient",
    )
    return visit_occurrence_inpatient_visit_fixed, inpatient_visit_mapping


def step_2_connect_outpatient_to_inpatient(
        spark: SparkSession,
        visit_occurrence: DataFrame,
        output_folder: str,
) -> Tuple[DataFrame, DataFrame]:
    # We need to connect the visits together
    workspace_folder = os.path.join(output_folder, "outpatient_to_inpatient_visit_workspace")
    inpatient_visits = visit_occurrence.where(
        f.col("visit_concept_id").isin(9201, 262)
    ).select(
        "person_id", "visit_occurrence_id",
        "visit_start_date", "visit_start_datetime",
        "visit_end_date", "visit_end_datetime"
    )
    outpatient_visits = visit_occurrence.where(
        ~f.col("visit_concept_id").isin(9201, 262)
    ).select(
        "person_id", "visit_occurrence_id",
        "visit_start_date", "visit_start_datetime",
        "visit_end_date", "visit_end_datetime"
    )
    outpatient_to_inpatient_visit_mapping = inpatient_visits.alias("in").join(
        outpatient_visits.alias("out"),
        (f.col("in.person_id") == f.col("out.person_id"))
        & (f.col("in.visit_start_datetime") < f.col("out.visit_start_datetime"))
        & (f.col("out.visit_start_datetime") < f.col("in.visit_end_datetime")),
    ).groupby(
        f.col("out.visit_occurrence_id").alias("visit_occurrence_id")
    ).agg(
        f.min("in.visit_occurrence_id").alias("master_visit_occurrence_id"),
    )
    outpatient_to_inpatient_visit_mapping_folder = os.path.join(
        workspace_folder,
        "outpatient_to_inpatient_visit_mapping"
    )
    outpatient_to_inpatient_visit_mapping.write.mode("overwrite").parquet(
        outpatient_to_inpatient_visit_mapping_folder
    )
    outpatient_to_inpatient_visit_mapping = spark.read.parquet(
        outpatient_to_inpatient_visit_mapping_folder
    )
    visit_occurrence_fixed = visit_occurrence.join(
        outpatient_to_inpatient_visit_mapping.select("visit_occurrence_id"),
        on="visit_occurrence_id",
        how="left_anti"
    )
    visit_occurrence_outpatient_to_inpatient_fix_folder = os.path.join(
        workspace_folder, "visit_occurrence_outpatient_to_inpatient_fix"
    )
    visit_occurrence_fixed.write.mode("overwrite").parquet(
        visit_occurrence_outpatient_to_inpatient_fix_folder
    )
    visit_occurrence_fixed = spark.read.parquet(visit_occurrence_outpatient_to_inpatient_fix_folder)
    return visit_occurrence_fixed, outpatient_to_inpatient_visit_mapping


def main(args):
    spark = SparkSession.builder.appName("Clean up visit_occurrence").getOrCreate()
    visit_occurrence = spark.read.parquet(os.path.join(args.input_folder, "visit_occurrence"))
    visit_occurrence_step_1, in_to_in_visit_mapping = step_1_consolidate_inpatient_visits(
        spark,
        visit_occurrence,
        output_folder=args.output_folder,
        inpatient_hour_diff_threshold=args.inpatient_hour_diff_threshold,
    )
    visit_occurrence_step_2, out_to_in_visit_mapping = step_2_connect_outpatient_to_inpatient(
        spark,
        visit_occurrence_step_1,
        output_folder=args.output_folder,
    )
    visit_occurrence_step_3, out_to_out_visit_mapping = step_3_consolidate_outpatient_visits(
        spark,
        visit_occurrence_step_2,
        output_folder=args.output_folder,
        outpatient_hour_diff_threshold=args.outpatient_hour_diff_threshold,
    )
    visit_occurrence_step_3.write.mode("overwrite").parquet(os.path.join(args.output_folder, "visit_occurrence"))
    mapping_columns = ["visit_occurrence_id", "master_visit_occurrence_id"]
    visit_mapping = in_to_in_visit_mapping.select(mapping_columns).unionByName(
        out_to_in_visit_mapping.select(mapping_columns)
    ).unionByName(out_to_out_visit_mapping.select(mapping_columns))
    visit_mapping.write.mode("overwrite").parquet(os.path.join(args.output_folder, "visit_mapping"))


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
    parser.add_argument(
        "--inpatient_hour_diff_threshold",
        dest="inpatient_hour_diff_threshold",
        action="store",
        type=int,
        default=24,
        required=False,
    )
    parser.add_argument(
        "--outpatient_hour_diff_threshold",
        dest="outpatient_hour_diff_threshold",
        action="store",
        type=int,
        default=1,
        required=False,
    )
    main(
        parser.parse_args()
    )

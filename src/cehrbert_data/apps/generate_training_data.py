import argparse
import datetime
import logging
import os
import shutil

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from cehrbert_data.const.common import (
    PERSON,
    VISIT_OCCURRENCE,
    DEATH,
    CONCEPT
)
from cehrbert_data.decorators import AttType
from cehrbert_data.utils.spark_utils import (
    create_sequence_data,
    create_sequence_data_with_att,
    preprocess_domain_table,
    extract_events_by_domain,
    validate_table_names,
    construct_artificial_visits
)
from cehrbert_data.utils.logging_utils import add_console_logging


def main(
        input_folder,
        output_folder,
        domain_table_list,
        date_filter,
        include_visit_type,
        is_new_patient_representation,
        exclude_visit_tokens,
        is_classic_bert,
        include_prolonged_stay,
        include_concept_list: bool,
        gpt_patient_sequence: bool,
        apply_age_filter: bool,
        include_death: bool,
        att_type: AttType,
        inpatient_att_type: AttType,
        exclude_demographic: bool = False,
        use_age_group: bool = False,
        with_drug_rollup: bool = True,
        include_inpatient_hour_token: bool = False,
        continue_from_events: bool = False,
        refresh_measurement: bool = False,
        aggregate_by_hour: bool = True,
        should_construct_artificial_visits: bool = False,
        duplicate_records: bool = False,
        disconnect_problem_list_records: bool = False,
):
    spark = SparkSession.builder.appName("Generate CEHR-BERT Training Data").getOrCreate()

    logger = logging.getLogger(__name__)
    logger.info(
        f"input_folder: {input_folder}\n"
        f"output_folder: {output_folder}\n"
        f"domain_table_list: {domain_table_list}\n"
        f"date_filter: {date_filter}\n"
        f"include_visit_type: {include_visit_type}\n"
        f"is_new_patient_representation: {is_new_patient_representation}\n"
        f"exclude_visit_tokens: {exclude_visit_tokens}\n"
        f"is_classic_bert: {is_classic_bert}\n"
        f"include_prolonged_stay: {include_prolonged_stay}\n"
        f"include_concept_list: {include_concept_list}\n"
        f"gpt_patient_sequence: {gpt_patient_sequence}\n"
        f"apply_age_filter: {apply_age_filter}\n"
        f"include_death: {include_death}\n"
        f"att_type: {att_type}\n"
        f"exclude_demographic: {exclude_demographic}\n"
        f"use_age_group: {use_age_group}\n"
        f"with_drug_rollup: {with_drug_rollup}\n"
        f"refresh_measurement: {refresh_measurement}\n"
        f"aggregate_by_hour: {aggregate_by_hour}\n"
        f"should_construct_artificial_visits: {should_construct_artificial_visits}\n"
        f"duplicate_records: {duplicate_records}\n"
        f"disconnect_problem_list_records: {disconnect_problem_list_records}\n"
    )

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
            aggregate_by_hour=aggregate_by_hour,
            refresh=refresh_measurement,
            persistence_folder=input_folder
        )
        if patient_ehr_events is None:
            patient_ehr_events = ehr_events
        else:
            patient_ehr_events = patient_ehr_events.unionByName(ehr_events)

    visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
    visit_occurrence = visit_occurrence.select(
        "visit_occurrence_id",
        "visit_start_date",
        "visit_start_datetime",
        "visit_end_date",
        "visit_end_datetime",
        "visit_concept_id",
        "person_id",
        "discharged_to_concept_id",
    )
    person = preprocess_domain_table(spark, input_folder, PERSON)
    birth_datetime_udf = F.coalesce("birth_datetime", F.concat("year_of_birth", F.lit("-01-01")).cast("timestamp"))
    person = person.select(
        "person_id",
        birth_datetime_udf.alias("birth_datetime"),
        "race_concept_id",
        "gender_concept_id",
    )

    visit_occurrence_person = visit_occurrence.join(person, "person_id").withColumn(
        "age",
        F.ceil(F.months_between(F.col("visit_start_date"), F.col("birth_datetime")) / F.lit(12)),
    )
    visit_occurrence_person = visit_occurrence_person.drop("birth_datetime")

    death = preprocess_domain_table(spark, input_folder, DEATH) if include_death else None

    if include_concept_list and patient_ehr_events:
        # Filter out concepts
        qualified_concepts = preprocess_domain_table(spark, input_folder, "qualified_concept_list")
        patient_ehr_events = patient_ehr_events.join(
            qualified_concepts.select("standard_concept_id"), "standard_concept_id"
        )

    patient_ehr_events = (
        patient_ehr_events.join(visit_occurrence_person, "visit_occurrence_id")
        .select(
            [patient_ehr_events[fieldName] for fieldName in patient_ehr_events.schema.fieldNames()]
            + ["visit_concept_id", "age"]
        )
        .withColumn("cohort_member_id", F.col("person_id"))
    )

    # Apply the age security measure
    # We only keep the patient records, whose corresponding age is less than 90
    if apply_age_filter:
        patient_ehr_events = patient_ehr_events.where(F.col("age") < 90)

    if not continue_from_events:
        patient_ehr_events.write.mode("overwrite").parquet(os.path.join(output_folder, "all_patient_events"))

    patient_ehr_events = spark.read.parquet(os.path.join(output_folder, "all_patient_events"))
    if should_construct_artificial_visits:
        # Construct artificial visits or re-link the visits for the problem list events
        patient_ehr_events, visit_occurrence_person = construct_artificial_visits(
            patient_ehr_events,
            visit_occurrence_person,
            spark=spark,
            persistence_folder=output_folder,
            duplicate_records=duplicate_records,
            disconnect_problem_list_records=disconnect_problem_list_records
        )
        # Update age if some of the ehr_records have been re-associated with the new visits
        patient_ehr_events = patient_ehr_events.join(
            person.select("person_id", "birth_datetime"),
            "person_id",
        ).join(
            visit_occurrence_person.select("visit_occurrence_id", "visit_start_date"), "visit_occurrence_id"
        ).withColumn(
            "age",
            F.ceil(F.months_between(F.col("visit_start_date"), F.col("birth_datetime")) / F.lit(12))
        ).drop("visit_start_date", "birth_datetime")

    if is_new_patient_representation:
        patient_sequence_data = create_sequence_data_with_att(
            patient_ehr_events,
            visit_occurrence_person,
            date_filter=date_filter,
            include_visit_type=include_visit_type,
            exclude_visit_tokens=exclude_visit_tokens,
            patient_demographic=person if gpt_patient_sequence else None,
            death=death,
            att_type=att_type,
            inpatient_att_type=inpatient_att_type,
            exclude_demographic=exclude_demographic,
            use_age_group=use_age_group,
            include_inpatient_hour_token=include_inpatient_hour_token,
            spark=spark,
            persistence_folder=output_folder,
        )
    else:
        patient_sequence_data = create_sequence_data(
            patient_ehr_events,
            date_filter=date_filter,
            include_visit_type=include_visit_type,
            classic_bert_seq=is_classic_bert,
        )

    if include_prolonged_stay:
        udf = F.when(
            F.col("visit_concept_id").isin([9201, 262, 9203]),
            F.coalesce(
                (F.datediff("visit_end_date", "visit_start_date") > 7).cast("int"),
                F.lit(0),
            ),
        ).otherwise(F.lit(0))
        visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
        visit_occurrence = (
            visit_occurrence.withColumn("prolonged_length_stay", udf)
            .select("person_id", "prolonged_length_stay")
            .withColumn(
                "prolonged_length_stay",
                F.max("prolonged_length_stay").over(Window.partitionBy("person_id")),
            )
            .distinct()
        )
        patient_sequence_data = patient_sequence_data.join(visit_occurrence, "person_id")

    patient_splits_folder = os.path.join(input_folder, "patient_splits")
    if os.path.exists(patient_splits_folder):
        patient_splits = spark.read.parquet(patient_splits_folder)
        temp_folder = os.path.join(output_folder, "patient_sequence", "temp")
        patient_sequence_data.join(
            patient_splits.select("person_id", "split"),
            "person_id"
        ).write.mode("overwrite").parquet(
            temp_folder
        )
        patient_sequence_data = spark.read.parquet(temp_folder)
        patient_sequence_data.where('split="train"').write.mode("overwrite").parquet(
            os.path.join(output_folder, "patient_sequence/train")
        )
        patient_sequence_data.where('split="test"').write.mode("overwrite").parquet(
            os.path.join(output_folder, "patient_sequence/test")
        )
        shutil.rmtree(temp_folder)
    else:
        patient_sequence_data.write.mode("overwrite").parquet(os.path.join(output_folder, "patient_sequence"))


def create_argparser():
    parser = argparse.ArgumentParser(description="Arguments for generate training data for CEHR-Bert/GPT")
    parser.add_argument(
        "--input_folder",
        dest="input_folder",
        action="store",
        help="The path for your input_folder where the raw data is",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The path for your output_folder",
        required=True,
    )
    parser.add_argument(
        "--domain_table_list",
        dest="domain_table_list",
        nargs="+",
        action="store",
        help="The list of domain tables you want to download",
        type=validate_table_names,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--date_filter",
        dest="date_filter",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"),
        action="store",
        required=False,
        default="1985-01-01",
    )
    parser.add_argument(
        "-iv",
        "--include_visit_type",
        dest="include_visit_type",
        action="store_true",
        help="Specify whether to include visit types for generating the training data",
    )
    parser.add_argument(
        "-ip",
        "--is_new_patient_representation",
        dest="is_new_patient_representation",
        action="store_true",
        help="Specify whether to generate the sequence of EHR records using the new patient " "representation",
    )
    parser.add_argument(
        "-ib",
        "--is_classic_bert_sequence",
        dest="is_classic_bert_sequence",
        action="store_true",
        help="Specify whether to generate the sequence of EHR records using the classic BERT " "sequence",
    )
    parser.add_argument(
        "-ev",
        "--exclude_visit_tokens",
        dest="exclude_visit_tokens",
        action="store_true",
        help="Specify whether or not to exclude the VS and VE tokens",
    )
    parser.add_argument(
        "--include_prolonged_length_stay",
        dest="include_prolonged_stay",
        action="store_true",
        help="Specify whether or not to include the data for the second learning objective for " "Med-BERT",
    )
    parser.add_argument("--include_concept_list", dest="include_concept_list", action="store_true")
    parser.add_argument("--gpt_patient_sequence", dest="gpt_patient_sequence", action="store_true")
    parser.add_argument("--apply_age_filter", dest="apply_age_filter", action="store_true")
    parser.add_argument("--include_death", dest="include_death", action="store_true")
    parser.add_argument("--exclude_demographic", dest="exclude_demographic", action="store_true")
    parser.add_argument("--use_age_group", dest="use_age_group", action="store_true")
    parser.add_argument("--with_drug_rollup", dest="with_drug_rollup", action="store_true")
    parser.add_argument(
        "--include_inpatient_hour_token",
        dest="include_inpatient_hour_token",
        action="store_true",
    )
    parser.add_argument(
        "--continue_from_events",
        dest="continue_from_events",
        action="store_true"
    )
    parser.add_argument(
        "--refresh_measurement",
        dest="refresh_measurement",
        action="store_true"
    )
    parser.add_argument(
        "--aggregate_by_hour",
        dest="aggregate_by_hour",
        action="store_true"
    )
    parser.add_argument(
        "--att_type",
        dest="att_type",
        action="store",
        choices=[e.value for e in AttType],
    )
    parser.add_argument(
        "--inpatient_att_type",
        dest="inpatient_att_type",
        action="store",
        choices=[e.value for e in AttType],
    )
    parser.add_argument(
        "--should_construct_artificial_visits",
        dest="should_construct_artificial_visits",
        action="store_true",
        help="Indicate whether we should construct artificial visits for "
             "the problem list records that could occur years ahead",
    )
    parser.add_argument(
        "--duplicate_records",
        dest="duplicate_records",
        action="store_true",
        help="Indicate whether we want to duplicate the problem list records when constructing artificial visits"
    )
    parser.add_argument(
        '--disconnect_problem_list_records',
        dest="disconnect_problem_list_records",
        action="store_true",
        help="Indicate whether we want to disconnect the problem list records when constructing artificial visits"
    )
    return parser


if __name__ == "__main__":
    ARGS = create_argparser().parse_args()

    # Enable logging
    add_console_logging()

    main(
        ARGS.input_folder,
        ARGS.output_folder,
        ARGS.domain_table_list,
        ARGS.date_filter,
        ARGS.include_visit_type,
        ARGS.is_new_patient_representation,
        ARGS.exclude_visit_tokens,
        ARGS.is_classic_bert_sequence,
        ARGS.include_prolonged_stay,
        ARGS.include_concept_list,
        ARGS.gpt_patient_sequence,
        ARGS.apply_age_filter,
        ARGS.include_death,
        AttType(ARGS.att_type),
        AttType(ARGS.inpatient_att_type),
        exclude_demographic=ARGS.exclude_demographic,
        use_age_group=ARGS.use_age_group,
        with_drug_rollup=ARGS.with_drug_rollup,
        include_inpatient_hour_token=ARGS.include_inpatient_hour_token,
        continue_from_events=ARGS.continue_from_events,
        refresh_measurement=ARGS.refresh_measurement,
        aggregate_by_hour=ARGS.aggregate_by_hour,
        should_construct_artificial_visits=ARGS.should_construct_artificial_visits,
        duplicate_records=ARGS.duplicate_records,
        disconnect_problem_list_records=ARGS.disconnect_problem_list_records
    )

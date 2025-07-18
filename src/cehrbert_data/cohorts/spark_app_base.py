import logging
import os
import re
import shutil
from abc import ABC
from typing import List

from pandas import to_datetime
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from cehrbert_data.decorators import AttType
from cehrbert_data.const.common import VISIT_OCCURRENCE
from cehrbert_data.utils.spark_utils import (
    create_concept_frequency_data,
    create_sequence_data,
    create_sequence_data_with_att,
    extract_ehr_records,
    preprocess_domain_table,
    construct_artificial_visits
)
from cehrbert_data.utils.vocab_utils import (
    build_ancestry_table_for,
    get_descendant_concept_ids
)
from cehrbert_data.utils.logging_utils import add_console_logging

from ..cohorts.query_builder import ENTRY_COHORT, NEGATIVE_COHORT, QueryBuilder

COHORT_TABLE_NAME = "cohort"
PERSON = "person"
OBSERVATION_PERIOD = "observation_period"
DEFAULT_DEPENDENCY = [
    "person",
    "visit_occurrence",
    "observation_period",
    "concept",
    "concept_ancestor",
    "concept_relationship",
]


def cohort_validator(required_columns_attribute):
    """
    Decorator for validating the cohort dataframe returned by build function in.

    AbstractCohortBuilderBase
    :param required_columns_attribute: attribute for storing cohort_required_columns
    in :class:`spark_apps.spark_app_base.AbstractCohortBuilderBase`
    :return:
    """

    def cohort_validator_decorator(function):
        def wrapper(self, *args, **kwargs):
            cohort = function(self, *args, **kwargs)
            required_columns = getattr(self, required_columns_attribute)
            for required_column in required_columns:
                if required_column not in cohort.columns:
                    raise AssertionError(f"{required_column} is a required column in the cohort")
            return cohort

        return wrapper

    return cohort_validator_decorator


def instantiate_dependencies(spark, input_folder, dependency_list):
    dependency_dict = dict()
    for domain_table_name in dependency_list + DEFAULT_DEPENDENCY:
        table = preprocess_domain_table(spark, input_folder, domain_table_name)
        table.createOrReplaceGlobalTempView(domain_table_name)
        dependency_dict[domain_table_name] = table
    return dependency_dict


def validate_date_folder(input_folder, table_list):
    for domain_table_name in table_list:
        parquet_file_path = os.path.join(input_folder, domain_table_name)
        if not os.path.exists(parquet_file_path):
            raise FileExistsError(f"{parquet_file_path} does not exist")


def validate_folder(folder_path):
    if not os.path.exists(folder_path):
        raise FileExistsError(f"{folder_path} does not exist")


class BaseCohortBuilder(ABC):
    cohort_required_columns = ["person_id", "index_date", "visit_occurrence_id"]

    def __init__(
            self,
            query_builder: QueryBuilder,
            input_folder: str,
            output_folder: str,
            date_lower_bound: str,
            date_upper_bound: str,
            age_lower_bound: int,
            age_upper_bound: int,
            prior_observation_period: int,
            post_observation_period: int,
            continue_job: bool = False
    ):

        self._query_builder = query_builder
        self._input_folder = input_folder
        self._output_folder = output_folder
        self._date_lower_bound = date_lower_bound
        self._date_upper_bound = date_upper_bound
        self._age_lower_bound = age_lower_bound
        self._age_upper_bound = age_upper_bound
        self._prior_observation_period = prior_observation_period
        self._post_observation_period = post_observation_period
        cohort_name = re.sub("[^a-z0-9]+", "_", self._query_builder.get_cohort_name().lower())
        self._output_data_folder = os.path.join(self._output_folder, cohort_name)
        self._continue_job = continue_job

        self.get_logger().info(
            f"query_builder: {query_builder}\n"
            f"input_folder: {input_folder}\n"
            f"output_folder: {output_folder}\n"
            f"date_lower_bound: {date_lower_bound}\n"
            f"date_upper_bound: {date_upper_bound}\n"
            f"age_lower_bound: {age_lower_bound}\n"
            f"age_upper_bound: {age_upper_bound}\n"
            f"prior_observation_period: {prior_observation_period}\n"
            f"post_observation_period: {post_observation_period}\n"
            f"continue_job: {continue_job}\n"
        )

        # Validate the age range, observation_window and prediction_window
        self._validate_integer_inputs()
        # Validate the input and output folders
        validate_folder(self._input_folder)
        validate_folder(self._output_folder)
        # Validate if the data folders exist
        validate_date_folder(self._input_folder, self._query_builder.get_dependency_list())

        self.spark = SparkSession.builder.appName(f"Generate {self._query_builder.get_cohort_name()}").getOrCreate()

        self._dependency_dict = instantiate_dependencies(
            self.spark, self._input_folder, self._query_builder.get_dependency_list()
        )

    @cohort_validator("cohort_required_columns")
    def create_cohort(self):
        """
        Create cohort.

        :return:
        """
        # Build the ancestor tables for the main query to use  if the ancestor_table_specs are
        # available
        if self._query_builder.get_ancestor_table_specs():
            for ancestor_table_spec in self._query_builder.get_ancestor_table_specs():
                func = get_descendant_concept_ids if ancestor_table_spec.is_standard else build_ancestry_table_for
                ancestor_table = func(self.spark, ancestor_table_spec.ancestor_concept_ids)
                ancestor_table.createOrReplaceGlobalTempView(ancestor_table_spec.table_name)

        # Build the dependencies for the main query to use if the dependency_queries are available
        if self._query_builder.get_dependency_queries():
            for dependency_query in self._query_builder.get_dependency_queries():
                query = dependency_query.query_template.format(**dependency_query.parameters)
                dependency_table = self.spark.sql(query)
                dependency_table.createOrReplaceGlobalTempView(dependency_query.table_name)

        # Build the dependency for the entry cohort if exists
        if self._query_builder.get_entry_cohort_query():
            entry_cohort_query = self._query_builder.get_entry_cohort_query()
            query = entry_cohort_query.query_template.format(**entry_cohort_query.parameters)
            dependency_table = self.spark.sql(query)
            dependency_table.createOrReplaceGlobalTempView(entry_cohort_query.table_name)

        # Build the negative cohort if exists
        if self._query_builder.get_negative_query():
            negative_cohort_query = self._query_builder.get_negative_query()
            query = negative_cohort_query.query_template.format(**negative_cohort_query.parameters)
            dependency_table = self.spark.sql(query)
            dependency_table.createOrReplaceGlobalTempView(negative_cohort_query.table_name)

        main_query = self._query_builder.get_query()
        cohort = self.spark.sql(main_query.query_template.format(**main_query.parameters))
        cohort.createOrReplaceGlobalTempView(main_query.table_name)

        # Post process the cohort if the post_process_queries are available
        if self._query_builder.get_post_process_queries():
            for post_query in self._query_builder.get_post_process_queries():
                cohort = self.spark.sql(post_query.query_template.format(**post_query.parameters))
                cohort.createOrReplaceGlobalTempView(main_query.table_name)

        return cohort

    def build(self):
        """Build the cohort and write the dataframe as parquet files to _output_data_folder."""

        # Check whether the cohort has been generated
        if self._continue_job and self.cohort_exists():
            return self

        cohort = self.create_cohort()

        cohort = self._apply_observation_period(cohort)

        cohort = self._add_demographics(cohort)

        cohort = cohort.where(F.col("age").between(self._age_lower_bound, self._age_upper_bound)).where(
            F.col("index_date").between(to_datetime(self._date_lower_bound), to_datetime(self._date_upper_bound))
        )

        cohort.write.mode("overwrite").parquet(self._output_data_folder)

        return self

    def cohort_exists(self) -> bool:
        try:
            self.load_cohort()
            return True
        except Exception:
            return False

    def load_cohort(self):
        return self.spark.read.parquet(self._output_data_folder)

    @cohort_validator("cohort_required_columns")
    def _apply_observation_period(self, cohort: DataFrame):
        cohort.createOrReplaceGlobalTempView("cohort")

        qualified_cohort = self.spark.sql(
            """
        SELECT
            c.*
        FROM global_temp.cohort AS c
        JOIN global_temp.observation_period AS p
            ON c.person_id = p.person_id
                AND c.index_date - INTERVAL {prior_observation_period} DAY >= p.observation_period_start_date
                AND c.index_date + INTERVAL {post_observation_period} DAY <= p.observation_period_end_date
        """.format(
                prior_observation_period=self._prior_observation_period,
                post_observation_period=self._post_observation_period,
            )
        )

        self.spark.sql(f"DROP VIEW global_temp.cohort")
        return qualified_cohort

    @cohort_validator("cohort_required_columns")
    def _add_demographics(self, cohort: DataFrame):
        return (
            cohort.join(self._dependency_dict[PERSON], "person_id")
            .withColumn("year_of_birth", F.coalesce(F.year("birth_datetime"), F.col("year_of_birth")))
            .withColumn("age", F.year("index_date") - F.col("year_of_birth"))
            .select(
                F.col("person_id"),
                F.col("age"),
                F.col("gender_concept_id"),
                F.col("race_concept_id"),
                F.col("index_date"),
                F.col("visit_occurrence_id"),
            )
            .distinct()
        )

    def _validate_integer_inputs(self):
        assert self._age_lower_bound >= 0
        assert self._age_upper_bound > 0
        assert self._age_lower_bound < self._age_upper_bound
        assert self._prior_observation_period >= 0
        assert self._post_observation_period >= 0

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)


class NestedCohortBuilder:
    def __init__(
            self,
            cohort_name: str,
            input_folder: str,
            output_folder: str,
            target_cohort: DataFrame,
            outcome_cohort: DataFrame,
            ehr_table_list: List[str],
            observation_window: int,
            hold_off_window: int,
            prediction_start_days: int,
            prediction_window: int,
            num_of_visits: int,
            num_of_concepts: int,
            patient_splits_folder: str = None,
            include_visit_type: bool = True,
            allow_measurement_only: bool = False,
            exclude_visit_tokens: bool = False,
            is_feature_concept_frequency: bool = False,
            is_roll_up_concept: bool = False,
            is_drug_roll_up_concept: bool = True,
            include_concept_list: bool = True,
            refresh_measurement: bool = False,
            aggregate_by_hour: bool = True,
            is_new_patient_representation: bool = False,
            gpt_patient_sequence: bool = False,
            classic_bert_seq: bool = False,
            is_first_time_outcome: bool = False,
            is_questionable_outcome_existed: bool = False,
            is_remove_index_prediction_starts: bool = False,
            is_prediction_window_unbounded: bool = False,
            is_observation_window_unbounded: bool = False,
            is_population_estimation: bool = False,
            att_type: AttType = AttType.CEHR_BERT,
            inpatient_att_type: AttType = AttType.MIX,
            include_inpatient_hour_token: bool = False,
            exclude_demographic: bool = True,
            use_age_group: bool = False,
            single_contribution: bool = False,
            exclude_features: bool = True,
            meds_format: bool = False,
            cache_events: bool = False,
            should_construct_artificial_visits: bool = False,
            duplicate_records: bool = False,
            disconnect_problem_list_records: bool = False,
    ):
        self._cohort_name = cohort_name
        self._input_folder = input_folder
        self._output_folder = output_folder
        self._patient_splits_folder = patient_splits_folder
        self._target_cohort = target_cohort
        self._outcome_cohort = outcome_cohort
        self._ehr_table_list = ehr_table_list
        self._observation_window = observation_window
        self._hold_off_window = hold_off_window
        self._prediction_start_days = prediction_start_days
        self._prediction_window = prediction_window
        self._num_of_visits = num_of_visits
        self._num_of_concepts = num_of_concepts
        self._is_observation_window_unbounded = is_observation_window_unbounded
        self._include_visit_type = include_visit_type
        self._exclude_visit_tokens = exclude_visit_tokens
        self._classic_bert_seq = classic_bert_seq
        self._is_feature_concept_frequency = is_feature_concept_frequency
        self._is_roll_up_concept = is_roll_up_concept
        self._is_drug_roll_up_concept = is_drug_roll_up_concept
        self._is_new_patient_representation = is_new_patient_representation
        self._gpt_patient_sequence = gpt_patient_sequence
        self._is_first_time_outcome = is_first_time_outcome
        self._is_remove_index_prediction_starts = is_remove_index_prediction_starts
        self._is_questionable_outcome_existed = is_questionable_outcome_existed
        self._is_prediction_window_unbounded = is_prediction_window_unbounded
        self._include_concept_list = include_concept_list
        self._refresh_measurement = refresh_measurement
        self._aggregate_by_hour = aggregate_by_hour
        self._allow_measurement_only = allow_measurement_only
        self._output_data_folder = os.path.join(
            self._output_folder, re.sub("[^a-z0-9]+", "_", self._cohort_name.lower())
        )
        self._is_population_estimation = is_population_estimation
        self._att_type = att_type
        self._inpatient_att_type = inpatient_att_type
        self._include_inpatient_hour_token = include_inpatient_hour_token
        self._exclude_demographic = exclude_demographic
        self._use_age_group = use_age_group
        self._single_contribution = single_contribution
        self._exclude_features = exclude_features
        self._meds_format = meds_format
        self._cache_events = cache_events
        self._should_construct_artificial_visits = should_construct_artificial_visits
        self._duplicate_records = duplicate_records
        self._disconnect_problem_list_records = disconnect_problem_list_records

        self.get_logger().info(
            f"cohort_name: {cohort_name}\n"
            f"input_folder: {input_folder}\n"
            f"output_folder: {output_folder}\n"
            f"ehr_table_list: {ehr_table_list}\n"
            f"observation_window: {observation_window}\n"
            f"prediction_start_days: {prediction_start_days}\n"
            f"prediction_window: {prediction_window}\n"
            f"hold_off_window: {hold_off_window}\n"
            f"num_of_visits: {num_of_visits}\n"
            f"num_of_concepts: {num_of_concepts}\n"
            f"include_visit_type: {include_visit_type}\n"
            f"exclude_visit_tokens: {exclude_visit_tokens}\n"
            f"allow_measurement_only: {allow_measurement_only}\n"
            f"is_feature_concept_frequency: {is_feature_concept_frequency}\n"
            f"is_roll_up_concept: {is_roll_up_concept}\n"
            f"is_drug_roll_up_concept: {is_drug_roll_up_concept}\n"
            f"is_new_patient_representation: {is_new_patient_representation}\n"
            f"gpt_patient_sequence: {gpt_patient_sequence}\n"
            f"is_first_time_outcome: {is_first_time_outcome}\n"
            f"is_questionable_outcome_existed: {is_questionable_outcome_existed}\n"
            f"is_remove_index_prediction_starts: {is_remove_index_prediction_starts}\n"
            f"is_prediction_window_unbounded: {is_prediction_window_unbounded}\n"
            f"include_concept_list: {include_concept_list}\n"
            f"refresh_measurement: {refresh_measurement}\n"
            f"aggregate_by_hour: {aggregate_by_hour}\n"
            f"is_observation_window_unbounded: {is_observation_window_unbounded}\n"
            f"is_population_estimation: {is_population_estimation}\n"
            f"att_type: {att_type}\n"
            f"inpatient_att_type: {inpatient_att_type}\n"
            f"include_inpatient_hour_token: {include_inpatient_hour_token}\n"
            f"exclude_demographic: {exclude_demographic}\n"
            f"use_age_group: {use_age_group}\n"
            f"single_contribution: {single_contribution}\n"
            f"extract_features: {exclude_features}\n"
            f"meds_format: {meds_format}\n"
            f"cache_events: {cache_events}\n"
            f"should_construct_artificial_visits: {should_construct_artificial_visits}\n"
            f"duplicate_records: {duplicate_records}\n"
            f"disconnect_problem_list_records: {disconnect_problem_list_records}\n"
        )

        self.spark = SparkSession.builder.appName(f"Generate {self._cohort_name}").getOrCreate()
        self._dependency_dict = instantiate_dependencies(self.spark, self._input_folder, DEFAULT_DEPENDENCY)

        # Validate the input and output folders
        validate_folder(self._input_folder)
        validate_folder(self._output_folder)
        # Validate if the data folders exist
        validate_date_folder(self._input_folder, self._ehr_table_list)

    def build(self):
        self._target_cohort.createOrReplaceGlobalTempView("target_cohort")
        self._outcome_cohort.createOrReplaceGlobalTempView("outcome_cohort")

        prediction_start_days = self._prediction_start_days
        prediction_window = self._prediction_window

        if self._is_first_time_outcome:
            target_cohort = self.spark.sql(
                """
            SELECT
                t.person_id AS cohort_member_id,
                t.*
            FROM global_temp.target_cohort AS t
            LEFT JOIN global_temp.{entry_cohort} AS o
                ON t.person_id = o.person_id
                AND t.index_date + INTERVAL {prediction_start_days} DAY > o.index_date
            WHERE o.person_id IS NULL
            """.format(
                    entry_cohort=ENTRY_COHORT,
                    prediction_start_days=prediction_start_days,
                )
            )
            target_cohort.createOrReplaceGlobalTempView("target_cohort")

        if self._is_questionable_outcome_existed:
            target_cohort = self.spark.sql(
                """
            SELECT
                t.*
            FROM global_temp.target_cohort AS t
            LEFT JOIN global_temp.{questionnation_outcome_cohort} AS o
                ON t.person_id = o.person_id
            WHERE o.person_id IS NULL
            """.format(
                    questionnation_outcome_cohort=NEGATIVE_COHORT
                )
            )
            target_cohort.createOrReplaceGlobalTempView("target_cohort")
        if self._is_remove_index_prediction_starts:
            # Remove the patients whose outcome date lies between index_date and index_date +
            # prediction_start_days
            target_cohort = self.spark.sql(
                """
            SELECT DISTINCT
                t.*
            FROM global_temp.target_cohort AS t
            LEFT JOIN global_temp.outcome_cohort AS exclusion
                ON t.person_id = exclusion.person_id
                    AND exclusion.index_date BETWEEN t.index_date
                        AND t.index_date + INTERVAL {prediction_start_days} DAY
            WHERE exclusion.person_id IS NULL
            """.format(
                    prediction_start_days=max(prediction_start_days - 1, 0)
                )
            )
            target_cohort.createOrReplaceGlobalTempView("target_cohort")

        if self._is_prediction_window_unbounded:
            query_template = """
            SELECT DISTINCT
                t.*,
                o.index_date as outcome_date,
                CAST(ISNOTNULL(o.person_id) AS INT) AS label
            FROM global_temp.target_cohort AS t
            LEFT JOIN global_temp.outcome_cohort AS o
                ON t.person_id = o.person_id
                    AND o.index_date >= t.index_date + INTERVAL {prediction_start_days} DAY
            """
        else:
            query_template = """
            SELECT DISTINCT
                t.*,
                o.index_date as outcome_date,
                CAST(ISNOTNULL(o.person_id) AS INT) AS label
            FROM global_temp.target_cohort AS t
            LEFT JOIN global_temp.observation_period AS op
                ON t.person_id = op.person_id
                    AND t.index_date + INTERVAL {prediction_window} DAY <= op.observation_period_end_date
            LEFT JOIN global_temp.outcome_cohort AS o
                ON t.person_id = o.person_id
                    AND o.index_date BETWEEN t.index_date + INTERVAL {prediction_start_days} DAY
                        AND t.index_date + INTERVAL {prediction_window} DAY
            WHERE op.person_id IS NOT NULL OR o.person_id IS NOT NULL
            """

        cohort_member_id_udf = F.dense_rank().over(Window.orderBy("person_id", "index_date", "visit_occurrence_id"))
        cohort = self.spark.sql(
            query_template.format(
                prediction_start_days=prediction_start_days,
                prediction_window=prediction_window,
            )
        ).withColumn("cohort_member_id", cohort_member_id_udf)

        # Keep one record in case that there are multiple samples generated for the same index_date.
        # This should not happen in theory, this is really just a safeguard
        row_rank = F.row_number().over(
            Window.partitionBy("person_id", "cohort_member_id", "index_date").orderBy(F.desc("label"))
        )
        cohort = cohort.withColumn("row_rank", row_rank).where("row_rank == 1").drop("row_rank")

        # We only allow the patient to contribute once to the dataset
        # If the patient has any positive outcomes, we will take the most recent positive outcome,
        # otherwise we will take the most recent negative outcome
        if self._single_contribution:
            record_rank = F.row_number().over(
                Window.partitionBy("person_id").orderBy(F.desc("label"), F.desc("index_date"))
            )
            cohort = cohort.withColumn("record_rank", record_rank).where("record_rank == 1").drop("record_rank")

        if self._exclude_features:
            # We need to remove the samples that do not have any records in the prediction window
            cohort = self.filter_cohort_with_ehr_records(cohort)
        else:
            ehr_records_for_cohorts = self.extract_ehr_records_for_cohort(cohort)
            cohort = (
                cohort.join(ehr_records_for_cohorts, ["person_id", "cohort_member_id"])
                .where(F.col("num_of_visits") >= self._num_of_visits)
                .where(F.col("num_of_concepts") >= self._num_of_concepts)
            )

        person_id_column = "person_id"
        index_date_column = "index_date"
        if self._meds_format:
            cohort = cohort.withColumnRenamed(
                "person_id", "subject_id"
            ).withColumnRenamed(
                "index_date", "prediction_time"
            ).withColumnRenamed(
                "label", "boolean_value"
            ).withColumn(
                "prediction_time", F.to_timestamp("prediction_time")
            ).withColumn(
                "boolean_value", F.col("boolean_value").cast("boolean")
            )
            person_id_column = "subject_id"
            index_date_column = "prediction_time"

        if self._is_prediction_window_unbounded:
            observation_period = self._dependency_dict[OBSERVATION_PERIOD]
            # Add time_to_event
            cohort = cohort.join(
                observation_period.select("person_id", "observation_period_end_date"),
                cohort[person_id_column] == observation_period["person_id"]
            ).select(
                [cohort[c] for c in cohort.columns] +
                [observation_period["observation_period_end_date"]]
            ).withColumn(
                "study_end_date",
                F.coalesce(F.col("outcome_date"), F.col("observation_period_end_date"))
            ).drop(
                "observation_period_end_date"
            )
        else:
            # Add time_to_event
            cohort = cohort.withColumn(
                "study_end_date",
                F.coalesce(
                    F.col("outcome_date"),
                    F.expr(f"{index_date_column} + INTERVAL {self._prediction_window} DAYS")
                )
            )
        cohort = cohort.withColumn("time_to_event", F.datediff("study_end_date", index_date_column))

        # if patient_splits is provided, we will
        if self._patient_splits_folder:
            patient_splits = self.spark.read.parquet(self._patient_splits_folder)
            cohort.alias("cohort").join(
                patient_splits.alias("split"),
                F.col(f"cohort.{person_id_column}") == F.col("split.person_id")
            ).select(
                [F.col(f"cohort.{c}").alias(c) for c in cohort.columns] + [F.col("split.split").alias("split")]
            ).orderBy(person_id_column, index_date_column).write.mode(
                "overwrite"
            ).parquet(
                os.path.join(self._output_data_folder, "temp")
            )
            # Reload the data from the disk
            cohort = self.spark.read.parquet(os.path.join(self._output_data_folder, "temp"))
            cohort.where('split="train"').write.mode("overwrite").parquet(
                os.path.join(self._output_data_folder, "train")
            )
            cohort.where('split="test"').write.mode("overwrite").parquet(os.path.join(self._output_data_folder, "test"))
            shutil.rmtree(os.path.join(self._output_data_folder, "temp"))
        else:
            cohort.orderBy(person_id_column, index_date_column).write.mode("overwrite").parquet(
                os.path.join(self._output_data_folder, "data")
            )

    def _create_ehr_record_filter(self):
        # Only allow the data records that occurred between the index date and the prediction window
        if self._is_population_estimation:
            if self._is_prediction_window_unbounded:
                record_window_filter = F.col("ehr.datetime") <= F.current_timestamp()
            else:
                record_window_filter = (
                        F.col("ehr.datetime") <=
                        F.expr(f"cohort.index_date - INTERVAL {self._hold_off_window} DAYS + INTERVAL 0.1 SECOND")
                )
        else:
            if self._is_observation_window_unbounded:
                record_window_filter = (
                        F.col("ehr.datetime")
                        <= F.expr(
                    f"cohort.index_date - INTERVAL {self._hold_off_window} DAYS + INTERVAL 0.1 SECOND")
                )
            else:
                record_window_filter = F.col("ehr.datetime").between(
                    F.expr(f"cohort.index_date - INTERVAL {self._observation_window + self._hold_off_window} DAYS"),
                    F.expr(f"cohort.index_date - INTERVAL {self._hold_off_window} DAYS + INTERVAL 0.1 SECOND"),
                )
        return record_window_filter

    def filter_cohort_with_ehr_records(self, cohort: DataFrame) -> DataFrame:
        # Extract all ehr records for the patients
        ehr_records = extract_ehr_records(
            spark=self.spark,
            input_folder=self._input_folder,
            domain_table_list=self._ehr_table_list,
            include_visit_type=self._include_visit_type,
            with_diagnosis_rollup=self._is_roll_up_concept,
            with_drug_rollup=self._is_drug_roll_up_concept,
            include_concept_list=self._include_concept_list,
            refresh_measurement=self._refresh_measurement,
            aggregate_by_hour=self._aggregate_by_hour,
            keep_orphan_records=self._should_construct_artificial_visits,
        )

        cohort = cohort.alias("cohort").join(
            ehr_records.select("person_id", "datetime").distinct().alias("ehr"),
            F.col("ehr.person_id") == F.col("cohort.person_id")
        ).where(
            self._create_ehr_record_filter()
        ).select(
            [F.col("cohort." + field_name) for field_name in cohort.schema.fieldNames()]
        ).distinct()

        return cohort

    def extract_ehr_records_for_cohort(self, cohort: DataFrame) -> DataFrame:
        """
        Create the patient sequence based on the observation window for the given cohort.

        :param cohort:
        :return:
        """
        # Extract all ehr records for the patients
        ehr_records = extract_ehr_records(
            spark=self.spark,
            input_folder=self._input_folder,
            domain_table_list=self._ehr_table_list,
            include_visit_type=self._include_visit_type,
            with_diagnosis_rollup=self._is_roll_up_concept,
            with_drug_rollup=self._is_drug_roll_up_concept,
            include_concept_list=self._include_concept_list,
            refresh_measurement=self._refresh_measurement,
            aggregate_by_hour=self._aggregate_by_hour,
            keep_orphan_records=self._should_construct_artificial_visits,
        )

        if self._cache_events:
            all_patient_events_dir = os.path.join(self._output_data_folder, "all_patient_events")
            ehr_records.write.mode("overwrite").parquet(
                all_patient_events_dir
            )
            ehr_records = self.spark.read.parquet(
                all_patient_events_dir
            )

        if self._should_construct_artificial_visits:
            person = self._dependency_dict[PERSON]
            birthdate_udf = F.coalesce(
                "birth_datetime",
                F.concat("year_of_birth", F.lit("-01-01")).cast("timestamp"),
            )
            patient_demographic = person.select(
                "person_id",
                birthdate_udf.alias("birth_datetime"),
            )
            ehr_records, visit_occurrence_with_artificial_visits = construct_artificial_visits(
                ehr_records,
                self._dependency_dict[VISIT_OCCURRENCE],
                spark=self.spark if self._cache_events else None,
                persistence_folder=self._output_data_folder if self._cache_events else None,
                duplicate_records=self._duplicate_records,
                disconnect_problem_list_records=self._disconnect_problem_list_records,
            )

            # Update age if some of the ehr_records have been re-associated with the new visits
            ehr_records = ehr_records.join(
                patient_demographic,
                "person_id"
            ).join(
                visit_occurrence_with_artificial_visits.select(
                    "visit_occurrence_id", "visit_start_date"
                ), "visit_occurrence_id"
            ).withColumn(
                "age",
                F.ceil(F.months_between(F.col("visit_start_date"), F.col("birth_datetime")) / F.lit(12))
            ).drop("visit_start_date", "birth_datetime")

            # Refresh the dependency
            self._dependency_dict[VISIT_OCCURRENCE] = visit_occurrence_with_artificial_visits

        # Duplicate the records for cohorts that allow multiple entries
        ehr_records = ehr_records.alias("ehr").join(
            cohort.alias("cohort"), F.col("ehr.person_id") == F.col("cohort.person_id")
        ).select(
            [F.col("ehr." + col) for col in ehr_records.columns] + [F.col("cohort.cohort_member_id")]
        ).selectExpr("*")

        # Somehow the dataframe join does not work without using the alias
        cohort_ehr_records = ehr_records.alias("ehr").join(
            cohort.alias("cohort"),
            (F.col("ehr.person_id") == F.col("cohort.person_id")) &
            (F.col("ehr.cohort_member_id") == F.col("cohort.cohort_member_id")),
        ).where(
            self._create_ehr_record_filter()
        ).select(
            [F.col("ehr." + field_name) for field_name in ehr_records.schema.fieldNames()]
        )

        if self._is_feature_concept_frequency:
            return create_concept_frequency_data(cohort_ehr_records, None)

        if self._is_new_patient_representation:
            birthdate_udf = F.coalesce(
                "birth_datetime",
                F.concat("year_of_birth", F.lit("-01-01")).cast("timestamp"),
            )
            patient_demographic = self._dependency_dict[PERSON].select(
                "person_id",
                birthdate_udf.alias("birth_datetime"),
                "race_concept_id",
                "gender_concept_id",
            )

            age_udf = F.ceil(F.months_between(F.col("visit_start_date"), F.col("birth_datetime")) / F.lit(12))
            visit_occurrence_person = (
                self._dependency_dict[VISIT_OCCURRENCE]
                .join(patient_demographic, "person_id")
                .withColumn("age", age_udf)
                .drop("birth_datetime")
            )

            return create_sequence_data_with_att(
                cohort_ehr_records,
                visit_occurrence=visit_occurrence_person,
                include_visit_type=self._include_visit_type,
                exclude_visit_tokens=self._exclude_visit_tokens,
                patient_demographic=(patient_demographic if self._gpt_patient_sequence else None),
                att_type=self._att_type,
                inpatient_att_type=self._inpatient_att_type,
                exclude_demographic=self._exclude_demographic,
                use_age_group=self._use_age_group,
                include_inpatient_hour_token=self._include_inpatient_hour_token,
                spark=self.spark if self._cache_events else None,
                persistence_folder=self._output_data_folder if self._cache_events else None,
                cohort_index=cohort.select("person_id", "cohort_member_id", "index_date")
            )

        return create_sequence_data(
            cohort_ehr_records,
            date_filter=None,
            include_visit_type=self._include_visit_type,
            classic_bert_seq=self._classic_bert_seq,
        )

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)


def create_prediction_cohort(
        spark_args,
        target_query_builder: QueryBuilder,
        outcome_query_builder: QueryBuilder,
        ehr_table_list,
):
    """
    TODO.

    :param spark_args:
    :param target_query_builder:
    :param outcome_query_builder:
    :param ehr_table_list:
    :return:
    """
    # Add logging to spark application output when enable_logging is set to True
    if spark_args.enable_logging:
        add_console_logging()
    # Generate the target cohort
    target_cohort = (
        BaseCohortBuilder(
            query_builder=target_query_builder,
            input_folder=spark_args.input_folder,
            output_folder=spark_args.output_folder,
            date_lower_bound=spark_args.date_lower_bound,
            date_upper_bound=spark_args.date_upper_bound,
            age_lower_bound=spark_args.age_lower_bound,
            age_upper_bound=spark_args.age_upper_bound,
            prior_observation_period=spark_args.observation_window + spark_args.hold_off_window,
            post_observation_period=0,
            continue_job=spark_args.continue_job
        )
        .build()
        .load_cohort()
    )

    # Generate the outcome cohort
    outcome_cohort = (
        BaseCohortBuilder(
            query_builder=outcome_query_builder,
            input_folder=spark_args.input_folder,
            output_folder=spark_args.output_folder,
            date_lower_bound=spark_args.date_lower_bound,
            date_upper_bound=spark_args.date_upper_bound,
            age_lower_bound=spark_args.age_lower_bound,
            age_upper_bound=spark_args.age_upper_bound,
            prior_observation_period=0,
            post_observation_period=0,
            continue_job=spark_args.continue_job,
        )
        .build()
        .load_cohort()
    )

    NestedCohortBuilder(
        cohort_name=spark_args.cohort_name,
        input_folder=spark_args.input_folder,
        output_folder=spark_args.output_folder,
        patient_splits_folder=spark_args.patient_splits_folder,
        target_cohort=target_cohort,
        outcome_cohort=outcome_cohort,
        ehr_table_list=ehr_table_list,
        observation_window=spark_args.observation_window,
        hold_off_window=spark_args.hold_off_window,
        prediction_start_days=spark_args.prediction_start_days,
        prediction_window=spark_args.prediction_window,
        num_of_visits=spark_args.num_of_visits,
        num_of_concepts=spark_args.num_of_concepts,
        include_visit_type=spark_args.include_visit_type,
        exclude_visit_tokens=spark_args.exclude_visit_tokens,
        allow_measurement_only=spark_args.allow_measurement_only,
        is_feature_concept_frequency=spark_args.is_feature_concept_frequency,
        is_roll_up_concept=spark_args.is_roll_up_concept,
        is_drug_roll_up_concept=spark_args.is_drug_roll_up_concept,
        include_concept_list=spark_args.include_concept_list,
        refresh_measurement=spark_args.refresh_measurement,
        aggregate_by_hour=spark_args.aggregate_by_hour,
        is_new_patient_representation=spark_args.is_new_patient_representation,
        gpt_patient_sequence=spark_args.gpt_patient_sequence,
        classic_bert_seq=spark_args.classic_bert_seq,
        is_first_time_outcome=spark_args.is_first_time_outcome,
        # If the outcome negative query exists, that means we need to remove those questionable
        # outcomes from the target cohort
        is_questionable_outcome_existed=outcome_query_builder.get_negative_query() is not None,
        is_prediction_window_unbounded=spark_args.is_prediction_window_unbounded,
        # Do we want to remove those records whose outcome occur between index_date
        # and the start of the prediction window
        is_remove_index_prediction_starts=spark_args.is_remove_index_prediction_starts,
        is_observation_window_unbounded=spark_args.is_observation_window_unbounded,
        is_population_estimation=spark_args.is_population_estimation,
        att_type=AttType(spark_args.att_type),
        inpatient_att_type=AttType(spark_args.inpatient_att_type),
        include_inpatient_hour_token=spark_args.include_inpatient_hour_token,
        exclude_demographic=spark_args.exclude_demographic,
        use_age_group=spark_args.use_age_group,
        single_contribution=spark_args.single_contribution,
        exclude_features=spark_args.exclude_features,
        meds_format=spark_args.meds_format,
        cache_events=spark_args.cache_events,
        should_construct_artificial_visits=spark_args.should_construct_artificial_visits,
        duplicate_records=spark_args.duplicate_records,
        disconnect_problem_list_records=spark_args.disconnect_problem_list_records,
    ).build()

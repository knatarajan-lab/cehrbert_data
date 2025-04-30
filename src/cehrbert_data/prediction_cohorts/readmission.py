from cehrbert_data.cohorts.query_builder import QueryBuilder, QuerySpec
from cehrbert_data.cohorts.spark_app_base import create_prediction_cohort
from cehrbert_data.const.common import (
    CONDITION_OCCURRENCE,
    DRUG_EXPOSURE,
    PERSON,
    PROCEDURE_OCCURRENCE,
    VISIT_OCCURRENCE,
)
from cehrbert_data.utils.spark_parse_args import create_spark_args

FIRST_HOSPITALIZATION_QUERY = """
SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    COALESCE(v.visit_end_datetime, CAST(v.visit_end_date AS TIMESTAMP)) AS index_date
FROM global_temp.visit_occurrence AS v
WHERE v.visit_concept_id IN (9201, 262, 8971, 8920) --inpatient, er-inpatient
"""

SECOND_HOSPITALIZATION_QUERY = """
SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    COALESCE(v.visit_start_datetime, CAST(v.visit_start_date AS TIMESTAMP)) AS index_date
FROM global_temp.visit_occurrence AS v
WHERE v.visit_concept_id IN (9201, 262, 8971, 8920) --inpatient, er-inpatient
"""

FIRST_HOSPITALIZATION_COHORT = "first_hospitalization"
SECOND_HOSPITALIZATION_COHORT = "second_hospitalization"
DEPENDENCY_LIST = [PERSON, CONDITION_OCCURRENCE, VISIT_OCCURRENCE]

DOMAIN_TABLE_LIST = [CONDITION_OCCURRENCE, DRUG_EXPOSURE, PROCEDURE_OCCURRENCE]


def main(spark_args):
    target_query = QuerySpec(
        table_name=FIRST_HOSPITALIZATION_COHORT,
        query_template=FIRST_HOSPITALIZATION_QUERY,
        parameters={"date_lower_bound": spark_args.date_lower_bound},
    )
    target_querybuilder = QueryBuilder(
        cohort_name=FIRST_HOSPITALIZATION_COHORT,
        dependency_list=DEPENDENCY_LIST,
        query=target_query,
    )
    second_hospitalization_query = QuerySpec(
        table_name=SECOND_HOSPITALIZATION_COHORT,
        query_template=SECOND_HOSPITALIZATION_QUERY,
        parameters={},
    )
    outcome_querybuilder = QueryBuilder(
        cohort_name=SECOND_HOSPITALIZATION_COHORT,
        dependency_list=DEPENDENCY_LIST,
        query=second_hospitalization_query,
    )

    ehr_table_list = spark_args.ehr_table_list if spark_args.ehr_table_list else DOMAIN_TABLE_LIST

    create_prediction_cohort(spark_args, target_querybuilder, outcome_querybuilder, ehr_table_list)


if __name__ == "__main__":
    main(create_spark_args())

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

HEART_FAILURE_HOSPITALIZATION_QUERY = """
WITH hf_concepts AS (
    SELECT DISTINCT
        descendant_concept_id AS concept_id
    FROM global_temp.concept_ancestor AS ca
    WHERE ca.ancestor_concept_id = 316139
)

SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    COALESCE(v.visit_end_datetime, v.visit_end_date) AS index_date
FROM global_temp.visit_occurrence AS v
JOIN global_temp.condition_occurrence AS co
    ON v.visit_occurrence_id = co.visit_occurrence_id
JOIN hf_concepts AS hf
    ON co.condition_concept_id = hf.concept_id
WHERE v.visit_concept_id IN (9201, 262, 8971, 8920) --inpatient, er-inpatient
    AND v.discharged_to_concept_id NOT IN (4216643, 44814650, 8717, 8970, 8971) -- TBD
    --AND v.discharge_to_concept_id IN (8536, 8863, 4161979) -- Home, Skilled Nursing Facility, and Patient discharged alive
    AND v.visit_start_date <= co.condition_start_date
    AND v.visit_end_date >= '{date_lower_bound}'
"""

HOSPITALIZATION_QUERY = """
SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    COALESCE(v.visit_start_datetime, v.visit_start_date) AS index_date
FROM global_temp.visit_occurrence AS v
WHERE v.visit_concept_id IN (9201, 262, 8971, 8920) --inpatient, er-inpatient
"""

HF_HOSPITALIZATION_COHORT = "hf_hospitalization"
HOSPITALIZATION_COHORT = "hospitalization"
DEPENDENCY_LIST = [PERSON, CONDITION_OCCURRENCE, VISIT_OCCURRENCE]

DOMAIN_TABLE_LIST = [CONDITION_OCCURRENCE, DRUG_EXPOSURE, PROCEDURE_OCCURRENCE]


def main(spark_args):
    hf_inpatient_target_query = QuerySpec(
        table_name=HF_HOSPITALIZATION_COHORT,
        query_template=HEART_FAILURE_HOSPITALIZATION_QUERY,
        parameters={"date_lower_bound": spark_args.date_lower_bound},
    )

    hf_inpatient_target_querybuilder = QueryBuilder(
        cohort_name=HF_HOSPITALIZATION_COHORT,
        dependency_list=DEPENDENCY_LIST,
        query=hf_inpatient_target_query,
    )

    hospitalization_query = QuerySpec(
        table_name=HOSPITALIZATION_COHORT,
        query_template=HOSPITALIZATION_QUERY,
        parameters={},
    )
    hospitalization = QueryBuilder(
        cohort_name=HOSPITALIZATION_COHORT,
        dependency_list=DEPENDENCY_LIST,
        query=hospitalization_query,
    )

    ehr_table_list = spark_args.ehr_table_list if spark_args.ehr_table_list else DOMAIN_TABLE_LIST

    create_prediction_cohort(spark_args, hf_inpatient_target_querybuilder, hospitalization, ehr_table_list)


if __name__ == "__main__":
    main(create_spark_args())

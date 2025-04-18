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

COPD_HOSPITALIZATION_QUERY = """
WITH copd_conditions AS (
    SELECT DISTINCT
        descendant_concept_id AS concept_id
    FROM global_temp.concept_ancestor AS ca
    WHERE ca.ancestor_concept_id in (255573, 258780)
)

SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    COALESCE(v.visit_end_datetime, v.visit_end_date) AS index_date
FROM global_temp.visit_occurrence AS v
JOIN global_temp.condition_occurrence AS co
    ON v.visit_occurrence_id = co.visit_occurrence_id
JOIN copd_conditions AS copd
    ON co.condition_concept_id = copd.concept_id
WHERE v.visit_concept_id IN (9201, 262) --inpatient, er-inpatient
    AND v.discharged_to_concept_id = 8536 --discharge to home
    AND v.visit_start_date <= co.condition_start_date
"""

HOSPITALIZATION_QUERY = """
SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    COALESCE(v.visit_start_datetime, v.visit_start_date) AS index_date
FROM global_temp.visit_occurrence AS v
WHERE v.visit_concept_id IN (9201, 262) --inpatient, er-inpatient
"""

COPD_HOSPITALIZATION_COHORT = "copd_readmission"
HOSPITALIZATION_COHORT = "hospitalization"
DEPENDENCY_LIST = [PERSON, CONDITION_OCCURRENCE, VISIT_OCCURRENCE]
DOMAIN_TABLE_LIST = [CONDITION_OCCURRENCE, DRUG_EXPOSURE, PROCEDURE_OCCURRENCE]


def main(spark_args):
    copd_inpatient_query = QuerySpec(
        table_name=COPD_HOSPITALIZATION_COHORT,
        query_template=COPD_HOSPITALIZATION_QUERY,
        parameters={},
    )
    copd_inpatient = QueryBuilder(
        cohort_name=COPD_HOSPITALIZATION_COHORT,
        dependency_list=DEPENDENCY_LIST,
        query=copd_inpatient_query,
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

    create_prediction_cohort(spark_args, copd_inpatient, hospitalization, DOMAIN_TABLE_LIST)


if __name__ == "__main__":
    main(create_spark_args())

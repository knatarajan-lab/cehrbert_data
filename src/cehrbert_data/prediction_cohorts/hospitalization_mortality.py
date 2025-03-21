from cehrbert_data.cohorts.query_builder import QueryBuilder, QuerySpec
from cehrbert_data.cohorts.spark_app_base import create_prediction_cohort
from cehrbert_data.const.common import CONDITION_OCCURRENCE, DRUG_EXPOSURE, PROCEDURE_OCCURRENCE, VISIT_OCCURRENCE
from cehrbert_data.utils.spark_parse_args import create_spark_args

DEPENDENCY_LIST = [VISIT_OCCURRENCE]
DOMAIN_TABLE_LIST = [CONDITION_OCCURRENCE, PROCEDURE_OCCURRENCE, DRUG_EXPOSURE]

HOSPITALIZATION_QUERY = """
SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    v.index_date,
    v.expired
FROM
(
    SELECT
        v.person_id,
        v.visit_occurrence_id,
        v.visit_start_datetime + interval 24 hours AS index_date,
        CASE
            WHEN v.discharged_to_concept_id == 4216643 THEN 1
            ELSE 0
        END AS expired,
        ROW_NUMBER() OVER(PARTITION BY v.person_id ORDER BY COALESCE(v.visit_end_datetime, v.visit_end_date) DESC) AS rn
    FROM global_temp.visit_occurrence AS v
    WHERE v.visit_concept_id IN (9201, 262) --inpatient, er-inpatient
        AND v.visit_end_datetime IS NOT NULL
        -- the inpatient visit duration is greater than 48 hours
        AND (unix_timestamp(v.visit_end_datetime) - unix_timestamp(v.visit_start_datetime)) / 3600  > 48 
) AS v
    WHERE v.rn = 1 AND v.index_date >= '{date_lower_bound}'
"""

MORTALITY_QUERY = """
SELECT DISTINCT
    v.person_id,
    v.visit_occurrence_id,
    v.index_date AS index_date
FROM global_temp.{target_table_name} AS v
WHERE expired = 1
"""

HOSPITALIZATION_TARGET_COHORT = "hospitalization_target"
MORTALITY_COHORT = "hospitalization_mortality"

if __name__ == "__main__":
    spark_args = create_spark_args()
    ehr_table_list = spark_args.ehr_table_list if spark_args.ehr_table_list else DOMAIN_TABLE_LIST

    hospitalization_target_query = QuerySpec(
        table_name=HOSPITALIZATION_TARGET_COHORT,
        query_template=HOSPITALIZATION_QUERY,
        parameters={"date_lower_bound": spark_args.date_lower_bound},
    )

    hospitalization_querybuilder = QueryBuilder(
        cohort_name=HOSPITALIZATION_TARGET_COHORT,
        dependency_list=DEPENDENCY_LIST,
        query=hospitalization_target_query,
    )

    hospitalization_mortality_query = QuerySpec(
        table_name=MORTALITY_COHORT,
        query_template=MORTALITY_QUERY,
        parameters={"target_table_name": HOSPITALIZATION_TARGET_COHORT},
    )
    hospitalization_mortality_querybuilder = QueryBuilder(
        cohort_name=MORTALITY_COHORT,
        dependency_list=DEPENDENCY_LIST,
        query=hospitalization_mortality_query,
    )

    create_prediction_cohort(
        spark_args,
        hospitalization_querybuilder,
        hospitalization_mortality_querybuilder,
        ehr_table_list,
    )

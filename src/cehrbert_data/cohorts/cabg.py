from cehrbert_data.cohorts.query_builder import AncestorTableSpec, QueryBuilder, QuerySpec
from cehrbert_data.const.common import PERSON, PROCEDURE_OCCURRENCE, VISIT_OCCURRENCE

COHORT_QUERY_TEMPLATE = """
SELECT DISTINCT
    c.person_id,
    c.index_date,
    c.visit_occurrence_id
FROM
(
    SELECT DISTINCT
        vo.person_id,
        vo.visit_occurrence_id,
        CASE
            WHEN po.procedure_datetime < vo.visit_start_datetime AND datediff(vo.visit_start_datetime, po.procedure_datetime) <= 1 
                THEN to_timestamp(concat(date_format(vo.visit_start_date, 'yyyy-MM-dd'), ' 23:59:00'), 'yyyy-MM-dd HH:mm:ss')
            ELSE to_timestamp(concat(date_format(po.procedure_date, 'yyyy-MM-dd'), ' 23:59:00'), 'yyyy-MM-dd HH:mm:ss')
        END AS index_date,
        ROW_NUMBER() OVER(
            PARTITION BY po.person_id
            ORDER BY po.procedure_datetime,
                po.procedure_date,
                po.visit_occurrence_id
        ) as r_number
    FROM global_temp.procedure_occurrence AS po
    JOIN global_temp.visit_occurrence AS vo
        ON po.visit_occurrence_id = vo.visit_occurrence_id
    WHERE EXISTS (
        SELECT 1
        FROM global_temp.{cabg_concept_table} AS ie
        WHERE po.procedure_concept_id = ie.concept_id
    )
) c
WHERE c.index_date >= '{date_lower_bound}' AND c.r_number = 1
"""

DEFAULT_COHORT_NAME = "cabg"
DEPENDENCY_LIST = [PERSON, VISIT_OCCURRENCE, PROCEDURE_OCCURRENCE]
CABG_INCLUSION_TABLE = "CABG"
CABG_CONCEPTS = [
    43528001,
    43528003,
    43528004,
    43528002,
    4305852,
    4168831,
    2107250,
    2107216,
    2107222,
    2107231,
    4336464,
    4231998,
    4284104,
    2100873,
]


def query_builder(spark_args):
    query = QuerySpec(
        table_name=DEFAULT_COHORT_NAME,
        query_template=COHORT_QUERY_TEMPLATE,
        parameters={
            "cabg_concept_table": CABG_INCLUSION_TABLE,
            "date_lower_bound": spark_args.date_lower_bound,
        },
    )

    ancestor_table_specs = [
        AncestorTableSpec(
            table_name=CABG_INCLUSION_TABLE,
            ancestor_concept_ids=CABG_CONCEPTS,
            is_standard=True,
        )
    ]
    return QueryBuilder(
        cohort_name=DEFAULT_COHORT_NAME,
        dependency_list=DEPENDENCY_LIST,
        query=query,
        ancestor_table_specs=ancestor_table_specs,
    )

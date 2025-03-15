from cehrbert_data.cohorts.query_builder import AncestorTableSpec, QueryBuilder, QuerySpec
from cehrbert_data.const.common import CONDITION_OCCURRENCE, PERSON, VISIT_OCCURRENCE

COHORT_QUERY_TEMPLATE = """
SELECT DISTINCT
    c.person_id,
    c.index_date,
    c.visit_occurrence_id
FROM
(
    SELECT
        co.person_id,
        vo.visit_occurrence_id,
        coalesce(vo.visit_start_datetime, vo.visit_start_date) as index_date,
        ROW_NUMBER() OVER(
            PARTITION BY po.person_id
            ORDER BY vo.visit_start_date,
                vo.visit_start_datetime,
                vo.visit_occurrence_id
        ) as r_number
    FROM global_temp.condition_occurrence AS co
    JOIN global_temp.visit_occurrence AS vo
        ON co.visit_occurrence_id = vo.visit_occurrence_id
    JOIN global_temp.{ischemic_stroke_concepts} AS c
        ON co.condition_concept_id = c.concept_id
) AS c
WHERE c.r_number = 1
"""

ISCHEMIC_STROKE_CONCEPT_ID = [443454]

DEPENDENCY_LIST = [PERSON, VISIT_OCCURRENCE, CONDITION_OCCURRENCE]

DEFAULT_COHORT_NAME = "ischemic_stroke"
ISCHEMIC_STROKE_CONCEPTS = "ischemic_stroke_concepts"


def query_builder():
    query = QuerySpec(
        table_name=DEFAULT_COHORT_NAME,
        query_template=COHORT_QUERY_TEMPLATE,
        parameters={"ischemic_stroke_concepts": ISCHEMIC_STROKE_CONCEPTS},
    )

    ancestor_table_specs = [
        AncestorTableSpec(
            table_name=ISCHEMIC_STROKE_CONCEPTS,
            ancestor_concept_ids=ISCHEMIC_STROKE_CONCEPT_ID,
            is_standard=True,
        )
    ]
    return QueryBuilder(
        cohort_name=DEFAULT_COHORT_NAME,
        dependency_list=DEPENDENCY_LIST,
        query=query,
        ancestor_table_specs=ancestor_table_specs,
    )

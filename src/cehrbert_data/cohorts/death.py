from cehrbert_data.cohorts.query_builder import QueryBuilder, QuerySpec, create_cohort_entry_query_spec
from cehrbert_data.const.common import DEATH, PERSON, VISIT_OCCURRENCE, OBSERVATION

DEATH_COHORT_QUERY = """
WITH max_death_date_cte AS
(
    SELECT
        person_id,
        MAX(death_date) AS death_date
    FROM global_temp.death
    GROUP BY person_id
),
last_visit_start_date AS
(
    SELECT
        person_id,
        MAX(visit_start_date) AS last_visit_start_date
    FROM global_temp.visit_occurrence
    GROUP BY person_id
),
dnr AS (
    SELECT
        person_id,
        observation_datetime
    FROM global_temp.observation
    WHERE observation_concept_id in (36712886, 4119499, 4275347)
)

SELECT
    d.person_id,
    d.death_date AS index_date,
    CAST(null AS INT) AS visit_occurrence_id
FROM max_death_date_cte AS d
JOIN last_visit_start_date AS v
    ON d.person_id = v.person_id
        AND v.last_visit_start_date <= d.death_date
WHERE NOT EXISTS (
    SELECT 1
    FROM dnr
    WHERE d.person_id = dnr.person_id
        AND dnr.observation_datetime <= d.death_date 
)
"""

DEFAULT_COHORT_NAME = "mortality"
DEPENDENCY_LIST = [PERSON, DEATH, VISIT_OCCURRENCE, OBSERVATION]


def query_builder():
    query = QuerySpec(table_name=DEFAULT_COHORT_NAME, query_template=DEATH_COHORT_QUERY, parameters={})

    entry_cohort_query = create_cohort_entry_query_spec(entry_query_template=DEATH_COHORT_QUERY, parameters={})

    return QueryBuilder(
        cohort_name=DEFAULT_COHORT_NAME,
        dependency_list=DEPENDENCY_LIST,
        query=query,
        entry_cohort_query=entry_cohort_query,
    )

from pyspark.sql import functions
from pyspark.sql import Column

YEAR_TOKEN_PRIORITY = -10
AGE_TOKEN_PRIORITY = -9
GENDER_TOKEN_PRIORITY = -8
RACE_TOKEN_PRIORITY = -7
ATT_TOKEN_PRIORITY = -3
VS_TOKEN_PRIORITY = -2
VISIT_TYPE_TOKEN_PRIORITY = -1
DEFAULT_PRIORITY = 0
DISCHARGE_TOKEN_PRIORITY = 100
DEATH_TOKEN_PRIORITY = 199
VE_TOKEN_PRIORITY = 200


def get_inpatient_token_priority() -> Column:
    return functions.col("priority") + functions.col("concept_order") * 0.1


def get_inpatient_att_token_priority() -> Column:
    return functions.col("priority") - 0.01

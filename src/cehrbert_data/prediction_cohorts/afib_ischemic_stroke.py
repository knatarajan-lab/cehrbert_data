from cehrbert_data.cohorts import atrial_fibrillation, ischemic_stroke
from cehrbert_data.cohorts.spark_app_base import create_prediction_cohort
from cehrbert_data.const.common import CONDITION_OCCURRENCE, DRUG_EXPOSURE, PROCEDURE_OCCURRENCE
from cehrbert_data.utils.spark_parse_args import create_spark_args

DOMAIN_TABLE_LIST = [CONDITION_OCCURRENCE, DRUG_EXPOSURE, PROCEDURE_OCCURRENCE]

if __name__ == "__main__":
    create_prediction_cohort(
        create_spark_args(),
        atrial_fibrillation.query_builder(),
        ischemic_stroke.query_builder(),
        DOMAIN_TABLE_LIST,
    )

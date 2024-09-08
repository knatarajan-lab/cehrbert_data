from cehrbert_data.cohorts import death
from cehrbert_data.cohorts import last_visit_discharged_home as last
from cehrbert_data.cohorts.spark_app_base import create_prediction_cohort
from cehrbert_data.const.common import CONDITION_OCCURRENCE, DRUG_EXPOSURE, PROCEDURE_OCCURRENCE
from cehrbert_data.utils.spark_parse_args import create_spark_args

DOMAIN_TABLE_LIST = [CONDITION_OCCURRENCE, DRUG_EXPOSURE, PROCEDURE_OCCURRENCE]

if __name__ == "__main__":
    spark_args = create_spark_args()
    ehr_table_list = spark_args.ehr_table_list if spark_args.ehr_table_list else DOMAIN_TABLE_LIST

    create_prediction_cohort(
        spark_args,
        last.query_builder(spark_args),
        death.query_builder(),
        ehr_table_list,
    )

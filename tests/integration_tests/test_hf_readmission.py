import sys
import unittest

from cehrbert_data.prediction_cohorts.hf_readmission import main
from cehrbert_data.utils.spark_parse_args import create_spark_args

from ..pyspark_test_base import PySparkAbstract


class HfReadmissionTest(PySparkAbstract):

    def test_run_pyspark_app(self):
        sys.argv = [
            "hf_readmission.py",
            "--cohort_name",
            "hf_readmission",
            "--input_folder",
            self.get_sample_data_folder(),
            "--output_folder",
            self.get_output_folder(),
            "--date_lower_bound",
            "1985-01-01",
            "--date_upper_bound",
            "2023-12-31",
            "--age_lower_bound",
            "18",
            "--age_upper_bound",
            "100",
            "--observation_window",
            "360",
            "--prediction_start_days",
            "0",
            "--prediction_window",
            "30",
            "--include_visit_type",
            "--is_new_patient_representation",
            "--att_type",
            "cehr_bert",
            "--inpatient_att_type",
            "mix",
            "--ehr_table_list",
            "condition_occurrence",
            "procedure_occurrence",
            "drug_exposure",
            "--cache_events"
        ]

        main(create_spark_args())


if __name__ == "__main__":
    unittest.main()

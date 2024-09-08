import sys
import unittest
from ..pyspark_test import PySparkAbstract
from cehrbert_data.spark_parse_args import create_spark_args
from cehrbert_data.prediction_cohorts.hf_readmission import main


class HfReadmissionTest(PySparkAbstract):

    def run_pyspark_app_test(self):
        sys.argv = [
            "hf_readmission.py",
            "--cohort_name", "hf_readmission",
            "--input_folder", self.get_sample_data_folder(),
            "--output_folder", self.get_output_folder(),
            "--date_lower_bound", "1985-01-01",
            "--date_upper_bound", "2023-12-31",
            "--age_lower_bound", "18",
            "--age_upper_bound", "100",
            "--observation_window", "360",
            "--prediction_start_days", "0",
            "--prediction_window", "30",
            "--include_visit_type",
            "--is_new_patient_representation",
            "--att_type", "cehr_bert",
            "--ehr_table_list", "condition_occurrence procedure_occurrence drug_exposure"
        ]

        main(create_spark_args())


if __name__ == "__main__":
    unittest.main()

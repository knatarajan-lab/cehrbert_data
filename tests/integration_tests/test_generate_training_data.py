import unittest
from ..pyspark_test_base import PySparkAbstract
from cehrbert_data.decorators.patient_event_decorator import AttType
from cehrbert_data.apps.generate_training_data import main


class HfReadmissionTest(PySparkAbstract):

    def test_run_pyspark_app(self):
        main(
            input_folder=self.get_sample_data_folder(),
            output_folder=self.get_output_folder(),
            domain_table_list=["condition_occurrence", "drug_exposure", "procedure_occurrence"],
            date_filter="1985-01-01",
            include_visit_type=True,
            is_new_patient_representation=True,
            include_concept_list=False,
            gpt_patient_sequence=True,
            apply_age_filter=True,
            att_type=AttType.DAY
        )


if __name__ == "__main__":
    unittest.main()

import unittest

from cehrbert_data.apps.generate_training_data import main
from cehrbert_data.decorators.patient_event_decorator import AttType

from ..pyspark_test_base import PySparkAbstract


class HfReadmissionTest(PySparkAbstract):

    def test_run_pyspark_app(self):
        main(
            input_folder=self.get_sample_data_folder(),
            output_folder=self.get_output_folder(),
            domain_table_list=["condition_occurrence", "drug_exposure", "procedure_occurrence"],
            date_filter="1985-01-01",
            include_visit_type=True,
            is_new_patient_representation=True,
            exclude_visit_tokens=False,
            is_classic_bert=False,
            include_prolonged_stay=False,
            include_concept_list=False,
            gpt_patient_sequence=True,
            apply_age_filter=True,
            include_death=False,
            att_type=AttType.DAY,
        )


if __name__ == "__main__":
    unittest.main()

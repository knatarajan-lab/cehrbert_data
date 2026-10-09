import unittest

from pyspark.sql import functions as F

from cehrbert_data.apps.generate_training_data import main
from cehrbert_data.decorators import AttType

from ..pyspark_test_base import PySparkAbstract


class GenerateTrainingDataTest(PySparkAbstract):

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
            include_inpatient_hour_token=True,
            att_type=AttType.DAY,
            inpatient_att_type=AttType.DAY,
        )

        patient_events = self.spark.read.parquet(f"{self.get_output_folder()}/all_patient_events")
        non_omop_concept_ids = patient_events.where(
            ~F.col("standard_concept_id").rlike(r"^[0-9]+$")
        )
        self.assertEqual(non_omop_concept_ids.count(), 0)

    def test_exclude_condition_type_concept_ids(self):
        kwargs = dict(
            input_folder=self.get_sample_data_folder(),
            domain_table_list=["condition_occurrence", "procedure_occurrence"],
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
            inpatient_att_type=AttType.DAY,
        )
        # every condition of the sample data has the type 32020 (EHR encounter diagnosis)
        main(output_folder=self.get_output_folder(), **kwargs)
        events = self.spark.read.parquet(f"{self.get_output_folder()}/all_patient_events")
        self.assertGreater(events.where(F.col("domain") == "condition").count(), 0)

        main(output_folder=self.get_output_folder(), exclude_condition_type_concept_ids=[32020], **kwargs)
        events = self.spark.read.parquet(f"{self.get_output_folder()}/all_patient_events")
        self.assertEqual(events.where(F.col("domain") == "condition").count(), 0)
        self.assertGreater(events.where(F.col("domain") == "procedure").count(), 0)


if __name__ == "__main__":
    unittest.main()

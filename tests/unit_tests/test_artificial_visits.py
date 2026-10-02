import unittest
from datetime import date, datetime

from pyspark.sql import SparkSession

from cehrbert_data.apps.generate_training_data import join_visit_occurrence_with_person
from cehrbert_data.utils.spark_utils import construct_artificial_visits


class ArtificialVisitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("ArtificialVisitsUnitTest").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_events_on_artificial_visits_have_a_visit(self):
        # A single outpatient visit on 2020-01-01. One event happens in the visit, the other one is a problem list
        # record dated one and a half years later that is still attached to that visit.
        visit_occurrence = self.spark.createDataFrame(
            [(1, 100, 9202, date(2020, 1, 1), datetime(2020, 1, 1), date(2020, 1, 1), datetime(2020, 1, 1), 0)],
            "person_id long, visit_occurrence_id long, visit_concept_id long, visit_start_date date, "
            "visit_start_datetime timestamp, visit_end_date date, visit_end_datetime timestamp, "
            "discharged_to_concept_id long",
        )
        patient_events = self.spark.createDataFrame(
            [
                (1, 100, 9202, "A", datetime(2020, 1, 1), date(2020, 1, 1)),
                (1, 100, 9202, "B", datetime(2021, 6, 1), date(2021, 6, 1)),
            ],
            "person_id long, visit_occurrence_id long, visit_concept_id long, standard_concept_id string, "
            "datetime timestamp, date date",
        )
        person = self.spark.createDataFrame(
            [(1, datetime(1980, 1, 1), 8532, 8527)],
            "person_id long, birth_datetime timestamp, gender_concept_id long, race_concept_id long",
        )

        events, visits = construct_artificial_visits(
            patient_events, visit_occurrence, disconnect_problem_list_records=True
        )

        visit_id_of_event = {row["standard_concept_id"]: row["visit_occurrence_id"] for row in events.collect()}
        self.assertEqual(visit_id_of_event["A"], 100)
        # the problem list record is moved to an artificial visit
        self.assertNotEqual(visit_id_of_event["B"], 100)

        # The visit table that is joined to the events must contain the artificial visit, otherwise the event is
        # silently dropped when the events are joined to the visits
        visit_ids = {row["visit_occurrence_id"] for row in join_visit_occurrence_with_person(visits, person).collect()}
        self.assertEqual(visit_ids, {100, visit_id_of_event["B"]})
        stale_visit_ids = {
            row["visit_occurrence_id"] for row in join_visit_occurrence_with_person(visit_occurrence, person).collect()
        }
        self.assertNotIn(visit_id_of_event["B"], stale_visit_ids)


if __name__ == "__main__":
    unittest.main()

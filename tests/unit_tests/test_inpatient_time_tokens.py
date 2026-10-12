import unittest
from datetime import date, datetime

from pyspark.sql import SparkSession, functions as F

from cehrbert_data.apps.generate_training_data import join_visit_occurrence_with_person
from cehrbert_data.decorators import AttType
from cehrbert_data.utils.spark_utils import create_sequence_data_with_att

EVENT_SCHEMA = (
    "person_id long, standard_concept_id string, date date, datetime string, visit_occurrence_id long, "
    "domain string, event_group_id string, number_as_value float, concept_as_value string, unit string, "
    "parent_concept_id string, element_id int, visit_concept_id long, age int, cohort_member_id long"
)


def event(concept_id, event_date, event_datetime):
    return (1, concept_id, event_date, event_datetime, 10, "condition", "N/A", None, None, "N/A", concept_id, 0, 9201, 40, 1)


class InpatientTimeTokensTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.appName("InpatientTimeTokensUnitTest")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def tokens(self, events):
        # The timestamps of the events are strings, which are read in the UTC time zone of the Spark session like in
        # the pipeline (Python datetimes are read in the local time zone)
        patient_events = self.spark.createDataFrame(events, EVENT_SCHEMA).withColumn(
            "datetime", F.to_timestamp("datetime")
        )
        visit_occurrence = self.spark.createDataFrame(
            [(10, 1, date(2006, 4, 1), datetime(2006, 4, 1), date(2006, 4, 10), datetime(2006, 4, 10), 9201, 0)],
            "visit_occurrence_id long, person_id long, visit_start_date date, visit_start_datetime timestamp, "
            "visit_end_date date, visit_end_datetime timestamp, visit_concept_id long, discharged_to_concept_id long",
        )
        person = self.spark.createDataFrame(
            [(1, datetime(1966, 1, 1), 8532, 8527)],
            "person_id long, birth_datetime timestamp, gender_concept_id long, race_concept_id long",
        )
        sequence = create_sequence_data_with_att(
            patient_events,
            join_visit_occurrence_with_person(visit_occurrence, person),
            date_filter="1985-01-01",
            include_visit_type=True,
            exclude_visit_tokens=False,
            patient_demographic=person,
            att_type=AttType.DAY,
            inpatient_att_type=AttType.DAY,
            exclude_demographic=False,
            spark=self.spark,
        )
        return sequence.select("concept_ids").collect()[0][0]

    def test_inpatient_time_tokens_follow_the_datetime(self):
        # A is ordered before B by datetime but its date is a day later (midnight local time stored in UTC). The
        # datetime is authoritative, so A and B are on the same day and there is no negative time token.
        tokens = self.tokens(
            [
                event("A", date(2006, 4, 7), "2006-04-06 20:00:00"),
                event("B", date(2006, 4, 6), "2006-04-06 22:00:00"),
                event("C", date(2006, 4, 8), "2006-04-08 10:00:00"),
            ]
        )
        self.assertEqual(tokens[tokens.index("A"):], ["A", "B", "i-D2", "C", "i-D2", "Discharge/0", "[VE]"])

    def test_inpatient_time_tokens_in_order(self):
        tokens = self.tokens(
            [
                event("A", date(2006, 4, 2), "2006-04-02 08:00:00"),
                event("B", date(2006, 4, 2), "2006-04-02 09:00:00"),
                event("C", date(2006, 4, 5), "2006-04-05 10:00:00"),
            ]
        )
        self.assertEqual(tokens[tokens.index("A"):], ["A", "B", "i-D3", "C", "i-D5", "Discharge/0", "[VE]"])


if __name__ == "__main__":
    unittest.main()

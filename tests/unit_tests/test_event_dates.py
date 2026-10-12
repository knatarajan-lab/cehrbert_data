import unittest

from pyspark.sql import SparkSession, functions as F

from cehrbert_data.utils.spark_utils import extract_events_by_domain


class EventDatesTest(unittest.TestCase):
    """The datetime of an event is authoritative: the date is derived from it, and the date column of the OMOP table
    is only used when the datetime is missing."""

    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.appName("EventDatesUnitTest")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        cls.concept = cls.spark.createDataFrame(
            [(1, "SNOMED", "1")], "concept_id long, vocabulary_id string, concept_code string"
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def dates(self, events):
        # format the values in Spark, collect() would show the timestamps in the local time zone
        rows = events.select(
            "event_group_id",
            F.col("date").cast("string").alias("date"),
            F.date_format("datetime", "yyyy-MM-dd HH:mm:ss").alias("datetime"),
        ).collect()
        return {row["event_group_id"]: (row["date"], row["datetime"]) for row in rows}

    def test_condition_dates(self):
        # the datetime is read in the UTC time zone of the Spark session like in the pipeline
        condition_occurrence = self.spark.createDataFrame(
            [
                ("1", 1, 101, "2020-01-02", "2020-01-01 20:00:00", 101, 1),
                ("2", 1, 102, "2020-01-02", None, 102, 1),
                ("3", 1, 103, "2020-01-02", "2020-01-02 08:00:00", 103, 1),
            ],
            "condition_occurrence_id string, person_id long, condition_concept_id long, condition_start_date string, "
            "condition_start_datetime string, condition_source_concept_id long, visit_occurrence_id long",
        ).withColumn("condition_start_datetime", F.to_timestamp("condition_start_datetime"))
        events = extract_events_by_domain(condition_occurrence, concept=self.concept)
        self.assertEqual(
            self.dates(events),
            {
                "1": ("2020-01-01", "2020-01-01 20:00:00"),
                # no datetime: the date is used
                "2": ("2020-01-02", "2020-01-02 00:00:00"),
                "3": ("2020-01-02", "2020-01-02 08:00:00"),
            },
        )

    def test_measurement_dates(self):
        measurement = self.spark.createDataFrame(
            [
                (1, 1, 1, "2020-01-02", "2020-01-01 20:00:00", 1, 5.0, 0, 0, "mg"),
                (2, 1, 1, "2020-01-02", None, 1, 6.0, 0, 0, "mg"),
            ],
            "measurement_id long, person_id long, measurement_concept_id long, measurement_date string, "
            "measurement_datetime string, visit_occurrence_id long, value_as_number double, "
            "value_as_concept_id long, unit_concept_id long, unit_source_value string",
        ).withColumn("measurement_datetime", F.to_timestamp("measurement_datetime"))
        events = extract_events_by_domain(measurement, concept=self.concept, spark=self.spark)
        self.assertEqual(
            self.dates(events),
            {
                "1": ("2020-01-01", "2020-01-01 20:00:00"),
                "2": ("2020-01-02", "2020-01-02 00:00:00"),
            },
        )


if __name__ == "__main__":
    unittest.main()

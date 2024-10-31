import unittest
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType
from cehrbert_data.tools.ehrshot_to_omop import map_unit, map_answer, create_omop_person


# Define the test case
class EHRShotUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the Spark session for testing
        cls.spark = SparkSession.builder.appName("ehr_shot").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        # Stop the Spark session after tests are done
        cls.spark.stop()

    def setUp(self):
        concept_schema = StructType([
            StructField("concept_id", IntegerType(), True),
            StructField("concept_name", StringType(), True),
            StructField("concept_code", StringType(), True),
            StructField("domain_id", StringType(), True),
            StructField("vocabulary_id", StringType(), True),
        ])
        self.concept = self.spark.createDataFrame([
            (1, "percent", "%", "Unit", "UCUM"),
            (2, "Rare", "LA15679-6", "Meas Value", "LOINC"),
        ], concept_schema)

        # Define schemas for input DataFrames
        self.ehrshot_schema = StructType([
            StructField("omop_table", StringType(), True),
            StructField("patient_id", IntegerType(), True),
            StructField("code", StringType(), True),
            StructField("start", TimestampType(), True)
        ])

    def test_create_omop_person(self):
        # Sample concept data for mapping demographic codes to concept_ids
        concept_data = [
            ("Gender", "Male", 8507),  # concept_id for male gender
            ("Ethnicity", "Hispanic", 38003563),  # concept_id for Hispanic ethnicity
            ("Race", "White", 8527)  # concept_id for White race
        ]
        concept = self.spark.createDataFrame(concept_data, schema=StructType([
            StructField("vocabulary_id", StringType(), True),
            StructField("concept_code", StringType(), True),
            StructField("concept_id", IntegerType(), True)
        ]))
        # Sample EHR data simulating the "person" table with various demographic attributes
        ehr_data = [
            ("person", 1, "SNOMED/3950001", datetime(1980, 1, 1, 0, 0, 0)),  # birth_datetime
            ("person", 1, "Gender/Male", None),  # gender
            ("person", 1, "Ethnicity/Hispanic", None),  # ethnicity
            ("person", 1, "Race/White", None)  # race
        ]
        ehr_shot_data = self.spark.createDataFrame(ehr_data, schema=self.ehrshot_schema)
        expected_schema = StructType([
            StructField("person_id", IntegerType(), True),
            StructField("birth_datetime", TimestampType(), True),
            StructField("year_of_birth", IntegerType(), True),
            StructField("month_of_birth", IntegerType(), True),
            StructField("day_of_birth", IntegerType(), True),
            StructField("gender_concept_id", IntegerType(), True),
            StructField("gender_source_value", StringType(), True),
            StructField("ethnicity_concept_id", IntegerType(), True),
            StructField("ethnicity_source_value", StringType(), True),
            StructField("race_concept_id", IntegerType(), True),
            StructField("race_source_value", StringType(), True)
        ])
        expected_data = [
            (1, datetime(1980, 1, 1, 0, 0, 0),
             1980, 1, 1, 8507, "Gender/Male", 38003563, "Ethnicity/Hispanic", 8527, "Race/White")
        ]
        expected_df = self.spark.createDataFrame(expected_data, schema=expected_schema)
        actual_df = create_omop_person(ehr_shot_data, concept)
        self.assertEqual(expected_df.collect(), actual_df.collect())

    def test_map_answer(self):
        data = self.spark.createDataFrame([
            ("%", "1.2"), ("%", "2.2"), (None, "Rare"), (None, "unknown"), ("unknown", None)
        ], ["unit", "value"])

        mapped_answer = map_answer(data, self.concept)
        expected_data = [
            ("1.2", "%", 0),  # Removed both curly bracket content and leading /
            ("2.2", "%", 0),  # Removed curly bracket content
            ("Rare", None, 2),  # Removed leading /
            ("unknown", None, 0),  # No change
            (None, "unknown", None),  # No change
        ]
        expected_df = self.spark.createDataFrame(
            expected_data,
            ["value", "unit", "value_as_concept_id"]
        )

        # Collect the actual and expected results
        actual_result = mapped_answer.sort(mapped_answer.schema.fieldNames()).toPandas().fillna(value=-1).to_dict(
            orient="records")
        expected_result = expected_df.sort(expected_df.schema.fieldNames()).toPandas().fillna(value=-1).to_dict(
            orient="records")

        # Compare the actual and expected results
        self.assertListEqual(actual_result, expected_result)

    def test_map_unit(self):
        data = self.spark.createDataFrame([
            ("%",), ("%",), (None,), (None,), ("unknown",)
        ], ["unit"])

        # Call the function to clean up the units
        mapped_unit = map_unit(data, self.concept).sort("unit", "unit_concept_id")

        expected_data = [
            ("%", 1),  # Removed both curly bracket content and leading /
            ("%", 1),  # Removed curly bracket content
            (None, None),  # Removed leading /
            (None, None),  # No change
            ("unknown", 0),  # No change
        ]
        expected_df = self.spark.createDataFrame(
            expected_data, ["unit", "unit_concept_id"]
        ).sort("unit", "unit_concept_id")

        # Collect the actual and expected results
        actual_result = mapped_unit.toPandas().fillna(value=-1).to_dict(orient="records")
        expected_result = expected_df.toPandas().fillna(value=-1).to_dict(orient="records")

        # Compare the actual and expected results
        self.assertListEqual(actual_result, expected_result)


# Entry point for running the tests
if __name__ == "__main__":
    unittest.main()

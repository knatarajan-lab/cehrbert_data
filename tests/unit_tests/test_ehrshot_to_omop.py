import unittest
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from cehrbert_data.tools.ehrshot_to_omop import map_unit, map_answer


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
        actual_result = mapped_answer.sort(mapped_answer.schema.fieldNames()).toPandas().fillna(value=-1).to_dict(orient="records")
        expected_result = expected_df.sort(expected_df.schema.fieldNames()).toPandas().fillna(value=-1).to_dict(orient="records")

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

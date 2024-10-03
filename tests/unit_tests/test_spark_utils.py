import unittest
from pyspark.sql import SparkSession
from cehrbert_data.utils.spark_utils import clean_up_unit


# Define the test case
class CleanUpUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the Spark session for testing
        cls.spark = SparkSession.builder.appName("UnitTest").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        # Stop the Spark session after tests are done
        cls.spark.stop()

    def test_clean_up_unit(self):
        # Create a sample DataFrame
        test_data = [
            ("mg/dL{adult}",),  # Contains both a leading / and a curly bracket
            ("kg/m2{child}",),  # Contains only a curly bracket
            ("mmHg",),  # Contains only a leading /
            ("g/L",),  # Contains neither
            ("/min",),
        ]
        df = self.spark.createDataFrame(test_data, ["unit"])

        # Call the function to clean up the units
        cleaned_df = clean_up_unit(df)

        # Expected results after cleaning
        expected_data = [
            ("mg/dL",),  # Removed both curly bracket content and leading /
            ("kg/m2",),  # Removed curly bracket content
            ("mmHg",),  # Removed leading /
            ("g/L",),  # No change
            ("1/min",),
        ]
        expected_df = self.spark.createDataFrame(expected_data, ["unit"])

        # Collect the actual and expected results
        actual_result = cleaned_df.collect()
        expected_result = expected_df.collect()

        # Compare the actual and expected results
        self.assertEqual(actual_result, expected_result)


# Entry point for running the tests
if __name__ == "__main__":
    unittest.main()

import unittest
import tempfile
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, TimestampType, FloatType
from cehrbert_data.tools.ehrshot_to_omop import (
    map_unit,
    map_answer,
    create_omop_person,
    convert_code_to_omop_concept,
    extract_value,
    generate_visit_id,
    drop_duplicate_visits
)


# Define the test case
class EHRShotUnitTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize the Spark session for testing
        cls.spark = SparkSession.builder.appName("ehr_shot").getOrCreate()
        cls.spark.conf.set("spark.sql.analyzer.failAmbiguousSelfJoin", False)

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

    def test_generate_visit_id(self):
        # Define schema for input DataFrame
        schema = StructType([
            StructField("patient_id", IntegerType(), True),
            StructField("start", TimestampType(), True),
            StructField("end", TimestampType(), True),
            StructField("visit_id", IntegerType(), True),
            StructField("omop_table", StringType(), True),
            StructField("code", StringType(), True),
            StructField("unit", StringType(), True),
            StructField("value", StringType(), True),
        ])

        # Sample data with multiple events for each patient and different time gaps
        data = [
            (1, datetime(2023, 1, 1, 8), datetime(2023, 1, 1, 9), 1, "visit_occurrence", None, None, None),
            (1, datetime(2023, 1, 2, 20), datetime(2023, 1, 2, 20), None, "condition_occurrence", None, None, None), # merge with the visit record below
            (1, datetime(2023, 1, 2, 20), datetime(2023, 1, 2, 20), 2, "visit_occurrence", "Visit/IP", None, None),  # another visit
            (2, datetime(2023, 1, 1, 8), datetime(2023, 1, 1, 9), 3, "visit_occurrence", None, None, None),
            (2, datetime(2023, 1, 1, 10), datetime(2023, 1, 1, 11), None, "condition_occurrence", None, None, None),
            (3, datetime(2023, 1, 1, 8), datetime(2023, 1, 1, 9), 1000, "visit_occurrence", None, None, None),
            (4, datetime(2023, 1, 1, 8), datetime(2023, 1, 1, 9), None, "condition_occurrence", None, None, None)
        ]

        # Create DataFrame
        data = self.spark.createDataFrame(data, schema=schema)
        # Run the function to generate visit IDs
        temp_dir = tempfile.mkdtemp()
        result_df = generate_visit_id(data, self.spark, temp_dir)
        result_df.orderBy("patient_id", "start").show()
        # Validate the number of visits
        self.assertEqual(6, result_df.select("visit_id").where(f.col("visit_id").isNotNull()).distinct().count())
        # Two artificial visits are created therefore it's 7 + 2 = 9
        self.assertEqual(9, result_df.count())

        # Check that visit_id was generated as an integer (bigint)
        self.assertIn(
            result_df.schema["visit_id"].dataType.simpleString(), ["int", "bigint"],
            "visit_id should be of type bigint"
        )

        # Validate visit_id assignment based on time interval gaps
        patient_1_visits = result_df.filter(f.col("patient_id") == 1).select("visit_id").distinct().count()
        self.assertEqual(
            2,
            patient_1_visits,
            "Patient 1 should have three distinct visits based on time interval."
        )

        patient_2_visits = result_df.filter(f.col("patient_id") == 2).select("visit_id").distinct().count()
        self.assertEqual(
            2, patient_2_visits, "Patient 2 should have one visit as events are within time interval."
        )

        patient_3_visits = result_df.filter(f.col("patient_id") == 3).select("visit_id").distinct().count()
        self.assertEqual(
            1, patient_3_visits, "Patient 3 should have 1 visit as events are within time interval."
        )

        patient_4_visits = result_df.filter(f.col("patient_id") == 4).select("visit_id").collect()[0].visit_id
        self.assertTrue(
            patient_4_visits > 1000, "Patient 4 should have one generated visit_id."
        )

    def test_drop_duplicate_visits(self):
        # Define schema for input DataFrame
        schema = StructType([
            StructField("visit_id", IntegerType(), True),
            StructField("code", StringType(), True)
        ])

        # Sample data with duplicate visit IDs and varying priorities
        data = [
            (1, "Visit/IP"),  # Highest priority for visit_id 1
            (1, "Visit/ER"),  # Lower priority for visit_id 1
            (2, "Visit/OP"),  # Lowest priority for visit_id 2
            (2, "Visit/ER"),  # Medium priority for visit_id 2
            (3, "Visit/ERIP"),  # Highest priority for visit_id 3
            (3, "Visit/OP"),  # Lower priority for visit_id 3
            (4, "Visit/OP")  # Highest priority for visit_id 4
        ]

        # Create DataFrame
        data = self.spark.createDataFrame(data, schema=schema)

        # Run the function to drop duplicates
        result_df = drop_duplicate_visits(data)

        # Define expected data and schema
        expected_data = [
            (1, "Visit/IP"),  # Only highest priority Visit/IP retained for visit_id 1
            (2, "Visit/ER"),  # Only medium priority Visit/ER retained for visit_id 2
            (3, "Visit/ERIP"),  # Only highest priority Visit/ERIP retained for visit_id 3
            (4, "Visit/OP")  # Only highest priority Visit/OP retained for visit_id 3
        ]

        expected_df = self.spark.createDataFrame(expected_data, schema=data.schema)

        # Collect results for comparison
        actual_data = result_df.sort("visit_id").collect()
        expected_data = expected_df.sort("visit_id").collect()

        # Check that the actual data matches the expected data
        self.assertEqual(actual_data, expected_data, "The DataFrames do not match the expected result.")

    def test_extract_value(self):
        current_time = datetime.now()
        # Create DataFrames
        data = self.spark.createDataFrame([
            (1, current_time, "123.45", "mg"),  # Numeric with unit
            (1, current_time, "positive", None),  # Categorical answer
            (1, current_time, "not_available", None),  # Other, unhandled value
            (1, current_time, None, None),  # None
        ], schema=StructType([
            StructField("patient_id", IntegerType(), True),
            StructField("start", TimestampType(), True),
            StructField("value", StringType(), True),
            StructField("unit", StringType(), True)
        ]))
        concept = self.spark.createDataFrame([
            ("mg", 9001, "mg", "Unit"),  # Unit concept
            ("positive", 2001, "positive", "Meas Value")  # Meas Value concept
        ], schema=StructType([
            StructField("concept_code", StringType(), True),
            StructField("concept_id", IntegerType(), True),
            StructField("concept_name", StringType(), True),
            StructField("domain_id", StringType(), True)
        ]))

        # Run function
        actual_df = extract_value(data, concept)

        # Define expected data and schema
        expected_data = [
            (1, current_time, "123.45", "mg", 123.45, None, 9001),  # Numeric with mapped unit
            (1, current_time, "positive", None, None, 2001, None),  # Categorical with mapped answer
            (1, current_time, "not_available", None, None, 0, None),  # Unmatched
            (1, current_time, None, None, None, None, None)  # None
        ]

        expected_schema = StructType([
            StructField("patient_id", IntegerType(), True),
            StructField("start", TimestampType(), True),
            StructField("value_source_value", StringType(), True),
            StructField("unit_source_value", StringType(), True),
            StructField("value_as_number", FloatType(), True),
            StructField("value_as_concept_id", IntegerType(), True),
            StructField("unit_concept_id", IntegerType(), True),
        ])

        expected_df = self.spark.createDataFrame(expected_data, schema=expected_schema)

        # Collect results for comparison
        actual_data = actual_df.sort("value_source_value").collect()
        expected_data = expected_df.sort("value_source_value").collect()
        # Compare results
        for i, (actual, expected) in enumerate(zip(actual_data, expected_data)):
            self.assertDictEqual(actual.asDict(), expected.asDict(), f"Error index {i}")

    def test_convert_code_to_omop_concept(self):
        # Create DataFrames
        data = self.spark.createDataFrame([
            (1, "ICD10/1234"),
            (2, "SNOMED/5678"),
            (3, "ICD10/0000")  # No matching concept
        ], schema=StructType([
            StructField("patient_id", IntegerType(), True),
            StructField("code", StringType(), True)
        ]))
        concept = self.spark.createDataFrame(
            [
                ("ICD10", "1234", 1001),
                ("SNOMED", "5678", 1002)
            ],
            schema=StructType([
                StructField("vocabulary_id", StringType(), True),
                StructField("concept_code", StringType(), True),
                StructField("concept_id", IntegerType(), True)
            ])
        )
        # Run function
        actual_df = convert_code_to_omop_concept(data, concept, "code")

        # Define expected data and schema
        expected_data = [
            (1, "ICD10/1234", 1001),  # Match with concept_id 1001
            (2, "SNOMED/5678", 1002),  # Match with concept_id 1002
            (3, "ICD10/0000", 0)  # No match, default concept_id 0
        ]

        expected_schema = StructType([
            StructField("patient_id", IntegerType(), True),
            StructField("code", StringType(), True),
            StructField("concept_id", IntegerType(), True)
        ])

        expected_df = self.spark.createDataFrame(expected_data, schema=expected_schema)

        # Collect results for comparison
        actual_data = actual_df.sort("patient_id").collect()
        expected_data = expected_df.sort("patient_id").collect()

        # Compare results
        self.assertEqual(actual_data, expected_data)

    def test_create_omop_person(self):
        # Sample concept data for mapping demographic codes to concept_ids
        concept = self.spark.createDataFrame([
            ("Gender", "Male", 8507),  # concept_id for male gender
            ("Ethnicity", "Hispanic", 38003563),  # concept_id for Hispanic ethnicity
            ("Race", "White", 8527)  # concept_id for White race
        ], schema=StructType([
            StructField("vocabulary_id", StringType(), True),
            StructField("concept_code", StringType(), True),
            StructField("concept_id", IntegerType(), True)
        ]))
        ehr_shot_data = self.spark.createDataFrame([
            ("person", 1, "SNOMED/3950001", datetime(1980, 1, 1, 0, 0, 0)),  # birth_datetime
            ("person", 1, "Gender/Male", None),  # gender
            ("person", 1, "Ethnicity/Hispanic", None),  # ethnicity
            ("person", 1, "Race/White", None)  # race
        ], schema=self.ehrshot_schema)

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

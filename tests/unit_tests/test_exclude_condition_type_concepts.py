import unittest

from pyspark.sql import SparkSession

from cehrbert_data.utils.spark_utils import exclude_condition_type_concepts


class ExcludeConditionTypeConceptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("ExcludeConditionTypeConceptsUnitTest").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def condition_occurrence(self):
        return self.spark.createDataFrame(
            [(1, 32840), (2, 32821), (3, 32827), (4, None), (5, 32840)],
            "condition_occurrence_id long, condition_type_concept_id long",
        )

    def ids(self, condition_occurrence):
        return sorted(row["condition_occurrence_id"] for row in condition_occurrence.collect())

    def test_excludes_the_given_types_and_keeps_the_others(self):
        result = exclude_condition_type_concepts(self.condition_occurrence(), [32840, 32821])
        self.assertEqual(self.ids(result), [3, 4])

    def test_records_without_a_type_are_kept(self):
        result = exclude_condition_type_concepts(self.condition_occurrence(), [32827])
        self.assertIn(4, self.ids(result))

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            exclude_condition_type_concepts(self.spark.range(1).toDF("condition_occurrence_id"), [32840])


if __name__ == "__main__":
    unittest.main()

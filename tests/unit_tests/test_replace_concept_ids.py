import unittest

from pyspark.sql import SparkSession

from cehrbert_data.utils.spark_utils import replace_concept_ids_with_concept_codes


class ReplaceConceptIdsWithConceptCodesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("ReplaceConceptIdsUnitTest").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_replace_concept_ids(self):
        concept = self.spark.createDataFrame(
            [
                (8532, "Gender", "F"),
                (2313828, "CPT4", "72100"),
                (9202, "Visit", "OP"),
                (45, "CMS Place of Service", "12 A"),
                (777, "LOINC", None),
            ],
            "concept_id long, vocabulary_id string, concept_code string",
        )
        tokens = ["8532", "2313828", "9202", "45", "777", "999999", "0", "year:2012", "=6mt", "VALUE_BIN/3", "ATC/0/C07"]
        events = self.spark.createDataFrame([(i, t) for i, t in enumerate(tokens)], "row_id long, standard_concept_id string")
        result = {
            r["row_id"]: r["standard_concept_id"]
            for r in replace_concept_ids_with_concept_codes(events, concept).collect()
        }
        self.assertEqual(
            [result[i] for i in range(len(tokens))],
            [
                "Gender/F",
                "CPT4/72100",
                "Visit/OP",
                "CMS_Place_of_Service/12_A",
                "777",  # no concept code
                "999999",  # not in the concept table
                "0",
                "year:2012",
                "=6mt",
                "VALUE_BIN/3",
                "ATC/0/C07",
            ],
        )

    def test_integer_typed_tokens(self):
        concept = self.spark.createDataFrame([(8532, "Gender", "F")], "concept_id long, vocabulary_id string, concept_code string")
        events = self.spark.createDataFrame([(8532,), (1,)], "standard_concept_id long")
        result = sorted(r["standard_concept_id"] for r in replace_concept_ids_with_concept_codes(events, concept).collect())
        self.assertEqual(result, ["1", "Gender/F"])


if __name__ == "__main__":
    unittest.main()

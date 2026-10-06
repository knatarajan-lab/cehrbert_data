import csv
import os
import unittest

from pyspark.sql import SparkSession

from cehrbert_data.utils import icd_cm_tokens as icd_cm_tokens_module
from cehrbert_data.utils import icd_pcs_tokens as icd_pcs_tokens_module
from cehrbert_data.utils.icd_pcs_tokens import icd_pcs_tokens
from cehrbert_data.utils.spark_utils import extract_events_by_domain

RESOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


class IcdPcsTokensTest(unittest.TestCase):
    def test_same_tokens_as_ethos_ares(self):
        """ethos_ares_icd_pcs_tokens.csv has the seven characters of the tokens that ProcedureData.split_icd_codes of
        ethos-ares gives to the ICD procedure codes of an OMOP vocabulary and to some edge cases (see
        generate_ethos_ares_icd_tokens.py), every token is ICD//PCS// and one character."""
        with open(os.path.join(RESOURCES_DIR, "ethos_ares_icd_pcs_tokens.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertGreater(len(rows), 5000)
        different = []
        for row in rows:
            expected = tuple(f"ICD//PCS//{character}" for character in row["characters"])
            actual = icd_pcs_tokens(row["vocabulary_id"], row["concept_code"])
            if actual != expected:
                different.append((row["vocabulary_id"], row["concept_code"], expected, actual))
        self.assertEqual(different, [], f"{len(different)} of {len(rows)} codes have other tokens than in ethos-ares")

    def test_icd10pcs(self):
        self.assertEqual(
            icd_pcs_tokens("ICD10PCS", "0DTJ4ZZ"),
            ("ICD//PCS//0", "ICD//PCS//D", "ICD//PCS//T", "ICD//PCS//J", "ICD//PCS//4", "ICD//PCS//Z", "ICD//PCS//Z"),
        )
        self.assertEqual(icd_pcs_tokens("ICD10PCS", "0dtj4zz"), icd_pcs_tokens("ICD10PCS", "0DTJ4ZZ"))

    def test_icd9_procedure_codes_are_converted_to_icd10pcs(self):
        tokens = icd_pcs_tokens("ICD9Proc", "00.10")
        self.assertEqual(len(tokens), 7)
        # the dot and the trailing zeros do not matter
        self.assertEqual(icd_pcs_tokens("ICD9Proc", "0010"), tokens)

    def test_dropped_codes(self):
        for vocabulary_id, code in [
            ("ICD10PCS", "NoPCS"), ("ICD10PCS", "0DTJ4Z"), ("ICD10PCS", "0DTJ4ZZZ"), ("ICD9Proc", "XYZ"),
            ("ICD10PCS", None), ("ICD10PCS", ""),
        ]:
            with self.subTest(code=code):
                self.assertEqual(icd_pcs_tokens(vocabulary_id, code), ())


class IcdPcsTokensSparkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local").appName("icd_pcs_tokens").getOrCreate()
        cls.concept = cls.spark.createDataFrame(
            [
                (1, "ICD10PCS", "0DTJ4ZZ"),
                (2, "ICD10PCS", "NoPCS"),
                (3, "CPT4", "33519"),
                (4, "ICD9Proc", "00.10"),
                (5, "ICD9Proc", "XYZ"),
            ],
            ["concept_id", "vocabulary_id", "concept_code"],
        )
        cls.procedures = cls.spark.createDataFrame(
            [(str(i), 1, 100 + i, "2020-01-01", "2020-01-01 10:00:00", i, 1) for i in range(1, 6)],
            ["procedure_occurrence_id", "person_id", "procedure_concept_id", "procedure_date",
             "procedure_datetime", "procedure_source_concept_id", "visit_occurrence_id"],
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def tokens(self, **kwargs):
        events = extract_events_by_domain(self.procedures, concept=self.concept, **kwargs)
        result = {}
        for row in events.orderBy("event_group_id", "element_id").collect():
            result.setdefault(row["event_group_id"], []).append(row["standard_concept_id"])
        return result

    def test_procedures_with_ethos_icd_tokens(self):
        tokens = self.tokens(ethos_icd_tokens=True)
        self.assertEqual(tokens["1"], [f"ICD//PCS//{c}" for c in "0DTJ4ZZ"])
        # the ICD-9 procedure code is converted
        self.assertEqual(tokens["4"], list(icd_pcs_tokens("ICD9Proc", "00.10")))
        self.assertEqual(len(tokens["4"]), 7)
        # a procedure that is not an ICD code keeps its concept id
        self.assertEqual(tokens["3"], ["103"])
        # the placeholder and the ICD-9 code without a mapping are dropped
        self.assertNotIn("2", tokens)
        self.assertNotIn("5", tokens)

    def test_procedures_without_ethos_icd_tokens_are_unchanged(self):
        self.assertEqual(
            self.tokens(),
            {str(i): [str(100 + i)] for i in range(1, 6)},
        )


class SparkUdfsTest(unittest.TestCase):
    def test_udf_functions_are_sent_to_the_workers_by_name(self):
        """The cloudpickle of PySpark 3.1 can't pickle a lambda or a function defined inside another function with the
        bytecode of Python 3.11 and later (IndexError: tuple index out of range), so the functions of the UDFs have to
        be defined in their module, which makes Spark send them by their name."""
        for module in (icd_cm_tokens_module, icd_pcs_tokens_module):
            with self.subTest(module=module.__name__):
                function = module.get_spark_udf().func
                self.assertNotIn("<lambda>", function.__qualname__)
                self.assertNotIn("<locals>", function.__qualname__)
                self.assertEqual(function.__module__, module.__name__)


if __name__ == "__main__":
    unittest.main()

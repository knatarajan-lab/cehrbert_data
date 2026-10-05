import csv
import json
import os
import unittest

from pyspark.sql import SparkSession

from cehrbert_data.utils.icd_cm_tokens import icd_cm_tokens
from cehrbert_data.utils.spark_utils import extract_events_by_domain

RESOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


class IcdCmTokensTest(unittest.TestCase):
    def test_same_tokens_as_ethos_ares(self):
        """ethos_ares_icd_cm_tokens.csv has the tokens that DiagnosesData.split_icd_codes of ethos-ares gives to the
        ICD codes of an OMOP vocabulary and to some edge cases (see generate_ethos_ares_icd_cm_tokens.py)."""
        with open(os.path.join(RESOURCES_DIR, "ethos_ares_icd_cm_tokens.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertGreater(len(rows), 2500)
        different = [
            (row["vocabulary_id"], row["concept_code"], json.loads(row["tokens"]), list(icd_cm_tokens(row["vocabulary_id"], row["concept_code"])))
            for row in rows
            if list(icd_cm_tokens(row["vocabulary_id"], row["concept_code"])) != json.loads(row["tokens"])
        ]
        self.assertEqual(different, [], f"{len(different)} of {len(rows)} codes have other tokens than in ethos-ares")

    def test_icd10cm(self):
        self.assertEqual(
            icd_cm_tokens("ICD10CM", "I25.10"), ("ICD//CM//CHRONIC_ISCHEMIC_HEART_DISEASE", "ICD//CM//3-6//10")
        )
        self.assertEqual(
            icd_cm_tokens("ICD10CM", "S72.001A"),
            ("ICD//CM//FRACTURE_OF_FEMUR", "ICD//CM//3-6//001", "ICD//CM//SFX//A"),
        )
        self.assertEqual(icd_cm_tokens("ICD10CM", "I21.9"), ("ICD//CM//ACUTE_MYOCARDIAL_INFARCTION", "ICD//CM//3-6//9"))

    def test_trailing_zeros_are_removed_until_the_code_is_known(self):
        # padded codes
        self.assertEqual(icd_cm_tokens("ICD10CM", "E11.90"), icd_cm_tokens("ICD10CM", "E11.9"))
        self.assertEqual(len(icd_cm_tokens("ICD10CM", "I10.00")), 1)
        # a known code that ends with a zero is kept
        self.assertEqual(icd_cm_tokens("ICD10CM", "E11.40")[-1], "ICD//CM//3-6//40")
        # the list of the codes is of January 2021, so the code that was added later is shortened as in ethos-ares
        self.assertEqual(icd_cm_tokens("ICD10CM", "M54.50")[-1], "ICD//CM//3-6//5")

    def test_icd9_is_converted_to_icd10(self):
        self.assertEqual(icd_cm_tokens("ICD9CM", "428.22"), ("ICD//CM//HEART_FAILURE", "ICD//CM//3-6//22"))
        # a category without an exact match gets the category of its first subcode
        self.assertEqual(icd_cm_tokens("ICD9CM", "585"), ("ICD//CM//CHRONIC_KIDNEY_DISEASE_(CKD)",))

    def test_dropped_codes(self):
        for vocabulary_id, code in [("ICD10CM", "QQQ.1"), ("ICD10CM", "A"), ("ICD9CM", "XYZ"), ("ICD10CM", None), ("ICD10CM", "")]:
            with self.subTest(code=code):
                self.assertEqual(icd_cm_tokens(vocabulary_id, code), ())


class IcdCmTokensSparkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local").appName("icd_cm_tokens").getOrCreate()
        cls.concept = cls.spark.createDataFrame(
            [
                (1, "ICD10CM", "I21.9"),
                (2, "ICD10CM", "S72.001A"),
                (3, "ICD9CM", "428.22"),
                (4, "SNOMED", "44054006"),
                (5, "ICD10CM", "QQQ.1"),
                (6, "ICD10PCS", "0DTJ4ZZ"),
                (7, "ATC", "N02BE01"),
            ],
            ["concept_id", "vocabulary_id", "concept_code"],
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def tokens(self, domain_table, **kwargs):
        events = extract_events_by_domain(domain_table, concept=self.concept, **kwargs)
        result = {}
        for row in events.orderBy("event_group_id", "element_id").collect():
            result.setdefault(row["event_group_id"], []).append(row["standard_concept_id"])
        return result

    def condition_table(self):
        return self.spark.createDataFrame(
            [(str(i), 1, 100 + i, "2020-01-01", "2020-01-01 10:00:00", i, 1) for i in range(1, 6)],
            ["condition_occurrence_id", "person_id", "condition_concept_id", "condition_start_date",
             "condition_start_datetime", "condition_source_concept_id", "visit_occurrence_id"],
        )

    def test_conditions_with_ethos_icd_cm_tokens(self):
        self.assertEqual(
            self.tokens(self.condition_table(), ethos_icd_cm_tokens=True),
            {
                "1": ["ICD//CM//ACUTE_MYOCARDIAL_INFARCTION", "ICD//CM//3-6//9"],
                "2": ["ICD//CM//FRACTURE_OF_FEMUR", "ICD//CM//3-6//001", "ICD//CM//SFX//A"],
                "3": ["ICD//CM//HEART_FAILURE", "ICD//CM//3-6//22"],
                # a code that is not an ICD code keeps its concept id
                "4": ["104"],
                # the code with an unknown category is dropped, "5" has no tokens
            },
        )

    def test_conditions_without_ethos_icd_cm_tokens_are_unchanged(self):
        self.assertEqual(
            self.tokens(self.condition_table()),
            {
                "1": ["ICD10CM/0/I21", "ICD10CM/1/9"],
                "2": ["ICD10CM/0/S72", "ICD10CM/1/001A"],
                "3": ["ICD9CM/0/428", "ICD9CM/1/22"],
                "4": ["104"],
                "5": ["ICD10CM/0/QQQ", "ICD10CM/1/1"],
            },
        )

    def test_procedures_and_drugs_are_not_changed(self):
        procedures = self.spark.createDataFrame(
            [("6", 1, 106, "2020-01-01", "2020-01-01 10:00:00", 6, 1)],
            ["procedure_occurrence_id", "person_id", "procedure_concept_id", "procedure_date",
             "procedure_datetime", "procedure_source_concept_id", "visit_occurrence_id"],
        )
        drugs = self.spark.createDataFrame(
            [("7", 1, 107, "2020-01-01", "2020-01-01 10:00:00", 7, 1)],
            ["drug_exposure_id", "person_id", "drug_concept_id", "drug_exposure_start_date",
             "drug_exposure_start_datetime", "drug_source_concept_id", "visit_occurrence_id"],
        )
        for table in (procedures, drugs):
            tokens = self.tokens(table)
            self.assertTrue(tokens)
            self.assertEqual(self.tokens(table, ethos_icd_cm_tokens=True), tokens)


if __name__ == "__main__":
    unittest.main()

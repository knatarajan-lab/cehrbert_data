import unittest

from pyspark.sql import SparkSession

from cehrbert_data.utils.ethos_code_tokens import atc_tokens, icd10cm_tokens, icd10pcs_tokens
from cehrbert_data.utils.spark_utils import extract_events_by_domain


class EthosCodeTokensTest(unittest.TestCase):
    """The tokens are those of the ethos-ares tokenizer for the same codes, except that the category
    is written as its code and not as its description (ICD//CM//I21, not ICD//CM//ACUTE_MYOCARDIAL_INFARCTION)."""

    def test_icd10cm(self):
        self.assertEqual(
            icd10cm_tokens("I21.9"), ["ICD//CM//I21", "ICD//CM//3-6//9"]
        )
        self.assertEqual(
            icd10cm_tokens("S72.001A"),
            ["ICD//CM//S72", "ICD//CM//3-6//001", "ICD//CM//SFX//A"],
        )
        self.assertEqual(icd10cm_tokens("I50"), ["ICD//CM//I50"])
        self.assertEqual(
            icd10cm_tokens("I50.84"), ["ICD//CM//I50", "ICD//CM//3-6//84"]
        )

    def test_icd10cm_malformed_codes(self):
        # rules only: a padded code keeps its zeros, which ethos-ares removes with its list of codes
        self.assertEqual(icd10cm_tokens("I10.00"), ["ICD//CM//I10", "ICD//CM//3-6//00"])
        self.assertEqual(icd10cm_tokens("I10"), ["ICD//CM//I10"])
        self.assertEqual(icd10cm_tokens("QQQ.1"), [])
        self.assertEqual(icd10cm_tokens("NoDx"), [])
        self.assertEqual(icd10cm_tokens(None), [])

    def test_icd10pcs(self):
        self.assertEqual(
            icd10pcs_tokens("0DTJ4ZZ"),
            ["ICD//PCS//0", "ICD//PCS//D", "ICD//PCS//T", "ICD//PCS//J", "ICD//PCS//4",
             "ICD//PCS//Z", "ICD//PCS//Z"],
        )
        self.assertEqual(icd10pcs_tokens("NoPCS"), [])
        self.assertEqual(icd10pcs_tokens("0DTJ4Z"), [])

    def test_atc(self):
        self.assertEqual(
            atc_tokens("N02BE01"), ["ATC//N02", "ATC//4//B", "ATC//SFX//E01"]
        )
        self.assertEqual(atc_tokens("N02B"), ["ATC//N02", "ATC//4//B"])
        self.assertEqual(atc_tokens("N02"), ["ATC//N02"])


class EthosCodeTokensSparkTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.master("local").appName("ethos_code_tokens").getOrCreate()
        cls.concept = cls.spark.createDataFrame(
            [
                (1, "ICD10CM", "I21.9"),
                (2, "ICD10CM", "S72.001A"),
                (3, "ICD9CM", "428.22"),
                (4, "SNOMED", "44054006"),
                (5, "ICD10CM", "QQQ.1"),
                (6, "ICD10PCS", "0DTJ4ZZ"),
                (7, "ICD10PCS", "NoPCS"),
                (8, "CPT4", "33519"),
                (9, "ATC", "N02BE01"),
                (10, "RxNorm", "161"),
            ],
            ["concept_id", "vocabulary_id", "concept_code"],
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def tokens(self, domain_table, **kwargs):
        events = extract_events_by_domain(domain_table, concept=self.concept, **kwargs)
        rows = events.orderBy("event_group_id", "element_id").collect()
        result = {}
        for row in rows:
            result.setdefault(row["event_group_id"], []).append(row["standard_concept_id"])
        return result

    def condition_table(self):
        return self.spark.createDataFrame(
            [(str(i), 1, 100 + i, "2020-01-01", "2020-01-01 10:00:00", i, 1) for i in range(1, 6)],
            ["condition_occurrence_id", "person_id", "condition_concept_id", "condition_start_date",
             "condition_start_datetime", "condition_source_concept_id", "visit_occurrence_id"],
        )

    def procedure_table(self):
        return self.spark.createDataFrame(
            [(str(i), 1, 100 + i, "2020-01-01", "2020-01-01 10:00:00", i, 1) for i in (6, 7, 8)],
            ["procedure_occurrence_id", "person_id", "procedure_concept_id", "procedure_date",
             "procedure_datetime", "procedure_source_concept_id", "visit_occurrence_id"],
        )

    def drug_table(self):
        return self.spark.createDataFrame(
            [(str(i), 1, 100 + i, "2020-01-01", "2020-01-01 10:00:00", i, 1) for i in (9, 10)],
            ["drug_exposure_id", "person_id", "drug_concept_id", "drug_exposure_start_date",
             "drug_exposure_start_datetime", "drug_source_concept_id", "visit_occurrence_id"],
        )

    def test_conditions(self):
        self.assertEqual(
            self.tokens(self.condition_table(), ethos_code_tokens=True),
            {
                "1": ["ICD//CM//I21", "ICD//CM//3-6//9"],
                "2": ["ICD//CM//S72", "ICD//CM//3-6//001", "ICD//CM//SFX//A"],
                # ICD-9 is not converted to ICD-10
                "3": ["ICD9CM/0/428", "ICD9CM/1/22"],
                "4": ["104"],
                # the malformed ICD-10-CM code is dropped
            },
        )

    def test_conditions_without_ethos_tokens_are_unchanged(self):
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

    def test_procedures(self):
        self.assertEqual(
            self.tokens(self.procedure_table(), ethos_code_tokens=True),
            {
                "6": ["ICD//PCS//0", "ICD//PCS//D", "ICD//PCS//T", "ICD//PCS//J", "ICD//PCS//4",
                      "ICD//PCS//Z", "ICD//PCS//Z"],
                "8": ["108"],
            },
        )

    def test_drugs(self):
        self.assertEqual(
            self.tokens(self.drug_table(), ethos_code_tokens=True),
            {
                "9": ["ATC//N02", "ATC//4//B", "ATC//SFX//E01"],
                "10": ["110"],
            },
        )


if __name__ == "__main__":
    unittest.main()

import bz2
import gzip
import os
import shutil
import subprocess
import tempfile
import unittest

from pyspark.sql import SparkSession

from cehrbert_data.utils.omop_tables import find_omop_table, omop_table_exists, read_omop_table
from cehrbert_data.utils.spark_utils import preprocess_domain_table

PERSON_CSV = (
    "person_id,gender_concept_id,year_of_birth,birth_datetime\n"
    "1,8507,1980,1980-01-02 00:00:00\n"
    "2,8532,1990,\n"
)
CONCEPT_TSV = (
    "concept_id\tconcept_name\tvocabulary_id\tconcept_code\n"
    '1\t"Quoted" name, with a comma\tSNOMED\t1\n'
    "2\tOther\tICD10CM\tI21.9\n"
)


class OmopTablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.appName("OmopTablesUnitTest")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def write(self, name, content, opener=open):
        with opener(os.path.join(self.folder, name), "wt") as f:
            f.write(content)

    def person_ids(self, df):
        return sorted(row["person_id"] for row in df.collect())

    def test_csv(self):
        self.write("person.csv", PERSON_CSV)
        df = read_omop_table(self.spark, self.folder, "person")
        self.assertEqual(self.person_ids(df), [1, 2])
        # the types are inferred, the missing value is null
        self.assertEqual(dict(df.dtypes)["gender_concept_id"], "int")
        self.assertEqual(df.where("person_id = 2").collect()[0]["birth_datetime"], None)

    def test_compressed_csv(self):
        self.write("person.csv.gz", PERSON_CSV, gzip.open)
        self.assertEqual(self.person_ids(read_omop_table(self.spark, self.folder, "person")), [1, 2])
        os.remove(os.path.join(self.folder, "person.csv.gz"))
        self.write("person.csv.bz2", PERSON_CSV, bz2.open)
        self.assertEqual(self.person_ids(read_omop_table(self.spark, self.folder, "person")), [1, 2])

    def test_tab_separated_file_of_athena_with_quotes_in_the_text(self):
        self.write("CONCEPT.csv", CONCEPT_TSV)
        df = read_omop_table(self.spark, self.folder, "concept")
        names = {row["concept_id"]: row["concept_name"] for row in df.collect()}
        self.assertEqual(names, {1: '"Quoted" name, with a comma', 2: "Other"})

    def test_folder_with_csv_files(self):
        os.makedirs(os.path.join(self.folder, "person"))
        self.write("person/part-1.csv", PERSON_CSV)
        self.assertEqual(find_omop_table(self.folder, "person")[0], "csv")
        self.assertEqual(self.person_ids(read_omop_table(self.spark, self.folder, "person")), [1, 2])

    def test_parquet_is_the_default(self):
        self.spark.createDataFrame([(1,), (2,)], "person_id long").write.parquet(
            os.path.join(self.folder, "person")
        )
        # a CSV file of the same table is ignored
        self.write("person.csv", PERSON_CSV.replace("2,8532", "3,8532"))
        self.assertEqual(find_omop_table(self.folder, "person")[0], "parquet")
        self.assertEqual(self.person_ids(read_omop_table(self.spark, self.folder, "person")), [1, 2])

    def test_one_file_per_table(self):
        self.write("person.csv", PERSON_CSV)
        self.write("person.csv.gz", PERSON_CSV, gzip.open)
        self.assertEqual(self.person_ids(read_omop_table(self.spark, self.folder, "person")), [1, 2])

    def test_table_exists(self):
        self.write("concept_ancestor.csv", "ancestor_concept_id,descendant_concept_id\n1,2\n")
        self.assertTrue(omop_table_exists(self.folder, "concept_ancestor"))
        # concept_ancestor.csv is not the concept table
        self.assertFalse(omop_table_exists(self.folder, "concept"))
        self.assertFalse(omop_table_exists(self.folder, "person"))

    def test_csv_without_header(self):
        self.write("person.csv", "1,8507,1980\n2,8532,1990\n")
        with self.assertRaises(ValueError):
            read_omop_table(self.spark, self.folder, "person")

    @unittest.skipIf(shutil.which("lzop") is None, "lzop is not installed")
    def test_lzo_is_decompressed_with_lzop(self):
        self.write("person.csv", PERSON_CSV)
        subprocess.run(["lzop", "-q", os.path.join(self.folder, "person.csv")], check=True)  # person.csv.lzo
        os.remove(os.path.join(self.folder, "person.csv"))
        self.assertEqual(find_omop_table(self.folder, "person")[0], "csv")
        self.assertEqual(self.person_ids(read_omop_table(self.spark, self.folder, "person")), [1, 2])
        # the decompressed file is kept and used the next time
        self.assertTrue(os.path.exists(os.path.join(self.folder, "person.csv")))

    def test_preprocess_domain_table_reads_csv(self):
        """The date columns are converted to dates like for the parquet tables."""
        self.write("person.csv", PERSON_CSV)
        df = preprocess_domain_table(self.spark, self.folder, "person")
        self.assertEqual(dict(df.dtypes)["birth_datetime"], "timestamp")
        self.assertEqual(self.person_ids(df), [1, 2])


if __name__ == "__main__":
    unittest.main()

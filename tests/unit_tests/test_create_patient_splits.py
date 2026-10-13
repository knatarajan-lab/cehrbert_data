import os
import shutil
import tempfile
import unittest

from pyspark.sql import SparkSession

from cehrbert_data.tools.create_patient_splits import PATIENT_SPLITS, create_patient_splits, main


class CreatePatientSplitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("CreatePatientSplitsUnitTest").getOrCreate()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def setUp(self):
        self.folder = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.folder, ignore_errors=True)

    def person(self, n=2000):
        return self.spark.range(1, n + 1).withColumnRenamed("id", "person_id")

    def splits(self, df):
        return {row["person_id"]: row["split"] for row in df.collect()}

    def test_every_patient_is_in_one_split(self):
        splits = self.splits(create_patient_splits(self.person(), train_ratio=0.8, seed=42))
        self.assertEqual(sorted(splits), list(range(1, 2001)))
        self.assertEqual(set(splits.values()), {"train", "test"})
        train_ratio = sum(1 for s in splits.values() if s == "train") / len(splits)
        self.assertAlmostEqual(train_ratio, 0.8, delta=0.05)

    def test_same_seed_same_split(self):
        first = self.splits(create_patient_splits(self.person(), seed=7))
        # the same patients in another partitioning and with duplicated rows
        shuffled = self.person().union(self.person(50)).repartition(7)
        self.assertEqual(self.splits(create_patient_splits(shuffled, seed=7)), first)
        self.assertNotEqual(self.splits(create_patient_splits(self.person(), seed=8)), first)

    def test_ratio(self):
        splits = self.splits(create_patient_splits(self.person(), train_ratio=0.5, seed=1))
        train_ratio = sum(1 for s in splits.values() if s == "train") / len(splits)
        self.assertAlmostEqual(train_ratio, 0.5, delta=0.05)
        for invalid in (0, 1, -0.1, 1.5):
            with self.assertRaises(ValueError):
                create_patient_splits(self.person(10), train_ratio=invalid)

    def test_main_reads_a_csv_person_table_and_writes_patient_splits(self):
        with open(os.path.join(self.folder, "person.csv"), "w") as f:
            f.write("person_id,gender_concept_id\n" + "".join(f"{i},8507\n" for i in range(1, 201)))
        main(self.folder)
        written = self.spark.read.parquet(os.path.join(self.folder, PATIENT_SPLITS))
        self.assertEqual(written.columns, ["person_id", "split"])
        self.assertEqual(sorted(row["person_id"] for row in written.collect()), list(range(1, 201)))
        # an existing patient_splits is not replaced unless asked
        with self.assertRaises(FileExistsError):
            main(self.folder)
        main(self.folder, train_ratio=0.5, overwrite=True)

    def test_main_without_a_person_table(self):
        with self.assertRaises(FileNotFoundError):
            main(self.folder)


if __name__ == "__main__":
    unittest.main()

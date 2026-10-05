import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

from cehrbert_data.utils import icd_cm_tokens as icd_cm_tokens_module
from cehrbert_data.utils.icd_cm_tokens import icd_cm_tokens
from cehrbert_data.utils.icd_pcs_tokens import icd_pcs_tokens
from cehrbert_data.utils.spark_utils import split_atc_code


class SparkWorkersTest(unittest.TestCase):
    """What the workers of Spark need: a function is sent to them by its name if it is defined in a module, and by
    value otherwise, which the cloudpickle of PySpark 3.1 can't do with the bytecode of Python 3.11 and later
    (IndexError: tuple index out of range when it is sent, TypeError: code() argument 13 must be str, not int when it
    is read). And the data files have to be readable where the package is."""

    def test_split_atc_code(self):
        self.assertEqual(split_atc_code("A10BA02"), ["A10", "B", "A02"])
        self.assertEqual(split_atc_code("A10B"), ["A10", "B"])
        self.assertEqual(split_atc_code("A10"), ["A10"])
        self.assertIsNone(split_atc_code(None))
        # a function of the module and not a function inside another function
        self.assertEqual(split_atc_code.__qualname__, "split_atc_code")
        self.assertEqual(split_atc_code.__module__, "cehrbert_data.utils.spark_utils")

    def test_resources_are_read_when_the_package_is_a_zip(self):
        """The workers can get the package as a zip (spark-submit --py-files), where the data files are not files of a
        directory."""
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(icd_cm_tokens_module.__file__)))
        files = [
            "__init__.py", "utils/__init__.py", "utils/icd_cm_tokens.py", "utils/icd_pcs_tokens.py",
            "resources/icd10cm-order-Jan-2021.csv.gz", "resources/icd_cm_9_to_10_mapping.csv.gz",
            "resources/icd_pcs_9_to_10_mapping.csv.gz",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "cehrbert_data.zip")
            with zipfile.ZipFile(zip_path, "w") as zip_file:
                for file in files:
                    zip_file.write(os.path.join(package_dir, file), f"cehrbert_data/{file}")
            code = (
                "import cehrbert_data, json\n"
                "from cehrbert_data.utils.icd_cm_tokens import icd_cm_tokens\n"
                "from cehrbert_data.utils.icd_pcs_tokens import icd_pcs_tokens\n"
                "print(json.dumps([cehrbert_data.__file__, icd_cm_tokens('ICD9CM', '428.22'), "
                "icd_pcs_tokens('ICD9Proc', '00.10')]))\n"
            )
            result = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": zip_path}
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            module_file, cm_tokens, pcs_tokens = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertTrue(module_file.startswith(zip_path), module_file)
            self.assertEqual(cm_tokens, list(icd_cm_tokens("ICD9CM", "428.22")))
            self.assertEqual(pcs_tokens, list(icd_pcs_tokens("ICD9Proc", "00.10")))


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
import tempfile
from pathlib import Path
from abc import abstractmethod, ABC


class PySparkAbstract(unittest.TestCase, ABC):

    @classmethod
    def setUpClass(cls):
        # Set PySpark and Java environment variables
        os.environ['SPARK_HOME'] = os.path.dirname(__import__('pyspark').__file__)
        os.environ['PYSPARK_PYTHON'] = sys.executable
        os.environ['SPARK_WORKER_INSTANCES'] = "1"
        os.environ['SPARK_WORKER_CORES'] = "4"
        os.environ['SPARK_EXECUTOR_CORES'] = "2"
        os.environ['SPARK_DRIVER_MEMORY'] = "1g"
        os.environ['SPARK_EXECUTOR_MEMORY'] = "1g"

    def setUp(self):
        from pyspark.sql import SparkSession
        self.spark = SparkSession.builder.master("local").appName("test").getOrCreate()
        # Get the root folder of the project
        root_folder = Path(os.path.abspath(__file__)).parent.parent
        self.data_folder = os.path.join(root_folder, "sample_data", "omop_sample")
        # Create a temporary directory to store model and tokenizer
        self.temp_dir = tempfile.mkdtemp()
        self.output_folder = os.path.join(self.temp_dir, "output_folder")
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def test_run_pyspark_app(self):
        pass

    def get_sample_data_folder(self):
        return self.data_folder

    def get_output_folder(self):
        return self.output_folder

    def tearDown(self):
        self.spark.stop()

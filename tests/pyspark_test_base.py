import os
import sys
import tempfile
import unittest
from abc import ABC, abstractmethod
from pathlib import Path


class PySparkAbstract(unittest.TestCase, ABC):

    @classmethod
    def setUpClass(cls):
        # Set PySpark and Java environment variables
        os.environ["SPARK_HOME"] = os.path.dirname(__import__("pyspark").__file__)
        os.environ["PYSPARK_PYTHON"] = sys.executable
        os.environ["SPARK_WORKER_INSTANCES"] = "1"
        os.environ["SPARK_WORKER_CORES"] = "4"
        os.environ["SPARK_EXECUTOR_CORES"] = "2"
        os.environ["SPARK_DRIVER_MEMORY"] = "2g"
        os.environ["SPARK_EXECUTOR_MEMORY"] = "2g"

    def setUp(self):
        from pyspark.sql import SparkSession

        # The error InaccessibleObjectException: Unable to make private java.nio.DirectByteBuffer(long,int) accessible
        # occurs because the PySpark code is trying to make a private constructor accessible, which is prohibited by
        # the Java module system for security reasons.
        # Starting from Java 9, JPMS enforces strong encapsulation of Java modules unless explicitly opened up.
        # This means that unless the java.base module explicitly opens the java.nio package to your application,
        # reflection on its classes and members will be blocked.
        # Add JVM Options: If you must use a newer version of Java, you can try adding JVM options to open up the
        # necessary modules. This is done by adding arguments to the spark.driver.extraJavaOptions and
        # spark.executor.extraJavaOptions in your Spark configuration:
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

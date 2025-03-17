import os
import sys
import tempfile
import shutil
import unittest
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
import uuid


@contextmanager
def spark_temp_folder(base_dir=None, prefix="spark_test_", cleanup=True):
    """
    Context manager for creating and cleaning up temporary folders for Spark tests.

    Args:
        base_dir (str, optional): Base directory to create temp folder in. Defaults to None (system temp).
        prefix (str, optional): Prefix for the temp folder name. Defaults to "spark_test_".
        cleanup (bool, optional): Whether to clean up the folder after use. Defaults to True.

    Yields:
        str: Path to the temporary folder
    """
    unique_id = uuid.uuid4().hex
    if base_dir:
        temp_dir = os.path.join(base_dir, f"{prefix}{unique_id}")
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp(prefix=prefix)

    try:
        yield temp_dir
    finally:
        if cleanup and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to clean up temp directory {temp_dir}: {e}")


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
        self.spark = SparkSession.builder \
            .master("local") \
            .appName("test") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.sql.files.maxPartitionBytes", "1048576") \
            .config("spark.sql.parquet.filterPushdown", "true") \
            .config("spark.ui.enabled", "false") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()

        # Get the root folder of the project
        root_folder = Path(os.path.abspath(__file__)).parent.parent
        self.data_folder = os.path.join(root_folder, "sample_data", "omop_sample")

        # Create and store the context manager
        self._temp_folder_context = spark_temp_folder(cleanup=True)
        # Start the context and get the directory
        self.temp_dir = self._temp_folder_context.__enter__()

        # Create subdirectories
        self.output_folder = os.path.join(self.temp_dir, "output_folder")
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        # Set up checkpoint directory
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.spark.sparkContext.setCheckpointDir(checkpoint_dir)

    @abstractmethod
    def test_run_pyspark_app(self):
        pass

    def get_sample_data_folder(self):
        return self.data_folder

    def get_output_folder(self):
        return self.output_folder

    def tearDown(self):
        # Stop Spark
        if hasattr(self, 'spark') and self.spark is not None:
            self.spark.stop()

        # Exit the context manager to clean up
        if hasattr(self, '_temp_folder_context'):
            self._temp_folder_context.__exit__(None, None, None)

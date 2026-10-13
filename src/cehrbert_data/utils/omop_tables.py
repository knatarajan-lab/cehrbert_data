"""Read the OMOP tables of an input folder, which can be stored as parquet or as CSV files.

The table <name> is looked up in the input folder (the names of the files are not case sensitive) in this order:

1. the folder <name> with parquet files (the output of tools/download_omop_tables.py), which is the default format
2. the folder <name> with CSV files
3. the file <name>.csv or the compressed file <name>.csv.gz, .csv.bz2, .csv.lz4, .csv.snappy, .csv.deflate, .csv.lzo

The CSV files must have a header row. The delimiter (comma, tab, semicolon or pipe) is detected from the header, so both
the CSV files of the ETLs (e.g. the Synthea OMOP files) and the tab separated files of Athena (e.g. CONCEPT.csv) work.

A .lzo file can only be read by Spark with the Hadoop LZO codec. If the codec is not available, the file is decompressed
with the lzop command (e.g. brew install lzop, apt install lzop) to <name>.csv in the input folder, which is also
splittable and therefore read in parallel.
"""

import logging
import os
import re
import shutil
import subprocess
from typing import List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession

LOG = logging.getLogger(__name__)

PARQUET = "parquet"
CSV = "csv"

# The compressions Spark reads out of the box, which are not splittable except for bz2 and lz4 (hadoop frame)
COMPRESSED_CSV_SUFFIXES = (".csv.gz", ".csv.bz2", ".csv.lz4", ".csv.snappy", ".csv.deflate")
LZO_SUFFIX = ".csv.lzo"
LZO_CODEC_CLASS = "com.hadoop.compression.lzo.LzopCodec"
DELIMITERS = ["\t", ",", "|", ";"]
HEADER_FIELD = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_csv_file(file_name: str) -> bool:
    name = file_name.lower()
    return name.endswith(".csv") or name.endswith(COMPRESSED_CSV_SUFFIXES + (LZO_SUFFIX,))


def _list_dir(folder: str) -> List[str]:
    try:
        return sorted(os.listdir(folder))
    except OSError:
        return []


def _find_entry(input_folder: str, table_name: str) -> Optional[str]:
    """The entry of the input folder that is the folder <name>, regardless of the case."""
    for entry in _list_dir(input_folder):
        if entry.lower() == table_name.lower() and os.path.isdir(os.path.join(input_folder, entry)):
            return os.path.join(input_folder, entry)
    return None


def _find_csv_files(input_folder: str, table_name: str) -> List[str]:
    """The files <name>.csv and <name>.csv.<compression>, regardless of the case."""
    prefix = table_name.lower()
    return [
        os.path.join(input_folder, entry)
        for entry in _list_dir(input_folder)
        if _is_csv_file(entry)
        and os.path.isfile(os.path.join(input_folder, entry))
        and entry.lower()[: len(prefix)] == prefix
        and entry.lower()[len(prefix):].startswith(".csv")
    ]


def find_omop_table(input_folder: str, table_name: str) -> Optional[Tuple[str, List[str]]]:
    """
    Find the table in the input folder.

    :return: (format, paths) where the format is "parquet" or "csv", or None if the table is not in the input folder
    """
    table_folder = _find_entry(input_folder, table_name)
    if table_folder is not None:
        files = _list_dir(table_folder)
        csv_files = [os.path.join(table_folder, f) for f in files if _is_csv_file(f)]
        has_parquet = any(f.lower().endswith(".parquet") or f in ("_SUCCESS", "_metadata") for f in files)
        if csv_files and not has_parquet:
            return CSV, csv_files
        return PARQUET, [table_folder]
    csv_files = _find_csv_files(input_folder, table_name)
    if csv_files:
        # The table could be stored more than once (e.g. <name>.csv and <name>.csv.lzo), read only one of the files so
        # that the records are not duplicated. The uncompressed file goes first, it is read in parallel.
        suffixes = (".csv",) + COMPRESSED_CSV_SUFFIXES + (LZO_SUFFIX,)
        csv_files.sort(key=lambda f: next(i for i, suffix in enumerate(suffixes) if f.lower().endswith(suffix)))
        return CSV, csv_files[:1]
    return None


def omop_table_exists(input_folder: str, table_name: str) -> bool:
    return find_omop_table(input_folder, table_name) is not None


def _detect_delimiter(header: str) -> str:
    counts = {delimiter: header.count(delimiter) for delimiter in DELIMITERS}
    delimiter = max(DELIMITERS, key=lambda d: counts[d])
    if counts[delimiter] == 0:
        raise ValueError(f"Cannot detect the delimiter of the CSV header: {header[:200]!r}")
    return delimiter


def _check_header(table_name: str, path: str, header: str, delimiter: str) -> None:
    fields = [field.strip().strip('"').lower() for field in header.split(delimiter)]
    if not all(HEADER_FIELD.match(field) for field in fields):
        raise ValueError(
            f"The first line of {path} is not a header row of column names, but the CSV files of the OMOP table "
            f"{table_name} must have one: {header[:200]!r}"
        )


def _lzo_codec_available(spark: SparkSession) -> bool:
    try:
        spark._jvm.java.lang.Class.forName(LZO_CODEC_CLASS)
    except Exception:  # the class is not on the classpath
        return False
    codecs = spark._jsc.hadoopConfiguration().get("io.compression.codecs") or ""
    return LZO_CODEC_CLASS in codecs


def _decompress_lzo(path: str) -> str:
    """Decompress <name>.csv.lzo to <name>.csv next to it with lzop, unless it has been decompressed already."""
    target = path[: -len(".lzo")]
    if os.path.exists(target):
        LOG.info("Using %s, which has been decompressed from %s", target, path)
        return target
    if shutil.which("lzop") is None:
        raise RuntimeError(
            f"Spark cannot read {path} without the Hadoop LZO codec and the lzop command is not installed to "
            f"decompress it. Install lzop (brew install lzop or apt install lzop), or decompress the file to "
            f"{target} yourself."
        )
    LOG.info("Decompressing %s to %s", path, target)
    temp_target = f"{target}.tmp"
    try:
        with open(temp_target, "wb") as f:
            subprocess.run(["lzop", "-dc", path], stdout=f, check=True)
        os.replace(temp_target, target)
    finally:
        if os.path.exists(temp_target):
            os.remove(temp_target)
    return target


def _read_csv(spark: SparkSession, table_name: str, paths: List[str]) -> DataFrame:
    prepared = []
    for path in paths:
        if path.lower().endswith(LZO_SUFFIX):
            path = path if _lzo_codec_available(spark) else _decompress_lzo(path)
        prepared.append(path)

    header = spark.read.text(prepared[0]).limit(1).collect()
    if not header:
        raise ValueError(f"{prepared[0]} is empty")
    header = header[0][0]
    delimiter = _detect_delimiter(header)
    _check_header(table_name, prepared[0], header, delimiter)

    reader = (
        spark.read.option("header", True)
        .option("sep", delimiter)
        .option("inferSchema", True)
        .option("mode", "FAILFAST")
    )
    if delimiter == "\t":
        # The tab separated files of Athena do not quote the text, which can contain double quotes
        reader = reader.option("quote", "")
    else:
        reader = reader.option("escape", '"')
    df = reader.csv(prepared)
    if any(path.lower().endswith(COMPRESSED_CSV_SUFFIXES + (LZO_SUFFIX,)) for path in prepared):
        # A compressed file is not always splittable, so it is read by one task
        df = df.repartition(spark.sparkContext.defaultParallelism * 4)
    return df


def read_omop_table(spark: SparkSession, input_folder: str, table_name: str) -> DataFrame:
    """
    Read the OMOP table from the input folder, in the format it is stored in (see the module docstring).

    The parquet folder is the default, so a missing table is reported by Spark as before.
    """
    found = find_omop_table(input_folder, table_name)
    if found is None or found[0] == PARQUET:
        path = found[1][0] if found else os.path.join(input_folder, table_name)
        return spark.read.parquet(path)
    return _read_csv(spark, table_name, found[1])

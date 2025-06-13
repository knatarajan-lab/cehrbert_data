#!/usr/bin/env python3
import pandas as pd
import os
import glob
import time
import argparse
import multiprocessing
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor

try:
    import polars as pl
except ImportError:
    print("Polars is required. Install it with: pip install polars")
    exit(1)


def convert_file(file_pair):
    """
    Convert a single Parquet file from ZStd to Snappy compression,
    converting datetime columns to ISO8601 string to avoid timezone issues.
    """
    input_file, output_file = file_pair
    try:
        # Read with Polars
        df = pl.read_parquet(input_file)

        # Convert datetime to string (ISO format)
        df = df.with_columns([
            pl.col("prediction_time")
            .cast(pl.Datetime("us"))
            .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            .alias("prediction_time")
        ])

        # Convert to pandas for writing
        pdf = df.to_pandas()

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write using pandas (PyArrow) to preserve strings
        pdf.to_parquet(output_file, compression="snappy", engine="pyarrow")

        return True, input_file
    except Exception as e:
        return False, f"{input_file} - Error: {str(e)}"


def process_directory(input_path, output_path, workers=None):
    """
    Process all Parquet files in a directory (recursively)
    """
    # Find all Parquet files
    parquet_files = []
    for pattern in ["**/*.parquet", "**/*.pq"]:
        parquet_files.extend(list(Path(input_path).glob(pattern)))

    if not parquet_files:
        print(f"No Parquet files found in {input_path}")
        return

    print(f"Found {len(parquet_files)} Parquet files to process")

    # Create a list of (input_file, output_file) pairs
    file_pairs = []
    for input_file in parquet_files:
        # Calculate the relative path from the input directory
        rel_path = os.path.relpath(input_file, input_path)

        # Create the corresponding output path
        output_file = os.path.join(output_path, rel_path)

        file_pairs.append((str(input_file), output_file))

    # Use a process pool to convert files in parallel
    if workers is None:
        workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Starting conversion with {workers} workers")
    start_time = time.time()

    success_count = 0
    failure_count = 0
    failures = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, (success, message) in enumerate(
                executor.map(convert_file, file_pairs)
        ):
            if success:
                success_count += 1
                if success_count % 100 == 0 or success_count == 1:
                    elapsed = time.time() - start_time
                    print(f"Processed {success_count}/{len(file_pairs)} files ({elapsed:.2f}s elapsed)")
            else:
                failure_count += 1
                failures.append(message)
                print(f"Failed: {message}")

    # Print summary
    elapsed = time.time() - start_time
    print("\n==== Summary ====")
    print(f"Total files: {len(file_pairs)}")
    print(f"Successfully converted: {success_count}")
    print(f"Failed: {failure_count}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average time per file: {elapsed / max(1, len(file_pairs)):.4f} seconds")

    if failures:
        print("\nFailed files:")
        for failure in failures:
            print(f"  - {failure}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ZStd-compressed Parquet files to Snappy compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input", "-i", required=True, help="Input directory containing Parquet files")
    parser.add_argument("--output", "-o", required=True, help="Output directory for converted files")
    parser.add_argument("--workers", "-w", type=int, help="Number of worker processes (default: CPU count - 1)")

    args = parser.parse_args()

    # Validate input path
    if not os.path.exists(args.input):
        print(f"Error: Input path does not exist: {args.input}")
        return 1

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Process the directory
    process_directory(args.input, args.output, args.workers)

    return 0


if __name__ == "__main__":
    exit(main())
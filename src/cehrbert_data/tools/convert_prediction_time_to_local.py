import pandas as pd
from pathlib import Path
import argparse


def process_and_save_parquet(
        input_path: Path,
        output_path: Path,
        local_timezone: str
):
    df = pd.read_parquet(input_path)
    if "prediction_time" in df.columns:
        # Ensure timezone-aware: if naive, assume UTC
        if pd.api.types.is_datetime64_any_dtype(df["prediction_time"]):
            if df["prediction_time"].dt.tz is None:
                df["prediction_time"] = df["prediction_time"].dt.tz_localize("UTC")
            # Convert to local timezone
            df["prediction_time"] = df["prediction_time"].dt.tz_convert(local_timezone)
            # Remove timezone information after conversion
            df["prediction_time"] = df["prediction_time"].dt.tz_localize(None)
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved: {output_path}")


def main(input_dir: Path, output_dir: Path, local_timezone: str):
    for input_path in input_dir.rglob("*.parquet"):
        rel_path = input_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        process_and_save_parquet(input_path, output_path, local_timezone)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively update prediction_time timezone in parquet files.")
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        type=Path,
        help="Path to input directory containing parquet files."
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=Path,
        help="Path to output directory for processed files."
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="America/New_York",
        help="Target timezone (default: America/New_York)."
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.timezone)

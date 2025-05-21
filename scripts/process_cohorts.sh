#!/bin/sh

# Script: process_cohorts.sh
# Description: Process prediction_time in parquet files across multiple cohort directories
# Usage: ./process_cohorts.sh --input_dir <path> --output_dir <path> [--timezone <timezone>]

# Default values
INPUT_DIR=""
OUTPUT_DIR=""
TIMEZONE="America/New_York"

# Function to show usage information
show_usage() {
  echo "Usage: $0 --input_dir <path> --output_dir <path> [--timezone <timezone>]"
  echo
  echo "Arguments:"
  echo "  --input_dir <path>     Base directory containing cohort folders with parquet files"
  echo "  --output_dir <path>    Base directory where processed cohort folders will be created"
  echo "  --timezone <timezone>  Optional: Target timezone (default: America/New_York)"
  echo
  echo "Examples:"
  echo "  $0 --input_dir /data/raw_cohorts --output_dir /data/processed_cohorts"
  echo "  $0 --input_dir /data/raw_cohorts --output_dir /data/processed_cohorts --timezone America/Los_Angeles"
  echo
  echo "This script will:"
  echo "  1. Find all cohort folders in the input directory"
  echo "  2. Process all parquet files in each cohort"
  echo "  3. Convert prediction_time to the specified timezone"
  echo "  4. Save processed files to the output directory, maintaining the original structure"
}

# Parse command line arguments
while [ $# -gt 0 ]; do
  key="$1"
  case $key in
    --input_dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --timezone)
      TIMEZONE="$2"
      shift 2
      ;;
    --help)
      show_usage
      exit 0
      ;;
    *)
      echo "Error: Unknown option: $1"
      show_usage
      exit 1
      ;;
  esac
done

# Check required arguments
if [ -z "$INPUT_DIR" ]; then
  echo "Error: --input_dir is required"
  show_usage
  exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
  echo "Error: --output_dir is required"
  show_usage
  exit 1
fi

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
  echo "Error: Input directory does not exist: $INPUT_DIR"
  exit 1
fi

# Ensure output base directory exists
mkdir -p "$OUTPUT_DIR"

echo "Starting processing of cohorts..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Timezone: $TIMEZONE"
echo

# Find all directories directly under the input base (these are our cohorts)
COHORT_COUNT=0
for COHORT_PATH in "$INPUT_DIR"/*/ ; do
  if [ -d "$COHORT_PATH" ]; then
    # Extract cohort name from path
    COHORT_NAME=$(basename "$COHORT_PATH")
    COHORT_OUTPUT_DIR="$OUTPUT_DIR/$COHORT_NAME"

    echo "Processing cohort: $COHORT_NAME"

    # Process all parquet files in this cohort using the Python module
    # Note: Using positional arguments for input_dir and output_dir as per Python script
    python -m cehrbert_data.tools.convert_prediction_time_to_local \
      --input_dir "$COHORT_PATH" \
      --output_dir "$COHORT_OUTPUT_DIR" \
      --timezone "$TIMEZONE"

    # Check if Python script executed successfully
    if [ $? -eq 0 ]; then
      echo "✓ Successfully processed cohort: $COHORT_NAME"
    else
      echo "✗ Error processing cohort: $COHORT_NAME"
    fi

    COHORT_COUNT=$((COHORT_COUNT + 1))
    echo
  fi
done

if [ $COHORT_COUNT -eq 0 ]; then
  echo "Warning: No cohort directories found in $INPUT_DIR"
  exit 0
else
  echo "Processing complete! Processed $COHORT_COUNT cohort(s)"
fi
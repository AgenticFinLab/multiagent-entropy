import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any


def extract_summary_fields(
    input_csv_path: Path, output_csv_path: Path
) -> List[Dict[str, Any]]:
    """Extract summary fields from aggregated CSV file.

    Args:
        input_csv_path: Path to the input aggregated CSV file
        output_csv_path: Path to the output summary CSV file

    Returns:
        List of summary records
    """
    # Check if input file exists
    if not input_csv_path.exists():
        print(f"Error: Input file {input_csv_path} does not exist")
        return []

    # Define fields to extract from the aggregated data
    summary_fields = [
        "model_name",
        "architecture",
        "exp_total_entropy",
        "exp_infer_average_entropy",
        "exp_accuracy",
        "exp_format_compliance_rate",
        "exp_total_time",
        "base_model_accuracy",
        "base_model_format_compliance_rate",
    ]

    # Initialize list to store summary records
    summary_records = []
    # Track seen keys to avoid duplicates
    seen_keys = set()

    # Define fields that should be treated as numeric values
    numeric_fields = [
        "exp_total_entropy",
        "exp_infer_average_entropy",
        "exp_accuracy",
        "exp_format_compliance_rate",
        "exp_total_time",
        "base_model_accuracy",
        "base_model_format_compliance_rate",
    ]

    # Open and read the input CSV file
    with open(input_csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        # Process each row in the CSV file
        for row in reader:
            # Create unique key from model name, experiment name, and architecture
            key = (row["model_name"], row["experiment_name"], row["architecture"])

            # Skip if this key has already been processed
            if key in seen_keys:
                continue

            # Mark this key as seen
            seen_keys.add(key)

            # Create summary record with selected fields
            summary_record = {}
            for field in summary_fields:
                # Get value from row, default to empty string
                value = row.get(field, "")
                # Convert numeric fields to float and round to 3 decimal places
                if field in numeric_fields and value:
                    try:
                        summary_record[field] = round(float(value), 3)
                    except (ValueError, TypeError):
                        summary_record[field] = value
                else:
                    summary_record[field] = value
            # Add summary record to list
            summary_records.append(summary_record)

    # Check if any records were found
    if not summary_records:
        print("No records found in the input file")
        return []

    # Create output directory if it doesn't exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Write summary records to output CSV file
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fields)
        # Write CSV header
        writer.writeheader()
        # Write all summary records
        writer.writerows(summary_records)

    # Print success message
    print(f"Successfully wrote {len(summary_records)} summary records to {output_csv_path}")

    return summary_records


def main():
    """Main function to extract summary fields from aggregated CSV."""
    # Create argument parser with description
    parser = argparse.ArgumentParser(
        description="Extract summary fields from aggregated CSV file"
    )
    # Add input argument for specifying custom input CSV path
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the input aggregated CSV file. If not provided, uses evaluation/results/{dataset}/all_aggregated_data.csv",
    )
    # Add output argument for specifying custom output CSV path
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output summary CSV file. If not provided, uses evaluation/results/{dataset}/all_summary_data.csv",
    )
    # Add dataset argument for default path construction
    parser.add_argument(
        "--dataset",
        type=str,
        default="aime2025",
        help="Dataset name (used if input/output paths are not provided)",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Construct base path for evaluation results
    base_path = Path(__file__).parent.parent / "evaluation" / "results"

    # Determine input path from argument or default
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = base_path / args.dataset / "all_aggregated_data.csv"

    # Determine output path from argument or default
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = base_path / args.dataset / "all_summary_data.csv"

    # Print input and output paths for user information
    print(f"Extracting summary from: {input_path}")
    print(f"Output will be saved to: {output_path}")

    # Extract summary fields from input CSV and write to output CSV
    summary_records = extract_summary_fields(input_path, output_path)
    
if __name__ == "__main__":
    # Execute main function when script is run directly
    main()

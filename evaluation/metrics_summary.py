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
    if not input_csv_path.exists():
        print(f"Error: Input file {input_csv_path} does not exist")
        return []

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

    summary_records = []
    seen_keys = set()

    numeric_fields = [
        "exp_total_entropy",
        "exp_infer_average_entropy",
        "exp_accuracy",
        "exp_format_compliance_rate",
        "exp_total_time",
        "base_model_accuracy",
        "base_model_format_compliance_rate",
    ]

    with open(input_csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            key = (row["model_name"], row["experiment_name"], row["architecture"])

            if key in seen_keys:
                continue

            seen_keys.add(key)

            summary_record = {}
            for field in summary_fields:
                value = row.get(field, "")
                if field in numeric_fields and value:
                    try:
                        summary_record[field] = round(float(value), 3)
                    except (ValueError, TypeError):
                        summary_record[field] = value
                else:
                    summary_record[field] = value
            summary_records.append(summary_record)

    if not summary_records:
        print("No records found in the input file")
        return []

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_records)

    print(f"Successfully wrote {len(summary_records)} summary records to {output_csv_path}")

    return summary_records


def main():
    """Main function to extract summary fields from aggregated CSV."""
    parser = argparse.ArgumentParser(
        description="Extract summary fields from aggregated CSV file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the input aggregated CSV file. If not provided, uses evaluation/results/{dataset}/all_aggregated_data.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output summary CSV file. If not provided, uses evaluation/results/{dataset}/all_summary_data.csv",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aime2025",
        help="Dataset name (used if input/output paths are not provided)",
    )

    args = parser.parse_args()

    base_path = Path(__file__).parent.parent / "evaluation" / "results"

    if args.input:
        input_path = Path(args.input)
    else:
        input_path = base_path / args.dataset / "all_aggregated_data.csv"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = base_path / args.dataset / "all_summary_data.csv"

    print(f"Extracting summary from: {input_path}")
    print(f"Output will be saved to: {output_path}")

    summary_records = extract_summary_fields(input_path, output_path)
    
if __name__ == "__main__":
    main()

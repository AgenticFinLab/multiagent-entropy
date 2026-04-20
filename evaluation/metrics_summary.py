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
        "exp_accuracy",
        "base_model_accuracy",
        "base_model_mean_answer_token_entropy",
        "sample_mean_answer_token_entropy",
        "exp_total_time",
        "exp_total_token",
    ]

    # Initialize list to store summary records
    summary_records = []
    # Track seen keys to avoid duplicates
    seen_keys = set()

    # Define fields that should be treated as numeric values
    numeric_fields = [
        "exp_accuracy",
        "base_model_accuracy",
        "base_model_mean_answer_token_entropy",
        "sample_mean_answer_token_entropy",
        "exp_total_time",
        "exp_total_token",
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
                        num_value = float(value)
                        # Apply transformations according to requirements
                        if field == "exp_accuracy":
                            # Convert to percentage
                            summary_record[field] = round(num_value * 100, 3)
                        if field == "base_model_accuracy":
                            # Convert to percentage
                            summary_record[field] = round(num_value * 100, 3)
                        elif field == "exp_total_token":
                            # Convert to 100K units
                            summary_record[field] = round(num_value / 100000, 3)
                        else:
                            summary_record[field] = round(num_value, 3)
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
    
    # Define the desired column names in the final output for single dataset
    output_summary_fields = [
        "dataset",  # Add dataset field to the beginning
        "model_name",
        "architecture",
        "sample_mean_answer_token_entropy",  # entropy
        "exp_accuracy",  # accuracy (will be converted to percentage)
        "exp_total_token",  # token (will be converted to 100K units)
        "exp_total_time",  # time
        "base_model_accuracy",  # base model accuracy
        "base_model_mean_answer_token_entropy",  # base model entropy
    ]
    
    # Map the internal field names to the desired column names in output
    field_mapping = {
        "sample_mean_answer_token_entropy": "entropy",
        "exp_accuracy": "accuracy",
        "exp_total_token": "token",
        "exp_total_time": "time",
        "base_model_accuracy": "base model accuracy",
        "base_model_mean_answer_token_entropy": "base model entropy",
        "model_name": "model"
    }
    
    # Remap field names for the output
    output_fieldnames = []
    for field in output_summary_fields:
        output_fieldnames.append(field_mapping.get(field, field))
    
    # Transform records to use new field names
    transformed_records = []
    for record in summary_records:
        # Add empty dataset field for single dataset case
        record["dataset"] = ""  # Will be filled by the calling function if needed
        transformed_record = {}
        for field in output_summary_fields:
            new_field_name = field_mapping.get(field, field)
            transformed_record[new_field_name] = record.get(field, "")
        transformed_records.append(transformed_record)
    
    # Sort the records by model, then by architecture for consistent ordering
    transformed_records.sort(key=lambda x: (x.get('model', ''), x.get('architecture', '')))
    
    # Write summary records to output CSV file
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output_fieldnames)
        # Write CSV header
        writer.writeheader()
        # Write all transformed summary records
        writer.writerows(transformed_records)

    # Print success message
    print(f"Successfully wrote {len(summary_records)} summary records to {output_csv_path}")

    return summary_records


def extract_summary_fields_for_multiple_datasets(
    datasets: List[str], output_csv_path: Path
) -> List[Dict[str, Any]]:
    """Extract summary fields from multiple datasets and combine them.

    Args:
        datasets: List of dataset names to process
        output_csv_path: Path to the output summary CSV file

    Returns:
        List of combined summary records
    """
    all_summary_records = []
    
    # Process each dataset in the list
    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        
        # Construct paths for this dataset
        base_path = Path(__file__).parent.parent / "evaluation" / "results_R_2"
        input_path = base_path / dataset / "all_aggregated_data.csv"
        
        # Process the individual dataset
        if input_path.exists():
            dataset_records = extract_summary_fields(input_path, base_path / dataset / "all_summary_data.csv")
            
            # Add dataset information to each record
            for record in dataset_records:
                record["dataset"] = dataset  # Add dataset name to each record
                
            # Append to the combined list
            all_summary_records.extend(dataset_records)
        else:
            print(f"Warning: Input file does not exist for dataset {dataset}: {input_path}")
    
    # Create output directory if it doesn't exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Define the desired column names in the final output
    summary_fields = [
        "dataset",  # Add dataset field to the beginning
        "model_name",
        "architecture",
        "sample_mean_answer_token_entropy",  # entropy
        "exp_accuracy",  # accuracy (will be converted to percentage)
        "exp_total_token",  # token (will be converted to 100K units)
        "exp_total_time",  # time
        "base_model_accuracy",  # base model accuracy
        "base_model_mean_answer_token_entropy",  # base model entropy
    ]
    
    # Map the internal field names to the desired column names in output
    field_mapping = {
        "sample_mean_answer_token_entropy": "entropy",
        "exp_accuracy": "accuracy",
        "exp_total_token": "token",
        "exp_total_time": "time",
        "base_model_accuracy": "base model accuracy",
        "base_model_mean_answer_token_entropy": "base model entropy",
        "model_name": "model"
    }

    # Write combined summary records to output CSV file
    # Remap field names for the output
    output_fieldnames = []
    for field in summary_fields:
        output_fieldnames.append(field_mapping.get(field, field))
    
    # Transform records to use new field names
    transformed_records = []
    for record in all_summary_records:
        transformed_record = {}
        for field in summary_fields:
            new_field_name = field_mapping.get(field, field)
            transformed_record[new_field_name] = record.get(field, "")
        transformed_records.append(transformed_record)
    
    # Sort the records by dataset, model, then by architecture for consistent ordering
    transformed_records.sort(key=lambda x: (x.get('dataset', ''), x.get('model', ''), x.get('architecture', '')))
    
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output_fieldnames)
        # Write CSV header
        writer.writeheader()
        # Write all transformed summary records
        writer.writerows(transformed_records)

    # Print success message
    print(f"Successfully wrote {len(all_summary_records)} combined summary records from {len(datasets)} datasets to {output_csv_path}")

    return all_summary_records


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
        help="Path to the input aggregated CSV file. If not provided, uses evaluation/results_R_2/{dataset}/all_aggregated_data.csv",
    )
    # Add output argument for specifying custom output CSV path
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output summary CSV file. If not provided, uses evaluation/results_R_2/{dataset}/all_summary_data.csv",
    )
    # Add dataset argument for default path construction - now accepts single dataset or comma-separated list
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500,aime2025_16384",
        help="Dataset name or comma-separated list of dataset names to process",
    )
    # Add argument to control whether to generate architecture analysis CSV
    parser.add_argument(
        "--analyze_single",
        action="store_true",
        help="Generate single architecture performance analysis CSV",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Construct base path for evaluation results
    base_path = Path(__file__).parent.parent / "evaluation" / "results_R_2"

    # Determine datasets to process - support both single dataset and multiple datasets
    datasets = [ds.strip() for ds in args.dataset.split(',')] if ',' in args.dataset else [args.dataset]
    
    # Determine output path from argument or default
    if args.output:
        output_path = Path(args.output)
    else:
        # If processing multiple datasets, use a default name for the combined output
        if len(datasets) > 1:
            output_path = base_path / "combined_summary_data.csv"
        else:
            output_path = base_path / datasets[0] / "all_summary_data.csv"

    # Print input and output paths for user information
    if len(datasets) > 1:
        print(f"Extracting summary from multiple datasets: {datasets}")
    else:
        input_path = base_path / datasets[0] / "all_aggregated_data.csv"
        print(f"Extracting summary from: {input_path}")
    print(f"Output will be saved to: {output_path}")

    # Extract summary fields - use different method based on whether we have multiple datasets
    if len(datasets) > 1:
        summary_records = extract_summary_fields_for_multiple_datasets(datasets, output_path)
    else:
        input_path = base_path / datasets[0] / "all_aggregated_data.csv"
        summary_records = extract_summary_fields(input_path, output_path)

    
if __name__ == "__main__":
    # Execute main function when script is run directly
    main()

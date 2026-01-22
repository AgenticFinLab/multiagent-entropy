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
        base_path = Path(__file__).parent.parent / "evaluation" / "results"
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


def analyze_single_architecture_performance(
    input_csv_path: Path, output_csv_path: Path
) -> None:
    """Analyze single architecture performance compared to other architectures.
    
    Args:
        input_csv_path: Path to the input combined summary CSV file
        output_csv_path: Path to the output analysis CSV file
    """
    # Read the combined summary data
    if not input_csv_path.exists():
        print(f"Error: Input file {input_csv_path} does not exist")
        return
    
    records = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    
    # Group records by dataset and model
    grouped_data = {}
    for record in records:
        key = (record['dataset'], record['model'])
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(record)
    
    # Define architectures to analyze
    architectures = ['centralized', 'debate', 'hybrid', 'sequential', 'single']
    
    # Analyze each group (dataset, model)
    analysis_results = []
    best_single_count = 0  # Count of times single was the best
    superior_single_count = 0  # Count of times single was better than at least one other
    total_groups = 0
    
    # Collect entropy improvements for various scenarios
    entropy_improvements_when_best = []  # Entropy improvement when single is best
    entropy_comparisons_when_superior = []  # Entropy differences when single is superior to others
    
    # Also collect entropy absolute differences (not percentages)
    entropy_absolute_differences_when_best = []  # Absolute entropy difference when single is best
    entropy_absolute_differences_when_superior = []  # Absolute entropy difference when single is superior to others
    
    # Store records where single architecture was the best
    single_best_records = []
    
    for (dataset, model), group_records in grouped_data.items():
        # Filter to only include the 5 architectures we're interested in
        relevant_records = [r for r in group_records if r['architecture'] in architectures]
            
        total_groups += 1
        
        # Find the record with highest accuracy for this group
        best_record = max(relevant_records, key=lambda x: float(x['accuracy']) if x['accuracy'] else 0)
        
        # Find the single architecture record for this group
        single_record = None
        for record in relevant_records:
            if record['architecture'] == 'single':
                single_record = record
                break
        
        if single_record is None:
            continue
        
        # Check if single architecture was the best
        single_is_best = (best_record['architecture'] == 'single')
        if single_is_best:
            best_single_count += 1
            
            # Add the single record to our collection of best single records
            single_best_records.append(single_record)
            
            # Calculate entropy comparison when single is best vs other architectures
            single_entropy = float(single_record['entropy']) if single_record['entropy'] else 0
            
            # Compare with all other architectures
            for other_record in relevant_records:
                if other_record['architecture'] != 'single':
                    other_entropy = float(other_record['entropy']) if other_record['entropy'] else 0
                    if other_entropy != 0:
                        # Calculate the percentage change from other to single: (single - other) / other * 100%
                        # This matches the formula: (a-b)/b * 100%, (a-c)/c * 100%, etc.
                        entropy_change_pct = ((single_entropy - other_entropy) / other_entropy) * 100
                        entropy_improvements_when_best.append(entropy_change_pct)
                        # Also calculate absolute difference
                        absolute_diff = single_entropy - other_entropy
                        entropy_absolute_differences_when_best.append(absolute_diff)
        
        # Check if single architecture was better than at least one other
        single_accuracy = float(single_record['accuracy']) if single_record['accuracy'] else 0
        other_records_better_than = []
        for record in relevant_records:
            if record['architecture'] != 'single' and float(record['accuracy']) if record['accuracy'] else 0 < single_accuracy:
                other_records_better_than.append(record)
        
        single_superior_to_any = len(other_records_better_than) > 0
        if single_superior_to_any:
            superior_single_count += 1
            
            # Calculate entropy comparison when single is superior to others
            single_entropy = float(single_record['entropy']) if single_record['entropy'] else 0
            
            # Compare with architectures that single is superior to
            for other_record in other_records_better_than:
                other_entropy = float(other_record['entropy']) if other_record['entropy'] else 0
                if other_entropy != 0:
                    # Calculate the percentage change from other to single: (single - other) / other * 100%
                    # This matches the formula: (a-b)/b * 100%, (a-c)/c * 100%, etc.
                    entropy_change_pct = ((single_entropy - other_entropy) / other_entropy) * 100
                    entropy_comparisons_when_superior.append(entropy_change_pct)
                    # Also calculate absolute difference
                    absolute_diff = single_entropy - other_entropy
                    entropy_absolute_differences_when_superior.append(absolute_diff)
        
        # Record analysis for this group
        analysis_results.append({
            'dataset': dataset,
            'model': model,
            'single_is_best': single_is_best,
            'single_superior_to_any': single_superior_to_any,
            'single_accuracy': single_accuracy,
            'best_architecture': best_record['architecture'],
            'best_accuracy': float(best_record['accuracy']) if best_record['accuracy'] else 0
        })
    
    # Prepare summary statistics
    summary_stats = {
        'total_groups_analyzed': total_groups,
        'single_best_count': best_single_count,
        'single_superior_count': superior_single_count,
        'single_best_percentage': (best_single_count / total_groups * 100) if total_groups > 0 else 0,
        'single_superior_percentage': (superior_single_count / total_groups * 100) if total_groups > 0 else 0,
        'avg_entropy_change_from_others_to_single_when_best': sum(entropy_improvements_when_best) / len(entropy_improvements_when_best) if entropy_improvements_when_best else 0,
        'entropy_changes_when_best_count': len(entropy_improvements_when_best),
        'avg_entropy_change_from_others_to_single_when_superior': sum(entropy_comparisons_when_superior) / len(entropy_comparisons_when_superior) if entropy_comparisons_when_superior else 0,
        'entropy_changes_when_superior_count': len(entropy_comparisons_when_superior),
        'avg_entropy_absolute_difference_when_best': sum(entropy_absolute_differences_when_best) / len(entropy_absolute_differences_when_best) if entropy_absolute_differences_when_best else 0,
        'entropy_absolute_differences_when_best_count': len(entropy_absolute_differences_when_best),
        'avg_entropy_absolute_difference_when_superior': sum(entropy_absolute_differences_when_superior) / len(entropy_absolute_differences_when_superior) if entropy_absolute_differences_when_superior else 0,
        'entropy_absolute_differences_when_superior_count': len(entropy_absolute_differences_when_superior),
        # Descriptions for clarity
        'avg_entropy_change_from_others_to_single_when_best_desc': 'Average entropy change in percentage when single is best compared to other architectures',
        'avg_entropy_change_from_others_to_single_when_superior_desc': 'Average entropy change in percentage when single is superior to other architectures'
    }
    
    # Create output directory if it doesn't exist
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the analysis results to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'metric',
            'value'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write summary statistics
        for key, value in summary_stats.items():
            writer.writerow({
                'metric': key,
                'value': value
            })
    
    # Also save the single best records to a separate CSV file
    single_best_output_path = output_csv_path.parent / "single_architecture_best_records.csv"
    with open(single_best_output_path, 'w', newline='', encoding='utf-8') as f:
        if single_best_records:
            writer = csv.DictWriter(f, fieldnames=single_best_records[0].keys())
            writer.writeheader()
            writer.writerows(single_best_records)
    
    print(f"Analysis complete. Summary stats:")
    print(f"Total groups analyzed: {summary_stats['total_groups_analyzed']}")
    print(f"Times single was best: {summary_stats['single_best_count']} ({summary_stats['single_best_percentage']:.2f}%)")
    print(f"Times single was superior to at least one other: {summary_stats['single_superior_count']} ({summary_stats['single_superior_percentage']:.2f}%)")
    print(f"Avg entropy change from others to single when single was best: {summary_stats['avg_entropy_change_from_others_to_single_when_best']:.2f}% (based on {summary_stats['entropy_changes_when_best_count']} comparisons)")
    print(f"Avg entropy change from others to single when single was superior to others: {summary_stats['avg_entropy_change_from_others_to_single_when_superior']:.2f}% (based on {summary_stats['entropy_changes_when_superior_count']} comparisons)")
    print(f"Avg entropy absolute difference when single was best: {summary_stats['avg_entropy_absolute_difference_when_best']:.4f} (based on {summary_stats['entropy_absolute_differences_when_best_count']} comparisons)")
    print(f"Avg entropy absolute difference when single was superior to others: {summary_stats['avg_entropy_absolute_difference_when_superior']:.4f} (based on {summary_stats['entropy_absolute_differences_when_superior_count']} comparisons)")
    print(f"Saved {len(single_best_records)} single architecture best records to {single_best_output_path}")


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
    # Add dataset argument for default path construction - now accepts single dataset or comma-separated list
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k,humaneval,mmlu,math500,aime2024_16384,aime2025_16384",
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
    base_path = Path(__file__).parent.parent / "evaluation" / "results"

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
    
    analysis_output_path = base_path / "single_architecture_analysis.csv"
    print(f"Generating single architecture performance analysis...")
    analyze_single_architecture_performance(output_path, analysis_output_path)
    
if __name__ == "__main__":
    # Execute main function when script is run directly
    main()

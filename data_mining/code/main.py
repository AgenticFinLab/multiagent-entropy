import os
import sys
from pathlib import Path

from data_quality_assessment import DataQualityAssessor
from exploratory_data_analysis import ExploratoryDataAnalyzer
from feature_engineering import FeatureEngineer
from model_building import ModelBuilder
from pattern_identification import PatternIdentifier
from generate_report import ReportGenerator


def main():
    """Main function to run the complete data mining analysis pipeline."""

    print("=" * 80)
    print("MULTI-AGENT SYSTEM ENTROPY RESEARCH - DATA MINING ANALYSIS")
    print("=" * 80)

    # Define paths
    data_path = '/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/aggregated_data.csv'
    output_dir = '/home/yuxuanzhao/multiagent-entropy/data_mining'

    print(f"\nData path: {data_path}")
    print(f"Output directory: {output_dir}")

    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"\nERROR: Data file not found at {data_path}")
        sys.exit(1)

    # Step 1: Data Quality Assessment
    print("\n" + "=" * 80)
    print("STEP 1: DATA QUALITY ASSESSMENT")
    print("=" * 80)
    try:
        dqa = DataQualityAssessor(data_path, output_dir)
        dqa.load_data()
        dqa.identify_column_types()
        dqa.analyze_missing_values()
        dqa.detect_outliers_zscore()
        dqa.detect_outliers_iqr()
        dqa.analyze_distribution()
        dqa.generate_summary_report()
        print("\nData quality assessment completed successfully!")
    except Exception as e:
        print(f"\nERROR in data quality assessment: {e}")
        sys.exit(1)

    # Step 2: Exploratory Data Analysis
    print("\n" + "=" * 80)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    try:
        eda = ExploratoryDataAnalyzer(data_path, output_dir)
        eda.load_data()
        eda.identify_column_types()
        eda.analyze_architecture_distribution()
        eda.analyze_entropy_by_architecture()
        eda.analyze_entropy_performance_correlation()
        eda.analyze_entropy_dynamics_over_rounds()
        eda.analyze_agent_entropy_patterns()
        eda.generate_statistical_summary()
        print("\nExploratory data analysis completed successfully!")
    except Exception as e:
        print(f"\nERROR in exploratory data analysis: {e}")
        sys.exit(1)

    # Step 3: Feature Engineering
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 80)
    try:
        fe = FeatureEngineer(data_path, output_dir)
        fe.load_data()
        fe.extract_entropy_features()
        fe.handle_missing_values()
        fe.normalize_features()
        importance_df = fe.evaluate_feature_importance()
        fe.select_features(importance_df)
        fe.apply_pca()
        fe.generate_feature_report()
        print("\nFeature engineering completed successfully!")
    except Exception as e:
        print(f"\nERROR in feature engineering: {e}")
        sys.exit(1)

    # Step 4: Model Building
    print("\n" + "=" * 80)
    print("STEP 4: MODEL BUILDING")
    print("=" * 80)
    try:
        mb = ModelBuilder(data_path, output_dir)
        mb.load_data()
        X, y = mb.prepare_features_and_target()
        mb.split_data(X, y)
        mb.scale_features()
        mb.build_regression_models()
        mb.build_classification_models()
        mb.evaluate_regression_models()
        mb.evaluate_classification_models()
        mb.hyperparameter_tuning()
        mb.feature_importance_analysis()
        mb.generate_model_report()
        print("\nModel building completed successfully!")
    except Exception as e:
        print(f"\nERROR in model building: {e}")
        sys.exit(1)

    # Step 5: Pattern Identification
    print("\n" + "=" * 80)
    print("STEP 5: PATTERN IDENTIFICATION")
    print("=" * 80)
    try:
        pi = PatternIdentifier(data_path, output_dir)
        pi.load_data()
        pi.analyze_entropy_performance_correlation()
        pi.analyze_architecture_entropy_patterns()
        pi.analyze_entropy_performance_by_architecture()
        pi.identify_optimal_entropy_ranges()
        pi.analyze_entropy_dynamics_over_execution_order()
        pi.validate_patterns_with_statistical_tests()
        pi.generate_pattern_report()
        print("\nPattern identification completed successfully!")
    except Exception as e:
        print(f"\nERROR in pattern identification: {e}")
        sys.exit(1)

    # Step 6: Generate Comprehensive Report
    print("\n" + "=" * 80)
    print("STEP 6: GENERATE COMPREHENSIVE REPORT")
    print("=" * 80)
    try:
        rg = ReportGenerator(data_path, output_dir)
        rg.load_data()
        rg.load_previous_results()
        rg.generate_comprehensive_report()
        print("\nComprehensive report generated successfully!")
    except Exception as e:
        print(f"\nERROR in report generation: {e}")
        sys.exit(1)

    # Final summary
    print("\n" + "=" * 80)
    print("DATA MINING ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nAll results have been saved to: {output_dir}")
    print("\nDirectory structure:")
    print(f"  - {output_dir}/results/data_quality_summary/")
    print(f"  - {output_dir}/results/eda_results/")
    print(f"  - {output_dir}/results/feature_engineering/")
    print(f"  - {output_dir}/results/model_building/")
    print(f"  - {output_dir}/results/pattern_identification/")
    print(f"  - {output_dir}/reports/")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

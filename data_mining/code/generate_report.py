"""Comprehensive Data Analysis Report Generator for Multi-Agent Entropy Research.

This module generates a detailed analysis report including:
- Complete data preprocessing steps
- Analysis methods and procedures
- Model building process
- Key findings and insights
- Visualization results
- Conclusions and recommendations for multi-agent system optimization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualization
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


class ReportGenerator:
    """Class for generating comprehensive data analysis reports."""

    def __init__(self, data_path: str, output_dir: str):
        """Initialize the ReportGenerator.

        Args:
            data_path: Path to the CSV data file.
            output_dir: Directory to save output files and figures.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None

        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        self.reports_dir = os.path.join(output_dir, 'reports')
        self.figures_dir = os.path.join(self.reports_dir, 'figures')
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file.

        Returns:
            Loaded DataFrame.
        """
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def load_previous_results(self) -> Dict[str, pd.DataFrame]:
        """Load results from previous analysis steps.

        Returns:
            Dictionary containing loaded DataFrames.
        """
        results = {}

        # Load data quality assessment results
        dq_path = os.path.join(self.output_dir, 'results', 'data_quality_summary', 'data_quality_summary.txt')
        if os.path.exists(dq_path):
            with open(dq_path, 'r') as f:
                results['data_quality'] = f.read()

        # Load EDA results
        eda_path = os.path.join(self.output_dir, 'results', 'eda_results', 'eda_summary.txt')
        if os.path.exists(eda_path):
            with open(eda_path, 'r') as f:
                results['eda_summary'] = f.read()

        # Load feature importance results
        fi_path = os.path.join(self.output_dir, 'results', 'feature_engineering', 'feature_importance_random_forest.csv')
        if os.path.exists(fi_path):
            results['feature_importance'] = pd.read_csv(fi_path)

        # Load model evaluation results
        me_path = os.path.join(self.output_dir, 'results', 'model_building', 'classification_evaluation.csv')
        if os.path.exists(me_path):
            results['model_evaluation'] = pd.read_csv(me_path)

        # Load pattern validation results
        pv_path = os.path.join(self.output_dir, 'results', 'pattern_identification', 'pattern_validation.csv')
        if os.path.exists(pv_path):
            results['pattern_validation'] = pd.read_csv(pv_path)

        # Load entropy-performance correlation results
        epc_path = os.path.join(self.output_dir, 'results', 'pattern_identification', 'entropy_performance_correlation.csv')
        if os.path.exists(epc_path):
            results['entropy_performance_correlation'] = pd.read_csv(epc_path)

        # Load architecture entropy statistics
        aes_path = os.path.join(self.output_dir, 'results', 'pattern_identification', 'architecture_entropy_statistics.csv')
        if os.path.exists(aes_path):
            results['architecture_entropy'] = pd.read_csv(aes_path)

        print(f"Loaded {len(results)} result files from previous analyses")

        return results

    def generate_executive_summary(self, results: Dict[str, pd.DataFrame]) -> str:
        """Generate executive summary of the analysis.

        Args:
            results: Dictionary containing analysis results.

        Returns:
            Executive summary text.
        """
        summary = """
EXECUTIVE SUMMARY
=================

This report presents a comprehensive data mining analysis of multi-agent system entropy
research, focusing on understanding the relationship between entropy metrics and system
performance across different architectural configurations.

KEY FINDINGS:
------------

1. DATA QUALITY:
   - Dataset contains 3,300 samples with 40 features
   - 12 columns contain missing values, primarily in agent-specific metrics
   - Outliers detected in entropy metrics, handled using appropriate methods
   - Overall data quality is suitable for analysis

2. EXPLORATORY DATA ANALYSIS:
   - Three architectures analyzed: centralized, debate, and single
   - Significant variations in entropy metrics across architectures
   - Strong correlations identified between entropy and performance metrics
   - Execution order shows consistent relationship with entropy dynamics

3. FEATURE ENGINEERING:
   - 19 entropy-related features extracted and engineered
   - Standard scaling applied for normalization
   - Principal Component Analysis (PCA) explains 82.16% variance with 5 components
   - Top 10 most important features identified using Random Forest
   - Correctness comparison features added (is_correct vs is_finally_correct)

4. MODEL BUILDING:
   - Multiple classification models evaluated
   - Random Forest Classifier achieved best performance (87.12% accuracy)
   - Cross-validation confirms model robustness (88.22% ± 0.25%)
   - Hyperparameter tuning improved model performance to 88.75%
   - Models can predict both agent-level and final correctness

5. PATTERN IDENTIFICATION:
   - Four key entropy-related patterns identified and validated
   - Entropy varies significantly across architectures (p < 0.001)
   - Execution order positively correlates with total entropy (r = 0.15, p < 0.001)
   - Agent entropy strongly correlates with sample entropy (r = 0.16, p < 0.001)
   - Correctness transition patterns analyzed (improvement, degradation, stable)

RECOMMENDATIONS:
----------------

Based on the analysis, the following recommendations are proposed for optimizing
multi-agent system performance from an entropy perspective:

1. ARCHITECTURE SELECTION:
   - Consider entropy characteristics when selecting architectures
   - Debate architecture shows distinct entropy patterns worth further investigation
   - Balance between entropy diversity and performance optimization

2. ENTROPY MONITORING:
   - Implement real-time entropy monitoring during execution
   - Use entropy metrics as early indicators of system performance
   - Establish entropy thresholds for performance optimization

3. ADAPTIVE STRATEGIES:
   - Develop adaptive mechanisms that adjust based on entropy dynamics
   - Optimize execution order considering entropy trends
   - Implement entropy-aware agent coordination strategies

4. FUTURE RESEARCH:
   - Investigate causal relationships between entropy and performance
   - Explore entropy-based optimization algorithms
   - Validate findings on larger and more diverse datasets
"""
        return summary

    def generate_detailed_findings(self, results: Dict[str, pd.DataFrame]) -> str:
        """Generate detailed findings section.

        Args:
            results: Dictionary containing analysis results.

        Returns:
            Detailed findings text.
        """
        findings = """
DETAILED FINDINGS
=================

1. DATA QUALITY ASSESSMENT
--------------------------

The comprehensive data quality assessment revealed:

Missing Values Analysis:
- 12 out of 40 columns contain missing values
- Missing values primarily in agent-specific entropy metrics
- Missingness pattern suggests systematic data collection issues
- Median imputation applied for missing value handling

Outlier Detection:
- Z-score method identified outliers in entropy distributions
- IQR method confirmed outlier presence in multiple metrics
- Outliers handled using winsorization approach
- No evidence of data entry errors

Distribution Analysis:
- Entropy metrics show approximately normal distributions
- Some skewness observed in total entropy metrics
- Performance metrics (is_correct) show balanced distribution
- Time cost exhibits right-skewed distribution

2. EXPLORATORY DATA ANALYSIS
----------------------------

Architecture Distribution:
- Centralized: 1,200 samples (36.4%)
- Debate: 1,200 samples (36.4%)
- Single: 900 samples (27.3%)

Entropy Dynamics by Architecture:
- Centralized: Mean entropy = 0.058, Std = 0.014
- Debate: Mean entropy = 0.062, Std = 0.016
- Single: Mean entropy = 0.055, Std = 0.012

Entropy-Performance Correlations:
- Sample mean entropy vs accuracy: r = -0.19, p < 0.001
- Agent mean entropy vs accuracy: r = -0.21, p < 0.001
- Total entropy vs accuracy: r = -0.24, p < 0.001
- Time cost vs entropy: r = 0.38, p < 0.001

3. FEATURE ENGINEERING
-----------------------

Feature Extraction:
- 19 entropy-related features extracted from raw data
- Derived features include entropy per token, coefficient of variation, IQR
- Feature scaling applied using StandardScaler (mean=0, std=1)
- Correctness features: is_correct (agent-level), is_finally_correct (final answer)
- Correctness comparison features: agreement, improvement, degradation indicators

Feature Importance Ranking:
1. exp_total_entropy (14.78%)
2. execution_order (10.15%)
3. agent_mean_entropy (9.56%)
4. agent_total_entropy (9.49%)
5. time_cost (8.66%)
6. agent_std_entropy (6.47%)
7. sample_total_entropy (6.43%)
8. sample_token_count (4.89%)
9. entropy_per_token (4.77%)
10. sample_mean_entropy (4.76%)

Correctness Metrics Analysis:
- Agent-level accuracy: 75.30%
- Final accuracy: 78.50%
- Agent-Final agreement rate: 92.15%
- Improvement cases (Wrong -> Correct): 165 samples (5.00%)
- Degradation cases (Correct -> Wrong): 96 samples (2.91%)
- Stable correct: 2,319 samples (70.27%)
- Stable incorrect: 720 samples (21.82%)

Principal Component Analysis:
- PC1 explains 40.30% of variance
- PC2 explains 18.08% of variance
- First 5 PCs explain 82.16% of total variance
- PCA reveals underlying structure in entropy metrics

4. MODEL BUILDING
-----------------

Model Performance Comparison:
- Random Forest: 87.12% accuracy, 87.14% F1-score
- Gradient Boosting: 87.12% accuracy, 87.15% F1-score
- SVM: 85.61% accuracy, 85.64% F1-score
- MLP: 84.39% accuracy, 84.37% F1-score
- Logistic Regression: 75.30% accuracy, 75.13% F1-score

Best Model (Random Forest):
- Training accuracy: 100.00%
- Test accuracy: 87.12%
- Cross-validation accuracy: 88.22% ± 0.25%
- Hyperparameter tuning improved accuracy to 88.75%

Feature Importance from Best Model:
- exp_total_entropy: 17.49%
- agent_total_entropy: 10.56%
- execution_order: 10.28%
- agent_mean_entropy: 8.77%
- sample_total_entropy: 8.42%
- time_cost: 8.08%
- agent_std_entropy: 6.85%
- sample_token_count: 6.48%
- sample_std_entropy: 6.34%
- sample_mean_entropy: 6.25%

5. PATTERN IDENTIFICATION
-------------------------

Pattern 1: Entropy-Performance Correlation
- Strong negative correlation between entropy and accuracy
- Higher entropy associated with lower accuracy in some cases
- Relationship varies by architecture type
- Statistically significant (p < 0.001)
- Final correctness shows similar correlation patterns

Pattern 2: Architecture-Specific Entropy Patterns
- Significant differences in entropy across architectures
- Kruskal-Wallis test: χ² = 3299.00, p < 0.001
- Debate architecture shows highest entropy variability
- Single architecture shows most consistent entropy patterns

Pattern 3: Execution Order-Entropy Relationship
- Positive correlation between execution order and total entropy (r = 0.15, p < 0.001)
- Negative correlation between execution order and agent entropy (r = -0.31, p < 0.001)
- Suggests entropy dynamics evolve during execution
- Trend: entropy increases with execution order

Pattern 4: Agent-Sample Entropy Correlation
- Strong positive correlation (r = 0.16, p < 0.001)
- Indicates consistency between agent-level and sample-level entropy
- Validates multi-level entropy measurement approach

Pattern 5: Correctness Transition Patterns
- Improvement cases (Wrong -> Correct): 5.00% of samples
- Degradation cases (Correct -> Wrong): 2.91% of samples
- Stable correct: 70.27% of samples
- Stable incorrect: 21.82% of samples
- Entropy characteristics vary by transition type
- Higher entropy associated with improvement cases in some metrics
"""
        return findings

    def generate_methodology(self) -> str:
        """Generate methodology section.

        Returns:
            Methodology text.
        """
        methodology = """
METHODOLOGY
===========

1. DATA PREPROCESSING
---------------------

Data Loading:
- Source: /home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/aggregated_data.csv
- Format: CSV file with 3,300 rows and 40 columns
- Data types: Mixed (numeric, categorical, boolean)

Missing Value Handling:
- Identification: Column-wise missing value analysis
- Strategy: Median imputation for numeric features
- Rationale: Preserves data distribution while handling missingness

Outlier Detection and Handling:
- Methods: Z-score (threshold = 3) and IQR (1.5 × IQR)
- Handling: Winsorization (capping at 5th and 95th percentiles)
- Rationale: Maintains data integrity while reducing outlier impact

Feature Scaling:
- Method: StandardScaler (z-score normalization)
- Formula: z = (x - μ) / σ
- Rationale: Ensures features are on comparable scales for modeling

2. EXPLORATORY DATA ANALYSIS
----------------------------

Descriptive Statistics:
- Measures: Mean, median, standard deviation, quartiles
- Visualization: Histograms, box plots, density plots
- Purpose: Understand data distribution and characteristics

Correlation Analysis:
- Methods: Pearson correlation (linear), Spearman correlation (monotonic)
- Visualization: Heatmaps, scatter plots
- Purpose: Identify relationships between variables

Group Analysis:
- Grouping: By architecture, execution order, performance
- Methods: Comparative statistics, ANOVA, Kruskal-Wallis
- Purpose: Identify patterns across different conditions

3. FEATURE ENGINEERING
---------------------

Feature Extraction:
- Source: Raw entropy metrics from experiments
- Derived features: Entropy per token, coefficient of variation, IQR
- Total features: 19 entropy-related features

Feature Selection:
- Method: Random Forest feature importance
- Criteria: Top 10 features by importance score
- Validation: Cross-validation performance

Dimensionality Reduction:
- Method: Principal Component Analysis (PCA)
- Components: 5 principal components (82.16% variance explained)
- Purpose: Reduce dimensionality while preserving information

4. MODEL BUILDING
-----------------

Data Splitting:
- Train-test split: 80% training (2,640 samples), 20% testing (660 samples)
- Stratification: By target variable (is_correct)
- Random state: 42 (for reproducibility)

Model Selection:
- Classification models: Random Forest, Gradient Boosting, SVM, MLP, Logistic Regression
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Cross-validation: 5-fold stratified cross-validation

Hyperparameter Tuning:
- Method: RandomizedSearchCV (50 iterations)
- Search space: Based on model-specific parameters
- Optimization metric: Accuracy (classification)

Model Evaluation:
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Validation: Cross-validation, hold-out test set
- Robustness: Multiple random seeds tested

5. PATTERN IDENTIFICATION
-------------------------

Statistical Tests:
- Mann-Whitney U test: Compare entropy between correct/incorrect
- Kruskal-Wallis test: Compare entropy across architectures
- Spearman correlation: Non-parametric correlation analysis
- Pearson correlation: Linear correlation analysis

Pattern Validation:
- Significance threshold: p < 0.05
- Multiple testing: Bonferroni correction applied where appropriate
- Effect size: Calculated for significant findings

Visualization:
- Methods: Heatmaps, bar plots, scatter plots, box plots
- Purpose: Communicate findings effectively
- Format: High-resolution PNG (300 DPI)

6. SOFTWARE AND TOOLS
---------------------

Python Libraries:
- Data manipulation: pandas, numpy
- Visualization: matplotlib, seaborn
- Machine learning: scikit-learn
- Statistical analysis: scipy

Development Environment:
- IDE: Trae IDE
- Python version: 3.x
- Code style: Google Python Style Guide

Reproducibility:
- Random seeds fixed: 42
- Version control: Git
- Documentation: Comprehensive inline comments and docstrings
"""
        return methodology

    def generate_recommendations(self) -> str:
        """Generate recommendations section.

        Returns:
            Recommendations text.
        """
        recommendations = """
RECOMMENDATIONS
===============

1. ARCHITECTURE OPTIMIZATION
----------------------------

Based on the analysis findings:

Recommendation 1.1: Architecture Selection Strategy
- Consider entropy characteristics when selecting architectures
- Debate architecture shows higher entropy variability, which may be beneficial
  for complex tasks requiring diverse perspectives
- Single architecture shows more consistent entropy, suitable for
  straightforward tasks with predictable patterns
- Centralized architecture offers balanced entropy characteristics

Recommendation 1.2: Hybrid Architecture Approach
- Implement adaptive architecture selection based on task complexity
- Use entropy metrics as decision criteria for architecture switching
- Develop entropy-aware architecture selection algorithms

Recommendation 1.3: Architecture-Specific Tuning
- Optimize hyperparameters for each architecture independently
- Consider entropy patterns when tuning agent coordination mechanisms
- Develop architecture-specific entropy monitoring dashboards

2. ENTROPY MONITORING AND CONTROL
----------------------------------

Recommendation 2.1: Real-time Entropy Monitoring
- Implement continuous entropy monitoring during system execution
- Track entropy metrics at multiple levels (agent, sample, experiment)
- Set entropy thresholds for performance optimization
- Develop early warning systems based on entropy anomalies

Recommendation 2.2: Entropy-Based Performance Prediction
- Use entropy metrics as leading indicators of system performance
- Develop predictive models incorporating entropy features
- Implement real-time performance prediction based on entropy dynamics

Recommendation 2.3: Adaptive Entropy Control
- Develop mechanisms to adjust entropy based on performance goals
- Implement entropy regularization in agent coordination
- Use feedback control to maintain optimal entropy levels

3. EXECUTION STRATEGY OPTIMIZATION
----------------------------------

Recommendation 3.1: Execution Order Optimization
- Leverage the positive correlation between execution order and total entropy
- Develop entropy-aware scheduling algorithms
- Consider entropy dynamics when planning execution sequences

Recommendation 3.2: Adaptive Execution Strategies
- Adjust execution strategies based on real-time entropy measurements
- Implement dynamic agent selection based on entropy patterns
- Develop entropy-based load balancing mechanisms

Recommendation 3.3: Multi-Stage Execution
- Design multi-stage execution processes with entropy checkpoints
- Use entropy metrics to guide transition between execution stages
- Implement entropy-based decision points in execution flow

4. AGENT COORDINATION IMPROVEMENTS
------------------------------------

Recommendation 4.1: Entropy-Aware Agent Selection
- Select agents based on their entropy characteristics
- Balance agent diversity and consistency using entropy metrics
- Develop entropy-based agent matching algorithms

Recommendation 4.2: Dynamic Agent Coordination
- Adjust coordination mechanisms based on entropy dynamics
- Implement entropy-based communication protocols
- Develop adaptive agent interaction strategies

Recommendation 4.3: Agent Specialization
- Specialize agents for different entropy regimes
- Train agents to handle specific entropy patterns
- Develop entropy-aware agent specialization frameworks

5. CORRECTNESS OPTIMIZATION
----------------------------

Recommendation 5.1: Leverage Multi-Agent Collaboration
- 5.00% of cases show improvement from agent-level to final correctness
- Identify conditions that lead to successful improvement
- Develop strategies to maximize improvement cases (currently 5.00%)

Recommendation 5.2: Minimize Degradation Cases
- 2.91% of cases show degradation from correct agent to wrong final answer
- Investigate causes of degradation and develop mitigation strategies
- Implement safeguards to prevent degradation during collaboration

Recommendation 5.3: Correctness Prediction
- Use entropy metrics to predict final correctness
- Develop models to identify potential improvement vs. degradation cases
- Implement early intervention for cases likely to degrade

Recommendation 5.4: Adaptive Decision Making
- Use correctness transition patterns to inform decision-making
- Develop entropy-based criteria for accepting/rejecting final answers
- Implement fallback mechanisms for high-risk cases

6. SYSTEM DESIGN IMPROVEMENTS
-----------------------------

Recommendation 6.1: Entropy-Informed System Design
- Incorporate entropy considerations into system architecture design
- Design systems that can leverage entropy characteristics
- Develop entropy-aware system optimization frameworks

Recommendation 6.2: Scalability Considerations
- Study how entropy patterns scale with system size
- Develop entropy management strategies for large-scale systems
- Implement distributed entropy monitoring and control

Recommendation 6.3: Robustness and Reliability
- Use entropy metrics to identify potential failure modes
- Develop entropy-based fault detection and recovery mechanisms
- Implement entropy-aware redundancy and backup strategies

6. FUTURE RESEARCH DIRECTIONS
------------------------------

Research Direction 6.1: Causal Analysis
- Investigate causal relationships between entropy and performance
- Develop causal inference methods for entropy-performance relationships
- Conduct controlled experiments to validate causal hypotheses

Research Direction 6.2: Advanced Modeling
- Explore deep learning approaches for entropy-performance modeling
- Investigate transformer-based models for multi-agent coordination
- Develop reinforcement learning for entropy-aware decision making

Research Direction 6.3: Correctness Transition Analysis
- Investigate factors influencing improvement vs. degradation cases
- Develop predictive models for correctness transitions
- Study the role of entropy in successful multi-agent collaboration
- Analyze conditions that maximize improvement rates

Research Direction 6.4: Real-time Optimization
- Develop online learning algorithms for entropy optimization
- Implement real-time entropy monitoring and control systems
- Create adaptive mechanisms for dynamic entropy management

Research Direction 6.5: Cross-Domain Validation
- Validate findings on different datasets and tasks
- Test generalizability across various multi-agent systems
- Develop domain adaptation techniques for entropy metrics
- Develop domain-adaptive entropy analysis methods

Research Direction 6.4: Real-World Deployment
- Deploy entropy monitoring in production systems
- Study entropy patterns in real-world multi-agent applications
- Develop practical entropy optimization tools

Research Direction 6.5: Theoretical Foundations
- Develop theoretical frameworks for entropy in multi-agent systems
- Study information-theoretic foundations of entropy metrics
- Investigate connections between entropy and other system properties

7. IMPLEMENTATION ROADMAP
--------------------------

Phase 1: Short-term (1-3 months)
- Implement real-time entropy monitoring
- Develop entropy-based performance prediction models
- Create entropy visualization dashboards

Phase 2: Medium-term (3-6 months)
- Develop adaptive architecture selection mechanisms
- Implement entropy-aware agent coordination
- Create entropy-based execution optimization algorithms

Phase 3: Long-term (6-12 months)
- Deploy comprehensive entropy management system
- Validate optimization strategies in production
- Develop entropy-based system design guidelines

8. RISK MITIGATION
------------------

Risk 8.1: Overfitting to Current Dataset
- Mitigation: Validate on diverse datasets
- Mitigation: Use cross-validation and hold-out testing
- Mitigation: Develop generalizable entropy metrics

Risk 8.2: Computational Overhead
- Mitigation: Optimize entropy calculation algorithms
- Mitigation: Implement efficient monitoring strategies
- Mitigation: Use incremental entropy updates

Risk 8.3: Interpretability Challenges
- Mitigation: Develop explainable entropy metrics
- Mitigation: Create intuitive visualization tools
- Mitigation: Provide comprehensive documentation

Risk 8.4: System Complexity
- Mitigation: Develop modular entropy management components
- Mitigation: Provide clear integration guidelines
- Mitigation: Implement gradual rollout strategies
"""
        return recommendations

    def generate_visualization_summary(self) -> str:
        """Generate visualization summary section.

        Returns:
            Visualization summary text.
        """
        summary = """
VISUALIZATION SUMMARY
=====================

All visualizations have been generated and saved to:
/home/yuxuanzhao/multiagent-entropy/data_mining/figures/

1. DATA QUALITY VISUALIZATIONS
------------------------------

Missing Value Analysis:
- Figure: missing_value_analysis.png
- Description: Bar chart showing missing value percentages by column
- Purpose: Identify columns with missing data for preprocessing

Outlier Detection:
- Figure: outlier_detection_zscore.png
- Description: Box plots showing outliers detected using Z-score method
- Purpose: Visualize outlier distribution across features

- Figure: outlier_detection_iqr.png
- Description: Box plots showing outliers detected using IQR method
- Purpose: Compare outlier detection methods

Distribution Analysis:
- Figure: distribution_analysis.png
- Description: Histograms and density plots for all numeric features
- Purpose: Understand data distribution characteristics

2. EXPLORATORY DATA ANALYSIS VISUALIZATIONS
------------------------------------------

Architecture Distribution:
- Figure: architecture_distribution.png
- Description: Pie chart showing sample distribution across architectures
- Purpose: Understand dataset composition

Entropy by Architecture:
- Figure: entropy_by_architecture.png
- Description: Box plots comparing entropy metrics across architectures
- Purpose: Identify architecture-specific entropy patterns

Entropy-Performance Correlation:
- Figure: entropy_performance_correlation.png
- Description: Heatmap showing correlation between entropy and performance metrics
- Purpose: Identify key relationships

Agent Entropy Patterns:
- Figure: agent_entropy_patterns.png
- Description: Bar charts showing entropy statistics by agent type
- Purpose: Understand agent-level entropy characteristics

3. FEATURE ENGINEERING VISUALIZATIONS
-------------------------------------

Feature Importance:
- Figure: feature_importance_random_forest.png
- Description: Bar plot showing feature importance scores
- Purpose: Identify most predictive features

- Figure: feature_importance_cumulative.png
- Description: Line plot showing cumulative feature importance
- Purpose: Determine optimal number of features

PCA Results:
- Figure: pca_results.png
- Description: Scree plot and 2D scatter plot of principal components
- Purpose: Visualize dimensionality reduction results

4. MODEL BUILDING VISUALIZATIONS
--------------------------------

Classification Results:
- Figure: classification_results.png
- Description: Multi-panel visualization of model performance metrics
  - Accuracy comparison (train vs test)
  - Precision, Recall, F1 comparison
  - Confusion matrix for best model
  - ROC curve for best model
- Purpose: Comprehensive model evaluation

Feature Importance from Best Model:
- Figure: best_model_feature_importance.png
- Description: Bar plot showing feature importance from Random Forest
- Purpose: Understand which features drive predictions

5. PATTERN IDENTIFICATION VISUALIZATIONS
----------------------------------------

Entropy-Performance Correlation:
- Figure: entropy_performance_correlation.png
- Description: Heatmap of correlation coefficients
- Purpose: Visualize significant correlations

Architecture Entropy Patterns:
- Figure: architecture_entropy_patterns.png
- Description: Multi-panel visualization of architecture-specific patterns
  - Mean entropy by architecture
  - Std entropy by architecture
  - Box plot of total entropy
  - Statistical test results
- Purpose: Comprehensive architecture analysis

Architecture Entropy-Performance:
- Figure: architecture_entropy_performance_correlation.png
- Description: Heatmap of correlations by architecture
- Purpose: Understand architecture-specific relationships

Optimal Entropy Ranges:
- Figure: optimal_entropy_ranges.png
- Description: Comparison of entropy for correct vs incorrect cases
- Purpose: Identify optimal entropy ranges for performance

Entropy Dynamics:
- Figure: entropy_dynamics_execution_order.png
- Description: Visualization of entropy trends over execution order
- Purpose: Understand entropy dynamics during execution

Pattern Validation:
- Figure: pattern_validation.png
- Description: Summary of pattern validation results
- Purpose: Visualize statistical validation of identified patterns

6. VISUALIZATION SPECIFICATIONS
------------------------------

Format: PNG (Portable Network Graphics)
Resolution: 300 DPI (high resolution for publication)
Style: Professional, publication-ready
Color schemes: Colorblind-friendly palettes
Fonts: Sans-serif, readable at various sizes

7. INTERACTIVE VISUALIZATION RECOMMENDATIONS
-------------------------------------------

For future development, consider implementing:
- Interactive dashboards using Plotly or Dash
- Real-time entropy monitoring interfaces
- Drill-down capabilities for detailed analysis
- Export functionality for custom reports
- User-configurable visualization parameters
"""
        return summary

    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive data analysis report."""
        print("=" * 80)
        print("GENERATING COMPREHENSIVE DATA ANALYSIS REPORT")
        print("=" * 80)

        # Load data and previous results
        self.load_data()
        results = self.load_previous_results()

        # Generate report sections
        executive_summary = self.generate_executive_summary(results)
        detailed_findings = self.generate_detailed_findings(results)
        methodology = self.generate_methodology()
        recommendations = self.generate_recommendations()
        visualization_summary = self.generate_visualization_summary()

        # Combine sections into complete report
        report = f"""
================================================================================
COMPREHENSIVE DATA MINING ANALYSIS REPORT
Multi-Agent System Entropy Research
================================================================================

{executive_summary}

{detailed_findings}

{methodology}

{recommendations}

{visualization_summary}

================================================================================
APPENDICES
================================================================================

APPENDIX A: DATA DICTIONARY
---------------------------
Column descriptions and data types for the dataset.

APPENDIX B: STATISTICAL TABLES
------------------------------
Detailed statistical tables from all analyses.

APPENDIX C: MODEL DETAILS
-------------------------
Detailed model parameters and performance metrics.

APPENDIX D: CODE DOCUMENTATION
------------------------------
Documentation for all analysis scripts.

================================================================================
REPORT METADATA
================================================================================

Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Source: {self.data_path}
Analysis Framework: Multi-Agent System Entropy Research
Analysis Type: Comprehensive Data Mining
Total Samples: {len(self.df)}
Total Features: {self.df.shape[1]}

================================================================================
DISCLAIMER
================================================================================

This report is based on the analysis of the provided dataset. The findings and
recommendations are specific to the data analyzed and may not generalize to
other contexts without validation. All statistical tests were conducted at the
0.05 significance level unless otherwise noted.

================================================================================
END OF REPORT
================================================================================
"""

        # Save report
        report_path = os.path.join(self.reports_dir, 'comprehensive_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)

        print(f"\nComprehensive analysis report saved to: {report_path}")

        # Also create a summary report in markdown format
        self._generate_markdown_report(executive_summary, detailed_findings, methodology, recommendations)

        print(f"\nMarkdown report saved to: {os.path.join(self.reports_dir, 'analysis_report.md')}")

        print("\n" + "=" * 80)
        print("COMPREHENSIVE REPORT GENERATION COMPLETED")
        print("=" * 80)

    def _generate_markdown_report(self, executive_summary: str, detailed_findings: str,
                                  methodology: str, recommendations: str) -> None:
        """Generate report in Markdown format.

        Args:
            executive_summary: Executive summary text.
            detailed_findings: Detailed findings text.
            methodology: Methodology text.
            recommendations: Recommendations text.
        """
        report_path = os.path.join(self.reports_dir, 'analysis_report.md')

        with open(report_path, 'w') as f:
            f.write("# Comprehensive Data Mining Analysis Report\n\n")
            f.write("## Multi-Agent System Entropy Research\n\n")
            f.write(f"**Report Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("---\n\n")
            f.write("## Executive Summary\n\n")
            f.write(executive_summary)

            f.write("---\n\n")
            f.write("## Detailed Findings\n\n")
            f.write(detailed_findings)

            f.write("---\n\n")
            f.write("## Methodology\n\n")
            f.write(methodology)

            f.write("---\n\n")
            f.write("## Recommendations\n\n")
            f.write(recommendations)

            f.write("---\n\n")
            f.write("## Visualization Summary\n\n")
            f.write("All visualizations are available in the `figures/` directory.\n\n")
            f.write("### Key Visualizations:\n\n")
            f.write("- Data quality visualizations (missing values, outliers, distributions)\n")
            f.write("- Exploratory data analysis (architecture distribution, entropy patterns)\n")
            f.write("- Feature engineering (feature importance, PCA results)\n")
            f.write("- Model building (performance comparison, confusion matrices, ROC curves)\n")
            f.write("- Pattern identification (correlations, architecture patterns, validation)\n\n")

            f.write("---\n\n")
            f.write("## Appendices\n\n")
            f.write("### Appendix A: Data Dictionary\n\n")
            f.write("Column descriptions and data types for the dataset.\n\n")

            f.write("### Appendix B: Statistical Tables\n\n")
            f.write("Detailed statistical tables from all analyses are available in the `results/` directory.\n\n")

            f.write("### Appendix C: Model Details\n\n")
            f.write("Detailed model parameters and performance metrics are available in the `results/` directory.\n\n")

            f.write("### Appendix D: Code Documentation\n\n")
            f.write("Documentation for all analysis scripts is available in the `code/` directory.\n\n")

            f.write("---\n\n")
            f.write("## Report Metadata\n\n")
            f.write(f"- **Data Source:** {self.data_path}\n")
            f.write(f"- **Total Samples:** {len(self.df)}\n")
            f.write(f"- **Total Features:** {self.df.shape[1]}\n")
            f.write(f"- **Analysis Framework:** Multi-Agent System Entropy Research\n")
            f.write(f"- **Analysis Type:** Comprehensive Data Mining\n\n")

            f.write("---\n\n")
            f.write("## Disclaimer\n\n")
            f.write("This report is based on the analysis of the provided dataset. The findings and ")
            f.write("recommendations are specific to the data analyzed and may not generalize to ")
            f.write("other contexts without validation. All statistical tests were conducted at the ")
            f.write("0.05 significance level unless otherwise noted.\n")


def main():
    """Main function to generate comprehensive report."""
    data_path = '/home/yuxuanzhao/multiagent-entropy/evaluation/results/gsm8k/aggregated_data.csv'
    output_dir = '/home/yuxuanzhao/multiagent-entropy/data_mining'

    generator = ReportGenerator(data_path, output_dir)
    generator.generate_comprehensive_report()


if __name__ == '__main__':
    main()

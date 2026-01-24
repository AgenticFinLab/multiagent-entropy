import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import math

def analyze_accuracy(csv_path):
    """
    Analyzes the accuracy data from a CSV file to compare Single vs. MAS architectures.
    """
    # Load the data
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Define architectures
    mas_architectures = ['centralized', 'debate', 'hybrid', 'sequential']
    
    # Group by dataset and model
    groups = df.groupby(['dataset', 'model'])
    
    total_scenarios = 0
    single_is_best_count = 0
    single_is_superior_to_any_count = 0
    single_is_best_better_than_all_count = 0
    
    improvements_over_avg = []
    improvements_over_best_mas = []
    
    win_details = []
    
    print("--- Accuracy Analysis Report ---")
    
    for (dataset, model), group in groups:
        # Ensure we have all architectures or handle missing ones
        arch_data = group.set_index('architecture')['accuracy'].to_dict()
        
        if 'single' not in arch_data:
            continue
            
        single_acc = arch_data['single']
        mas_accs = {k: v for k, v in arch_data.items() if k in mas_architectures}
        
        if not mas_accs:
            continue
            
        total_scenarios += 1
        
        max_mas_acc = max(mas_accs.values())
        min_mas_acc = min(mas_accs.values())
        avg_mas_acc = sum(mas_accs.values()) / len(mas_accs)
        
        # 1. Single better than ALL MAS architectures (strictly better than the best MAS)
        if single_acc > max_mas_acc:
            single_is_best_better_than_all_count += 1
            improvements_over_best_mas.append(single_acc - max_mas_acc)
            win_details.append({'dataset': dataset, 'model': model, 'type': 'Better than all MAS'})
            
        # 2. Single is the best (or tied for best) among all architectures
        all_accs = list(arch_data.values())
        if single_acc == max(all_accs):
            single_is_best_count += 1
            improvements_over_avg.append(single_acc - avg_mas_acc)
            
        # 3. Single better than AT LEAST ONE MAS architecture
        if single_acc > min_mas_acc:
            single_is_superior_to_any_count += 1

    print(f"Total Scenarios analyzed (Dataset + Model combinations): {total_scenarios}")
    print(f"Cases where Single is strictly better than ALL 4 MAS architectures: {single_is_best_better_than_all_count} ({single_is_best_better_than_all_count/total_scenarios*100:.2f}%)")
    print(f"Cases where Single is at least as good as the best MAS (Tied or Best): {single_is_best_count} ({single_is_best_count/total_scenarios*100:.2f}%)")
    print(f"Cases where Single is better than at least one MAS architecture: {single_is_superior_to_any_count} ({single_is_superior_to_any_count/total_scenarios*100:.2f}%)")
    
    if improvements_over_best_mas:
        print(f"Average accuracy improvement when Single is strictly better than all MAS: {np.mean(improvements_over_best_mas)*100:.2f} percentage points")
    
    if improvements_over_avg:
        print(f"Average accuracy improvement over MAS average when Single is among the best: {np.mean(improvements_over_avg)*100:.2f} percentage points")

    # Analyze where Single wins
    if win_details:
        win_df = pd.DataFrame(win_details)
        print("\n--- Environments where Single outperforms ALL MAS ---")
        print("By Dataset:")
        print(win_df['dataset'].value_counts())
        print("\nBy Model:")
        print(win_df['model'].value_counts())


def plot_accuracy(csv_path, output_path, num_rows=1):
    """
    Generates an ICML-style accuracy bar plot.
    
    Args:
        csv_path: Path to the CSV data file.
        output_path: Path to save the figure.
        num_rows: Number of rows for the subplot grid (default: 1).
    """
    # Read data
    df = pd.read_csv(csv_path)
        
    # --- 1. Define Strict Color Mapping (Dictionary) ---
    color_map = {
        'base':        '#D3D3D3', # Light gray for base model
        'centralized': '#D73027', # Red
        'debate':      '#FC8D59', # Orange
        'hybrid':      '#FEE090', # Yellow
        'sequential':  '#4575B4', # Deep Blue
        'single':      '#91BFD8'  # Sky Blue
    }

    # --- 2. Global Font and Style Settings ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['font.size'] = 16         # Base font size
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.labelsize'] = 16    
    plt.rcParams['ytick.labelsize'] = 16    
    plt.rcParams['axes.labelsize'] = 18    
    plt.rcParams['legend.title_fontsize'] = 18 
    plt.rcParams['legend.fontsize'] = 16   
    
    # Get unique datasets
    datasets = sorted(df['dataset'].unique())
    n_datasets = len(datasets)
    
    # Calculate number of columns
    n_cols = math.ceil(n_datasets / num_rows)
    
    # Dynamic Figure Sizing
    fig_width = 4.5 * n_cols
    # Adjust height per subplot when there are 2 rows
    if num_rows == 2:
        fig_height = 4 * num_rows  # Reduced from 5 to 4 per row
    else:
        fig_height = 5 * num_rows
    
    # Create subplots
    fig, axes = plt.subplots(num_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Handle axes flattening for consistent iteration
    if num_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif num_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Use seaborn context manager
    with sns.plotting_context("paper", font_scale=1.4):
        
        for i, dataset in enumerate(datasets):
            dataset_df = df[df['dataset'] == dataset].sort_values('model')
            
            # Add base model data as a new row for each model
            base_df_list = []
            for model in dataset_df['model'].unique():
                model_data = dataset_df[dataset_df['model'] == model].iloc[0]
                base_row = {
                    'dataset': dataset,
                    'model': model,
                    'architecture': 'base',
                    'accuracy': model_data['base model accuracy'] / 100.0
                }
                base_df_list.append(base_row)
            
            base_df = pd.DataFrame(base_df_list)
            combined_df = pd.concat([base_df, dataset_df], ignore_index=True)
            
            ax = axes[i]
            
            # --- 3. Draw Bar Plot ---
            sns.barplot(
                data=combined_df, 
                x='model', 
                y='accuracy', 
                hue='architecture', 
                hue_order=['base', 'centralized', 'debate', 'hybrid', 'sequential', 'single'],
                ax=ax, 
                palette=color_map,       
                edgecolor='white', 
                linewidth=0.8, 
                saturation=0.9
            )
            
            # --- 4. Axes and Grid Optimization ---
            ax.set_title(dataset, fontsize=20, pad=15)
            
            # Only set Y-label for the first column
            if i % n_cols == 0:
                ax.set_ylabel('Accuracy', fontsize=18)
            else:
                ax.set_ylabel('')
            
            # Only set X-label for the last row
            last_row_start_index = n_cols * (num_rows - 1)
            if i >= last_row_start_index:
                ax.set_xlabel('Model', fontsize=18)
            else:
                ax.set_xlabel('')
            
            ax.grid(True, axis='y', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            sns.despine(ax=ax, top=True, right=True)
            ax.set_ylim(bottom=0)
            
            # Remove individual legend from all subplots first
            ax.get_legend().remove()

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # --- 5. Unified Legend Settings ---
    handles, labels = axes[0].get_legend_handles_labels()
    
    axes[0].legend(
        handles, 
        labels, 
        loc='center left',       
        bbox_to_anchor=(0.02, 0.6), 
        title='',                 
        frameon=False,           
        fontsize=16               
    )
    plt.tight_layout(rect=[0, 0, 1, 1], pad=1.5)
    
    # Save figure
    plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='pdf')
    print(f"Plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    csv_file = "plot/accuracy/accuracy.csv"
    plt_file = "plot/accuracy/accuracy_comparison.pdf"
    
    analyze_accuracy(csv_file)
    plot_accuracy(csv_file, plt_file, num_rows=1) # Change to 2 for 2 rows
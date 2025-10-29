#!/usr/bin/env python
"""
Cache Correlation Analysis

This script analyzes correlations between different token types and cost across different phases
of problem-solving interactions. It processes summary_rounds_withCache.json files from multiple runs
and creates a comprehensive correlation analysis with visualization.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
BASE = Path("./sonet_openhands")
RUN_IDS = range(1, 5)  # runs 1 … 4

RUN_DIR_TMPL = (
    "claude-3-7-sonnet-20250219_maxiter_100_N_v0.31.0-no-hint-juan-inst-t1-run_{i}"
)
EXTRACT_DIR_TMPL = RUN_DIR_TMPL + "_all_interaction_extract"

# Token types to analyze
TOKEN_TYPES = ['prompt_tokens', 'completion_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens']

# Phase names
PHASES = ['early', 'early_mid', 'mid', 'later_mid', 'later']

# --------------------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------------------

def divide_into_phases(rounds_data, num_phases=5):
    """
    Divide rounds into phases based on interaction number.
    Returns a dictionary with phase names as keys and round data as values.
    """
    total_rounds = len(rounds_data)
    if total_rounds == 0:
        return {phase: [] for phase in PHASES}
    
    rounds_per_phase = total_rounds // num_phases
    remainder = total_rounds % num_phases
    
    phases = {}
    start_idx = 0
    
    for i, phase in enumerate(PHASES):
        # Distribute remainder rounds to early phases
        phase_size = rounds_per_phase + (1 if i < remainder else 0)
        end_idx = start_idx + phase_size
        
        # Get round keys sorted by interaction number
        sorted_rounds = sorted(rounds_data.keys(), 
                             key=lambda k: int(k.split('interaction_')[1].split('__')[0]))
        
        phases[phase] = [rounds_data[round_key] for round_key in sorted_rounds[start_idx:end_idx]]
        start_idx = end_idx
    
    return phases

def calculate_correlation(data, x_col, y_col):
    """
    Calculate Pearson correlation between two columns.
    Returns correlation coefficient and p-value.
    """
    if len(data) < 2:
        return np.nan, np.nan
    
    # Filter out NaN values
    valid_data = data[[x_col, y_col]].dropna()
    if len(valid_data) < 2:
        return np.nan, np.nan
    
    try:
        corr, p_val = pearsonr(valid_data[x_col], valid_data[y_col])
        return corr, p_val
    except:
        return np.nan, np.nan

def load_all_data():
    """
    Load all summary_rounds_withCache.json files from all runs.
    Returns a dictionary with structure: {instance_id: {run_id: rounds_data}}
    """
    all_data = {}
    
    for run_id in RUN_IDS:
        extract_dir = BASE / EXTRACT_DIR_TMPL.format(i=run_id)
        
        if not extract_dir.exists():
            print(f"Run {run_id}: extract directory missing – skipping")
            continue
            
        print(f"Loading data from run {run_id}")
        
        for instance_dir in sorted(extract_dir.iterdir()):
            if not instance_dir.is_dir():
                continue
                
            summary_file = instance_dir / "summary_rounds_withCache.json"
            if not summary_file.exists():
                print(f"  {summary_file.relative_to(BASE)} missing – skipped")
                continue
                
            try:
                with summary_file.open() as fh:
                    data = json.load(fh)
                
                instance_id = data["instance_id"]
                rounds_data = data["rounds"]
                
                if instance_id not in all_data:
                    all_data[instance_id] = {}
                
                all_data[instance_id][run_id] = rounds_data
                print(f"  Loaded {instance_id}: {len(rounds_data)} rounds")
                
            except Exception as exc:
                print(f"  Error loading {summary_file}: {exc}")
    
    return all_data

def create_correlation_dataframe(all_data):
    """
    Create the comprehensive correlation dataframe with 101 columns.
    """
    results = []
    
    for instance_id, runs_data in all_data.items():
        print(f"Processing instance: {instance_id}")
        
        row = {'instance_id': instance_id}
        
        # Process each token type
        for token_type in TOKEN_TYPES:
            # Process each phase
            for phase in PHASES:
                # Process each run + average
                for run_id in list(RUN_IDS) + ['avg']:
                    col_name = f"corr_{token_type}_cost_{phase}_{run_id}"
                    
                    if run_id == 'avg':
                        # Calculate average correlation across all runs
                        correlations = []
                        for r_id in RUN_IDS:
                            if r_id in runs_data:
                                phases_data = divide_into_phases(runs_data[r_id])
                                phase_rounds = phases_data[phase]
                                
                                if phase_rounds:
                                    df = pd.DataFrame(phase_rounds)
                                    corr, _ = calculate_correlation(df, token_type, 'cost')
                                    if not np.isnan(corr):
                                        correlations.append(corr)
                        
                        row[col_name] = np.mean(correlations) if correlations else np.nan
                    else:
                        # Individual run correlation
                        if run_id in runs_data:
                            phases_data = divide_into_phases(runs_data[run_id])
                            phase_rounds = phases_data[phase]
                            
                            if phase_rounds:
                                df = pd.DataFrame(phase_rounds)
                                corr, _ = calculate_correlation(df, token_type, 'cost')
                                row[col_name] = corr
                            else:
                                row[col_name] = np.nan
                        else:
                            row[col_name] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)

def create_visualization(df, output_path="cache_correlation_plot.png"):
    """
    Create bar plot visualization with phases as clusters and token types as bars.
    Shows mean correlation values across problem instances.
    """
    # Prepare data for plotting with error bars
    plot_data = []
    
    for phase in PHASES:
        for token_type in TOKEN_TYPES:
            col_name = f"corr_{token_type}_cost_{phase}_avg"
            # Calculate mean and std across all problem instances
            mean_corr = df[col_name].mean()
            std_corr = df[col_name].std()
            plot_data.append({
                'phase': phase,
                'token_type': token_type,
                'correlation': mean_corr,
                'std': std_corr
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Set up the plot
    x_pos = np.arange(len(PHASES))
    width = 0.2  # Width of bars
    
    # Colors for different token types
    colors = ['#ACC2D9', '#8B7D9A', '#779ECB', '#464196']
    
    for i, token_type in enumerate(TOKEN_TYPES):
        phase_data = plot_df[plot_df['token_type'] == token_type]
        phase_corrs = phase_data['correlation'].values
        phase_stds = phase_data['std'].values
        
        plt.bar(x_pos + i * width, phase_corrs, width, 
                label=token_type.replace('_', ' ').title(), 
                color=colors[i], alpha=0.8, linewidth=0.5)
    
    # Customize the plot
    plt.xlabel('Problem-Solving Phase', fontsize=16)
    plt.ylabel('Correlation with Cost', fontsize=16)
    # plt.title('Token Type Correlations with Cost Across Problem-Solving Phases', fontsize=18)
    plt.xticks(x_pos + width * 1.5, [p.replace('_', '-').title() for p in PHASES], fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, phase in enumerate(PHASES):
        for j, token_type in enumerate(TOKEN_TYPES):
            value = plot_df[(plot_df['phase'] == phase) & 
                           (plot_df['token_type'] == token_type)]['correlation'].iloc[0]
            std_val = plot_df[(plot_df['phase'] == phase) & 
                             (plot_df['token_type'] == token_type)]['std'].iloc[0]
            if not np.isnan(value):
                # Position text above bar for positive values, below for negative values
                if value >= 0:
                    y_pos = value + 0.01
                    va_align = 'bottom'
                else:
                    y_pos = value - 0.01
                    va_align = 'top'
                
                plt.text(i + j * width, y_pos, f'{value:.3f}', 
                        ha='center', va=va_align, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as {output_path}")

def load_existing_csv(csv_path="cache_correlation_analysis.csv"):
    """
    Load existing CSV file with correlation data.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded existing CSV with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"CSV file {csv_path} not found. Please run the full analysis first.")
        return None

def create_visualization_only(csv_path="cache_correlation_analysis.csv", output_path="cache_correlation_plot.png"):
    """
    Load existing CSV and create visualization with error bars.
    """
    print("Loading existing correlation data...")
    df = load_existing_csv(csv_path)
    
    if df is None:
        return
    
    print("Creating visualization with error bars...")
    create_visualization(df, output_path)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    
    for phase in PHASES:
        print(f"\n{phase.replace('_', '-').title()} Phase:")
        for token_type in TOKEN_TYPES:
            col_name = f"corr_{token_type}_cost_{phase}_avg"
            mean_corr = df[col_name].mean()
            std_corr = df[col_name].std()
            print(f"  {token_type.replace('_', ' ').title()}: {mean_corr:.3f} ± {std_corr:.3f}")
    
    print("\nVisualization complete!")

def main():
    """
    Main function to run the complete analysis.
    """
    print("Starting Cache Correlation Analysis...")
    
    # Load all data
    print("\n1. Loading data from all runs...")
    all_data = load_all_data()
    print(f"Loaded data for {len(all_data)} problem instances")
    
    # Create correlation dataframe
    print("\n2. Calculating correlations...")
    df = create_correlation_dataframe(all_data)
    print(f"Created dataframe with shape: {df.shape}")
    
    # Save to CSV
    csv_path = "cache_correlation_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n3. Saved correlation data to {csv_path}")
    
    # Create visualization
    print("\n4. Creating visualization...")
    create_visualization(df)
    
    # Print summary statistics
    print("\n5. Summary Statistics:")
    print("=" * 50)
    
    for phase in PHASES:
        print(f"\n{phase.replace('_', '-').title()} Phase:")
        for token_type in TOKEN_TYPES:
            col_name = f"corr_{token_type}_cost_{phase}_avg"
            mean_corr = df[col_name].mean()
            std_corr = df[col_name].std()
            print(f"  {token_type.replace('_', ' ').title()}: {mean_corr:.3f} ± {std_corr:.3f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Check if CSV already exists, if so just create visualization
    import os
    if os.path.exists("cache_correlation_analysis.csv"):
        print("CSV file found. Creating visualization with error bars...")
        create_visualization_only()
    else:
        print("CSV file not found. Running full analysis...")
        main()

#!/usr/bin/env python
"""
Token Ratio Analysis

This script analyzes the proportion of different token types within each stage
of problem-solving interactions. It calculates ratios for each token type
across different stages and runs, then creates comprehensive visualizations.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
BASE = Path(".")
RUN_IDS = range(1, 5)  # runs 1 … 4

RUN_DIR_TMPL = (
    "run_{i}"
)
EXTRACT_DIR_TMPL = RUN_DIR_TMPL + "_all_interaction_extract"

# Token types to analyze
TOKEN_TYPES = ['prompt_tokens_noncached', 'completion_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens']

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

def get_token_type_label(token_type):
    """
    Get display label for token type.
    """
    if token_type == 'prompt_tokens_noncached':
        return 'Prompt Tokens (non-cached)'
    else:
        return token_type.replace('_', ' ').title()

def calculate_token_ratios(phase_rounds):
    """
    Calculate ratios of each token type within a phase.
    Returns ratios and total counts for each token type.
    The total includes: completion_tokens + prompt_tokens_noncached + cache_creation_input_tokens + cache_read_input_tokens
    """
    if not phase_rounds:
        return {token_type: {'ratio': np.nan, 'count': 0, 'total': 0} for token_type in TOKEN_TYPES}
    
    # Sum up all tokens in this phase
    total_tokens = {}
    for token_type in TOKEN_TYPES:
        if token_type == 'prompt_tokens_noncached':
            # Calculate prompt_tokens - cache_read_input_tokens for each round
            total_tokens[token_type] = sum(
                round_data.get('prompt_tokens', 0) - round_data.get('cache_read_input_tokens', 0)
                for round_data in phase_rounds
            )
        else:
            total_tokens[token_type] = sum(round_data.get(token_type, 0) for round_data in phase_rounds)
    
    # Calculate total of all token types (should be the 4 components)
    grand_total = sum(total_tokens.values())
    
    # Calculate ratios - ensure they sum to 1.0
    ratios = {}
    if grand_total > 0:
        for token_type in TOKEN_TYPES:
            ratio = total_tokens[token_type] / grand_total
            ratios[token_type] = {
                'ratio': ratio,
                'count': total_tokens[token_type],
                'total': grand_total
            }
        
        # Normalize to ensure ratios sum to exactly 1.0 (handle floating point precision)
        ratio_sum = sum(r['ratio'] for r in ratios.values())
        if ratio_sum > 0 and abs(ratio_sum - 1.0) > 1e-10:  # Only normalize if there's a significant difference
            for token_type in TOKEN_TYPES:
                ratios[token_type]['ratio'] = ratios[token_type]['ratio'] / ratio_sum
    else:
        # If grand_total is 0, return NaN ratios
        for token_type in TOKEN_TYPES:
            ratios[token_type] = {
                'ratio': np.nan,
                'count': total_tokens[token_type],
                'total': grand_total
            }
    
    return ratios

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

def create_ratio_dataframe(all_data):
    """
    Create the comprehensive ratio dataframe.
    """
    results = []
    
    for instance_id, runs_data in all_data.items():
        print(f"Processing instance: {instance_id}")
        
        row = {'instance_id': instance_id}
        
        # Process each phase
        for phase in PHASES:
            # Pre-calculate averaged counts for this phase (for 'avg' run_id)
            # This ensures ratios sum to 1.0 when averaging across runs
            all_counts_avg = {t: [] for t in TOKEN_TYPES}
            for r_id in RUN_IDS:
                if r_id in runs_data:
                    phases_data = divide_into_phases(runs_data[r_id])
                    phase_rounds = phases_data[phase]
                    if phase_rounds:
                        token_ratios = calculate_token_ratios(phase_rounds)
                        for t in TOKEN_TYPES:
                            if not np.isnan(token_ratios[t]['count']):
                                all_counts_avg[t].append(token_ratios[t]['count'])
            
            # Calculate average counts for each token type
            avg_counts = {}
            for t in TOKEN_TYPES:
                avg_counts[t] = np.mean(all_counts_avg[t]) if all_counts_avg[t] else np.nan
            
            # Calculate total from summed averaged counts
            avg_total = sum(v for v in avg_counts.values() if not np.isnan(v))
            
            # Process each token type
            for token_type in TOKEN_TYPES:
                # Process each run + average
                for run_id in list(RUN_IDS) + ['avg']:
                    # Ratio columns
                    ratio_col = f"{token_type}_ratio_{phase}_{run_id}"
                    count_col = f"{token_type}_count_{phase}_{run_id}"
                    total_col = f"{token_type}_total_{phase}_{run_id}"
                    
                    if run_id == 'avg':
                        # Use pre-calculated averaged counts and recalculate ratio
                        avg_count = avg_counts[token_type]
                        if avg_total > 0 and not np.isnan(avg_count):
                            row[ratio_col] = avg_count / avg_total
                        else:
                            row[ratio_col] = np.nan
                        
                        row[count_col] = avg_count
                        row[total_col] = avg_total
                    else:
                        # Individual run ratio
                        if run_id in runs_data:
                            phases_data = divide_into_phases(runs_data[run_id])
                            phase_rounds = phases_data[phase]
                            
                            if phase_rounds:
                                token_ratios = calculate_token_ratios(phase_rounds)
                                row[ratio_col] = token_ratios[token_type]['ratio']
                                row[count_col] = token_ratios[token_type]['count']
                                row[total_col] = token_ratios[token_type]['total']
                            else:
                                row[ratio_col] = np.nan
                                row[count_col] = np.nan
                                row[total_col] = np.nan
                        else:
                            row[ratio_col] = np.nan
                            row[count_col] = np.nan
                            row[total_col] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)

def create_ratio_visualization(df, output_path="token_ratio_plot.png"):
    """
    Create bar plot visualization showing token type ratios across phases.
    """
    # Prepare data for plotting
    plot_data = []
    
    for phase in PHASES:
        for token_type in TOKEN_TYPES:
            col_name = f"{token_type}_ratio_{phase}_avg"
            # Calculate mean and std across all problem instances
            mean_ratio = df[col_name].mean()
            std_ratio = df[col_name].std()
            plot_data.append({
                'phase': phase,
                'token_type': token_type,
                'ratio': mean_ratio,
                'std': std_ratio
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Set up the plot
    x_pos = np.arange(len(PHASES))
    width = 0.2  # Width of bars
    
    # Desired plotting order (left->right) and legend order (top->bottom):
    # Completion Tokens, Cache Creation Input Tokens, Prompt Tokens (non-cached), Cache Read Input Tokens
    plot_token_types = ['completion_tokens', 'cache_creation_input_tokens', 'prompt_tokens_noncached', 'cache_read_input_tokens']

    # Colors aligned to plot_token_types order
    colors_in_order = ['#8B7D9A', '#779ECB', '#ACC2D9', '#2D4A6E']

    for i, token_type in enumerate(plot_token_types):
        phase_data = plot_df[plot_df['token_type'] == token_type]
        phase_ratios = phase_data['ratio'].values
        phase_stds = phase_data['std'].values
        
        plt.bar(x_pos + i * width, phase_ratios, width,
                label=get_token_type_label(token_type),
                color=colors_in_order[i], alpha=0.8, linewidth=0.5)
    
    # Customize the plot
    plt.xlabel('Problem-Solving Phase', fontsize=30)
    plt.ylabel('Proportion of Total Tokens', fontsize=30)
    plt.xticks(x_pos + width * 1.5, [p.replace('_', '-').title() for p in PHASES], fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(0, 1.07)  # Set y-axis maximum to 0.6
    # Force legend order to match plotting order
    handles, labels = plt.gca().get_legend_handles_labels()
    label_order = [get_token_type_label(t) for t in plot_token_types]
    ordered = [next(h for h, l in zip(handles, labels) if l == name) for name in label_order]
    plt.legend(ordered, label_order, loc='upper left', fontsize=15, ncol=2)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    # for i, phase in enumerate(PHASES):
    #     for j, token_type in enumerate(plot_token_types):
    #         value = plot_df[(plot_df['phase'] == phase) &
    #                        (plot_df['token_type'] == token_type)]['ratio'].iloc[0]
    #         if not np.isnan(value):
    #             # Place label just above the bar (no error bars)
    #             plt.text(i + j * width, value + 0.01, f'{value:.3f}',
    #                     ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as {output_path}")

def calculate_missing_noncached_columns(df, all_data):
    """
    Calculate missing prompt_tokens_noncached ratio columns from raw data.
    Adds the columns to the dataframe in place.
    """
    print("Calculating missing prompt_tokens_noncached columns from raw data...")
    
    # Check which columns are missing
    missing_cols = []
    for phase in PHASES:
        for run_id in list(RUN_IDS) + ['avg']:
            ratio_col = f"prompt_tokens_noncached_ratio_{phase}_{run_id}"
            count_col = f"prompt_tokens_noncached_count_{phase}_{run_id}"
            total_col = f"prompt_tokens_noncached_total_{phase}_{run_id}"
            if ratio_col not in df.columns:
                missing_cols.append((phase, run_id))
    
    if not missing_cols:
        print("All prompt_tokens_noncached columns already exist.")
        return df
    
    print(f"Found {len(missing_cols)} missing column sets. Calculating from raw data...")
    
    # Process each instance
    for idx, row in df.iterrows():
        instance_id = row['instance_id']
        
        if instance_id not in all_data:
            print(f"  Warning: Instance {instance_id} not found in raw data. Skipping.")
            continue
        
        runs_data = all_data[instance_id]
        
        # Calculate ratios for missing columns
        for phase, run_id in missing_cols:
            ratio_col = f"prompt_tokens_noncached_ratio_{phase}_{run_id}"
            count_col = f"prompt_tokens_noncached_count_{phase}_{run_id}"
            total_col = f"prompt_tokens_noncached_total_{phase}_{run_id}"
            
            if run_id == 'avg':
                # Calculate average ratio across all runs
                ratios = []
                counts = []
                totals = []
                
                for r_id in RUN_IDS:
                    if r_id in runs_data:
                        phases_data = divide_into_phases(runs_data[r_id])
                        phase_rounds = phases_data[phase]
                        
                        if phase_rounds:
                            token_ratios = calculate_token_ratios(phase_rounds)
                            if not np.isnan(token_ratios['prompt_tokens_noncached']['ratio']):
                                ratios.append(token_ratios['prompt_tokens_noncached']['ratio'])
                                counts.append(token_ratios['prompt_tokens_noncached']['count'])
                                totals.append(token_ratios['prompt_tokens_noncached']['total'])
                
                df.at[idx, ratio_col] = np.mean(ratios) if ratios else np.nan
                df.at[idx, count_col] = np.mean(counts) if counts else np.nan
                df.at[idx, total_col] = np.mean(totals) if totals else np.nan
            else:
                # Individual run ratio
                if run_id in runs_data:
                    phases_data = divide_into_phases(runs_data[run_id])
                    phase_rounds = phases_data[phase]
                    
                    if phase_rounds:
                        token_ratios = calculate_token_ratios(phase_rounds)
                        df.at[idx, ratio_col] = token_ratios['prompt_tokens_noncached']['ratio']
                        df.at[idx, count_col] = token_ratios['prompt_tokens_noncached']['count']
                        df.at[idx, total_col] = token_ratios['prompt_tokens_noncached']['total']
                    else:
                        df.at[idx, ratio_col] = np.nan
                        df.at[idx, count_col] = np.nan
                        df.at[idx, total_col] = np.nan
                else:
                    df.at[idx, ratio_col] = np.nan
                    df.at[idx, count_col] = np.nan
                    df.at[idx, total_col] = np.nan
    
    print("Finished calculating missing columns.")
    return df

def load_existing_csv(csv_path="token_ratio_analysis.csv"):
    """
    Load existing CSV file with ratio data.
    If prompt_tokens_noncached columns are missing, calculate them from raw data.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded existing CSV with shape: {df.shape}")
        
        # Check if prompt_tokens_noncached columns are missing
        sample_col = f"prompt_tokens_noncached_ratio_{PHASES[0]}_avg"
        if sample_col not in df.columns:
            print("\n" + "="*70)
            print("CSV file is missing prompt_tokens_noncached columns.")
            print("Loading raw data to calculate missing ratios...")
            print("="*70 + "\n")
            
            # Load raw data
            all_data = load_all_data()
            if not all_data:
                print("Error: Could not load raw data. Cannot calculate missing columns.")
                return None
            
            # Calculate missing columns
            df = calculate_missing_noncached_columns(df, all_data)
            
            # Save updated CSV
            print(f"\nSaving updated CSV with new columns to {csv_path}...")
            df.to_csv(csv_path, index=False)
            print("CSV updated successfully.")
        
        return df
    except FileNotFoundError:
        print(f"CSV file {csv_path} not found. Please run the full analysis first.")
        return None

def create_visualization_only(csv_path="token_ratio_analysis.csv", output_path="token_ratio_plot.png"):
    """
    Load existing CSV and create visualization.
    """
    print("Loading existing ratio data...")
    df = load_existing_csv(csv_path)
    
    if df is None:
        return
    
    print("Creating ratio visualization...")
    create_ratio_visualization(df, output_path)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    
    for phase in PHASES:
        print(f"\n{phase.replace('_', '-').title()} Phase:")
        for token_type in TOKEN_TYPES:
            col_name = f"{token_type}_ratio_{phase}_avg"
            mean_ratio = df[col_name].mean()
            std_ratio = df[col_name].std()
            print(f"  {get_token_type_label(token_type)}: {mean_ratio:.3f} ± {std_ratio:.3f}")
    
    print("\nVisualization complete!")

def main():
    """
    Main function to run the complete ratio analysis.
    """
    print("Starting Token Ratio Analysis...")
    
    # Load all data
    print("\n1. Loading data from all runs...")
    all_data = load_all_data()
    print(f"Loaded data for {len(all_data)} problem instances")
    
    # Create ratio dataframe
    print("\n2. Calculating token ratios...")
    df = create_ratio_dataframe(all_data)
    print(f"Created dataframe with shape: {df.shape}")
    
    # Save to CSV
    csv_path = "token_ratio_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n3. Saved ratio data to {csv_path}")
    
    # Create visualization
    print("\n4. Creating visualization...")
    create_ratio_visualization(df)
    
    # Print summary statistics
    print("\n5. Summary Statistics:")
    print("=" * 50)
    
    for phase in PHASES:
        print(f"\n{phase.replace('_', '-').title()} Phase:")
        for token_type in TOKEN_TYPES:
            col_name = f"{token_type}_ratio_{phase}_avg"
            mean_ratio = df[col_name].mean()
            std_ratio = df[col_name].std()
            print(f"  {get_token_type_label(token_type)}: {mean_ratio:.3f} ± {std_ratio:.3f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Check if CSV already exists, if so just create visualization
    import os
    if os.path.exists("token_ratio_analysis.csv"):
        print("CSV file found. Creating ratio visualization...")
        create_visualization_only()
    else:
        print("CSV file not found. Running full analysis...")
        main()

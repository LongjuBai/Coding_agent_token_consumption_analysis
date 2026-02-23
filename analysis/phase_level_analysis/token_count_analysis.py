#!/usr/bin/env python
"""
Token Count Analysis

This script analyzes the absolute number of different token types within each stage
of problem-solving interactions. It calculates absolute counts for each token type
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

def calculate_token_counts(phase_rounds):
    """
    Calculate absolute counts of each token type within a phase.
    Returns counts for each token type.
    """
    if not phase_rounds:
        return {token_type: {'count': 0} for token_type in TOKEN_TYPES}
    
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
    
    # Return counts
    counts = {}
    for token_type in TOKEN_TYPES:
        counts[token_type] = {
            'count': total_tokens[token_type]
        }
    
    return counts

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

def create_count_dataframe(all_data):
    """
    Create the comprehensive count dataframe.
    """
    results = []
    
    for instance_id, runs_data in all_data.items():
        print(f"Processing instance: {instance_id}")
        
        row = {'instance_id': instance_id}
        
        # Process each phase
        for phase in PHASES:
            # Pre-calculate averaged counts for this phase (for 'avg' run_id)
            all_counts_avg = {t: [] for t in TOKEN_TYPES}
            for r_id in RUN_IDS:
                if r_id in runs_data:
                    phases_data = divide_into_phases(runs_data[r_id])
                    phase_rounds = phases_data[phase]
                    if phase_rounds:
                        token_counts = calculate_token_counts(phase_rounds)
                        for t in TOKEN_TYPES:
                            if not np.isnan(token_counts[t]['count']):
                                all_counts_avg[t].append(token_counts[t]['count'])
            
            # Calculate average counts for each token type
            avg_counts = {}
            for t in TOKEN_TYPES:
                avg_counts[t] = np.mean(all_counts_avg[t]) if all_counts_avg[t] else np.nan
            
            # Process each token type
            for token_type in TOKEN_TYPES:
                # Process each run + average
                for run_id in list(RUN_IDS) + ['avg']:
                    # Count columns
                    count_col = f"{token_type}_count_{phase}_{run_id}"
                    
                    if run_id == 'avg':
                        # Use pre-calculated averaged counts
                        avg_count = avg_counts[token_type]
                        row[count_col] = avg_count
                    else:
                        # Individual run count
                        if run_id in runs_data:
                            phases_data = divide_into_phases(runs_data[run_id])
                            phase_rounds = phases_data[phase]
                            
                            if phase_rounds:
                                token_counts = calculate_token_counts(phase_rounds)
                                row[count_col] = token_counts[token_type]['count']
                            else:
                                row[count_col] = np.nan
                        else:
                            row[count_col] = np.nan
        
        results.append(row)
    
    return pd.DataFrame(results)

def create_count_visualization(df, output_path="token_count_plot.png"):
    """
    Create bar plot visualization showing token type absolute counts across phases.
    """
    # Prepare data for plotting
    plot_data = []
    
    for phase in PHASES:
        for token_type in TOKEN_TYPES:
            col_name = f"{token_type}_count_{phase}_avg"
            # Calculate mean and std across all problem instances
            mean_count = df[col_name].mean()
            std_count = df[col_name].std()
            plot_data.append({
                'phase': phase,
                'token_type': token_type,
                'count': mean_count,
                'std': std_count
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
        phase_counts = phase_data['count'].values
        phase_stds = phase_data['std'].values
        
        plt.bar(x_pos + i * width, phase_counts, width,
                label=get_token_type_label(token_type),
                color=colors_in_order[i], alpha=0.8, linewidth=0.5,
                yerr=phase_stds, capsize=3)
    
    # Customize the plot
    plt.xlabel('Problem-Solving Phase', fontsize=30)
    plt.ylabel('Number of Tokens', fontsize=30)
    plt.xticks(x_pos + width * 1.5, [p.replace('_', '-').title() for p in PHASES], fontsize=28)
    plt.yticks(fontsize=28)
    # Force legend order to match plotting order
    handles, labels = plt.gca().get_legend_handles_labels()
    label_order = [get_token_type_label(t) for t in plot_token_types]
    ordered = [next(h for h, l in zip(handles, labels) if l == name) for name in label_order]
    plt.legend(ordered, label_order, loc='upper left', fontsize=15, ncol=2)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as {output_path}")

def calculate_missing_noncached_columns(df, all_data):
    """
    Calculate missing prompt_tokens_noncached count columns from raw data.
    Adds the columns to the dataframe in place.
    """
    print("Calculating missing prompt_tokens_noncached columns from raw data...")
    
    # Check which columns are missing
    missing_cols = []
    for phase in PHASES:
        for run_id in list(RUN_IDS) + ['avg']:
            count_col = f"prompt_tokens_noncached_count_{phase}_{run_id}"
            if count_col not in df.columns:
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
        
        # Calculate counts for missing columns
        for phase, run_id in missing_cols:
            count_col = f"prompt_tokens_noncached_count_{phase}_{run_id}"
            
            if run_id == 'avg':
                # Calculate average count across all runs
                counts = []
                
                for r_id in RUN_IDS:
                    if r_id in runs_data:
                        phases_data = divide_into_phases(runs_data[r_id])
                        phase_rounds = phases_data[phase]
                        
                        if phase_rounds:
                            token_counts = calculate_token_counts(phase_rounds)
                            if not np.isnan(token_counts['prompt_tokens_noncached']['count']):
                                counts.append(token_counts['prompt_tokens_noncached']['count'])
                
                df.at[idx, count_col] = np.mean(counts) if counts else np.nan
            else:
                # Individual run count
                if run_id in runs_data:
                    phases_data = divide_into_phases(runs_data[run_id])
                    phase_rounds = phases_data[phase]
                    
                    if phase_rounds:
                        token_counts = calculate_token_counts(phase_rounds)
                        df.at[idx, count_col] = token_counts['prompt_tokens_noncached']['count']
                    else:
                        df.at[idx, count_col] = np.nan
                else:
                    df.at[idx, count_col] = np.nan
    
    print("Finished calculating missing columns.")
    return df

def load_existing_csv(csv_path="token_count_analysis.csv"):
    """
    Load existing CSV file with count data.
    If prompt_tokens_noncached columns are missing, calculate them from raw data.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded existing CSV with shape: {df.shape}")
        
        # Check if prompt_tokens_noncached columns are missing
        sample_col = f"prompt_tokens_noncached_count_{PHASES[0]}_avg"
        if sample_col not in df.columns:
            print("\n" + "="*70)
            print("CSV file is missing prompt_tokens_noncached columns.")
            print("Loading raw data to calculate missing counts...")
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

def create_visualization_only(csv_path="token_count_analysis.csv", output_path="token_count_plot.png"):
    """
    Load existing CSV and create visualization.
    """
    print("Loading existing count data...")
    df = load_existing_csv(csv_path)
    
    if df is None:
        return
    
    print("Creating count visualization...")
    create_count_visualization(df, output_path)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    
    for phase in PHASES:
        print(f"\n{phase.replace('_', '-').title()} Phase:")
        for token_type in TOKEN_TYPES:
            col_name = f"{token_type}_count_{phase}_avg"
            mean_count = df[col_name].mean()
            std_count = df[col_name].std()
            print(f"  {get_token_type_label(token_type)}: {mean_count:.1f} ± {std_count:.1f}")
    
    print("\nVisualization complete!")

def main():
    """
    Main function to run the complete count analysis.
    """
    print("Starting Token Count Analysis...")
    
    # Load all data
    print("\n1. Loading data from all runs...")
    all_data = load_all_data()
    print(f"Loaded data for {len(all_data)} problem instances")
    
    # Create count dataframe
    print("\n2. Calculating token counts...")
    df = create_count_dataframe(all_data)
    print(f"Created dataframe with shape: {df.shape}")
    
    # Save to CSV
    csv_path = "token_count_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n3. Saved count data to {csv_path}")
    
    # Create visualization
    print("\n4. Creating visualization...")
    create_count_visualization(df)
    
    # Print summary statistics
    print("\n5. Summary Statistics:")
    print("=" * 50)
    
    for phase in PHASES:
        print(f"\n{phase.replace('_', '-').title()} Phase:")
        for token_type in TOKEN_TYPES:
            col_name = f"{token_type}_count_{phase}_avg"
            mean_count = df[col_name].mean()
            std_count = df[col_name].std()
            print(f"  {get_token_type_label(token_type)}: {mean_count:.1f} ± {std_count:.1f}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    # Check if CSV already exists, if so just create visualization
    import os
    if os.path.exists("token_count_analysis.csv"):
        print("CSV file found. Creating count visualization...")
        create_visualization_only()
    else:
        print("CSV file not found. Running full analysis...")
        main()

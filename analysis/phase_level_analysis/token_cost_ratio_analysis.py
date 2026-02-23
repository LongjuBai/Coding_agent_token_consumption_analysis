#!/usr/bin/env python
"""
Token Cost Ratio Analysis

This script computes per-phase cost ratios derived from exact token numbers
using defined pricing rules. It aggregates across runs and instances, writes a
CSV, and produces a grouped bar chart with error bars.
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

# Cost components (derived from token types via pricing rules)
COST_COMPONENTS = [
    'base_input_cost',
    'cache_write_cost',
    'cache_hit_cost',
    'output_cost',
]

# Phase names (same as token ratio analysis)
PHASES = ['early', 'early_mid', 'mid', 'later_mid', 'later']

# Pricing (per million tokens)
PRICING = {
    'base_input_rate': 3.00,         # $3/MTok for base input tokens
    'cache_write_1h_rate': 3.75,     # $3.75/MTok for cache creation (1h cache)
    'cache_hit_rate': 0.30,          # $0.30/MTok for cache read
    'output_rate': 15.00,            # $15/MTok for completion tokens
}

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
        phase_size = rounds_per_phase + (1 if i < remainder else 0)
        end_idx = start_idx + phase_size

        sorted_rounds = sorted(rounds_data.keys(),
                               key=lambda k: int(k.split('interaction_')[1].split('__')[0]))

        phases[phase] = [rounds_data[round_key] for round_key in sorted_rounds[start_idx:end_idx]]
        start_idx = end_idx

    return phases

def compute_round_cost_components(round_data):
    """
    Compute cost components for a single round from token numbers using pricing rules.
    Returns a dict with component costs in dollars.
    """
    prompt_tokens = round_data.get('prompt_tokens', 0)
    completion_tokens = round_data.get('completion_tokens', 0)
    cache_creation_input_tokens = round_data.get('cache_creation_input_tokens', 0)
    cache_read_input_tokens = round_data.get('cache_read_input_tokens', 0)

    base_input_tokens = max(0, prompt_tokens - cache_read_input_tokens)

    base_input_cost = (base_input_tokens * PRICING['base_input_rate']) / 1_000_000
    cache_write_cost = (cache_creation_input_tokens * PRICING['cache_write_1h_rate']) / 1_000_000
    cache_hit_cost = (cache_read_input_tokens * PRICING['cache_hit_rate']) / 1_000_000
    output_cost = (completion_tokens * PRICING['output_rate']) / 1_000_000

    return {
        'base_input_cost': base_input_cost,
        'cache_write_cost': cache_write_cost,
        'cache_hit_cost': cache_hit_cost,
        'output_cost': output_cost,
    }

def calculate_cost_ratios(phase_rounds):
    """
    Sum cost components over rounds in a phase and compute component ratios.
    Returns {component: {ratio, cost, total}}.
    """
    if not phase_rounds:
        return {comp: {'ratio': np.nan, 'cost': 0.0, 'total': 0.0} for comp in COST_COMPONENTS}

    total_costs = {comp: 0.0 for comp in COST_COMPONENTS}
    for round_data in phase_rounds:
        comp_costs = compute_round_cost_components(round_data)
        for comp in COST_COMPONENTS:
            total_costs[comp] += comp_costs[comp]

    grand_total = sum(total_costs.values())

    ratios = {}
    for comp in COST_COMPONENTS:
        ratio = (total_costs[comp] / grand_total) if grand_total > 0 else np.nan
        ratios[comp] = {
            'ratio': ratio,
            'cost': total_costs[comp],
            'total': grand_total,
        }

    return ratios

def load_all_data():
    """
    Load all summary_rounds_withCache.json files from all runs.
    Structure: {instance_id: {run_id: rounds_data}}
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

def create_cost_ratio_dataframe(all_data):
    """
    Create a dataframe of cost ratios and component totals per phase, per run, plus across-run averages.
    Rows are instance_ids; columns are
    {component}_ratio_{phase}_{run_id|avg}, {component}_cost_{phase}_{run_id|avg}, {component}_total_{phase}_{run_id|avg}
    """
    results = []

    for instance_id, runs_data in all_data.items():
        print(f"Processing instance: {instance_id}")
        row = {'instance_id': instance_id}

        for comp in COST_COMPONENTS:
            for phase in PHASES:
                for run_id in list(RUN_IDS) + ['avg']:
                    ratio_col = f"{comp}_ratio_{phase}_{run_id}"
                    cost_col = f"{comp}_cost_{phase}_{run_id}"
                    total_col = f"{comp}_total_{phase}_{run_id}"

                    if run_id == 'avg':
                        ratios, costs, totals = [], [], []
                        for r_id in RUN_IDS:
                            if r_id in runs_data:
                                phases_data = divide_into_phases(runs_data[r_id])
                                phase_rounds = phases_data[phase]
                                if phase_rounds:
                                    comp_ratios = calculate_cost_ratios(phase_rounds)
                                    if not np.isnan(comp_ratios[comp]['ratio']):
                                        ratios.append(comp_ratios[comp]['ratio'])
                                        costs.append(comp_ratios[comp]['cost'])
                                        totals.append(comp_ratios[comp]['total'])
                        row[ratio_col] = np.mean(ratios) if ratios else np.nan
                        row[cost_col] = np.mean(costs) if costs else np.nan
                        row[total_col] = np.mean(totals) if totals else np.nan
                    else:
                        if run_id in runs_data:
                            phases_data = divide_into_phases(runs_data[run_id])
                            phase_rounds = phases_data[phase]
                            if phase_rounds:
                                comp_ratios = calculate_cost_ratios(phase_rounds)
                                row[ratio_col] = comp_ratios[comp]['ratio']
                                row[cost_col] = comp_ratios[comp]['cost']
                                row[total_col] = comp_ratios[comp]['total']
                            else:
                                row[ratio_col] = np.nan
                                row[cost_col] = np.nan
                                row[total_col] = np.nan
                        else:
                            row[ratio_col] = np.nan
                            row[cost_col] = np.nan
                            row[total_col] = np.nan

        results.append(row)

    return pd.DataFrame(results)

def create_cost_ratio_visualization(df, output_path="token_cost_ratio_plot.png"):
    """
    Create bar plot visualization showing cost component ratios across phases.
    Bars are the mean across instances of the across-run-averaged ratios; error bars are ±1 std across instances.
    """
    plot_data = []

    for phase in PHASES:
        for comp in COST_COMPONENTS:
            col_name = f"{comp}_ratio_{phase}_avg"
            mean_ratio = df[col_name].mean()
            std_ratio = df[col_name].std()
            plot_data.append({
                'phase': phase,
                'component': comp,
                'ratio': mean_ratio,
                'std': std_ratio,
            })

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 8))
    x_pos = np.arange(len(PHASES))
    width = 0.2

    # Map cost components to desired legend names
    component_to_label = {
        'base_input_cost': 'Prompt Tokens (non-cached)',
        'cache_write_cost': 'Cache Creation Input Tokens',
        'cache_hit_cost': 'Cache Read Input Tokens',
        'output_cost': 'Completion Tokens',
    }
    # Explicit color order aligned to plot_components left->right
    colors_in_order = ['#8B7D9A', '#779ECB', '#ACC2D9', '#2D4A6E']

    # Desired plotting order (left->right) and legend order (top->bottom):
    # Completion Tokens, Cache Creation Input Tokens, Prompt Tokens (non-cached), Cache Read Input Tokens
    plot_components = ['output_cost', 'cache_write_cost', 'base_input_cost', 'cache_hit_cost']

    for i, comp in enumerate(plot_components):
        phase_data = plot_df[plot_df['component'] == comp]
        phase_ratios = phase_data['ratio'].values
        phase_stds = phase_data['std'].values

        plt.bar(x_pos + i * width, phase_ratios, width,
                label=component_to_label.get(comp, comp.replace('_', ' ').title()),
                color=colors_in_order[i], alpha=0.8, linewidth=0.5)

    plt.xlabel('Problem-Solving Phase', fontsize=30)
    plt.ylabel('Proportion of Total Cost', fontsize=30)
    plt.xticks(x_pos + width * 1.5, [p.replace('_', '-').title() for p in PHASES], fontsize=28)
    plt.yticks(fontsize=28)
    # Force legend order to match plotting order
    handles, labels = plt.gca().get_legend_handles_labels()
    label_order = [component_to_label[c] for c in plot_components]
    ordered = [next(h for h, l in zip(handles, labels) if l == name) for name in label_order]
    plt.legend(ordered, label_order, loc='upper left', fontsize=18)
    plt.grid(True, alpha=0.3, axis='y')

    # for i, phase in enumerate(PHASES):
    #     for j, comp in enumerate(plot_components):
    #         value_series = plot_df[(plot_df['phase'] == phase) & (plot_df['component'] == comp)]['ratio']
    #         if not value_series.empty:
    #             value = value_series.iloc[0]
    #             if not np.isnan(value):
    #                 plt.text(i + j * width, value + 0.01,
    #                          f'{value:.3f}',
    #                          ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Plot saved as {output_path}")

def load_existing_csv(csv_path="token_cost_ratio_analysis.csv"):
    """
    Load existing CSV file with cost ratio data.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded existing CSV with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"CSV file {csv_path} not found. Please run the full analysis first.")
        return None

def create_visualization_only(csv_path="token_cost_ratio_analysis.csv", output_path="token_cost_ratio_plot.png"):
    """
    Load existing CSV and create visualization and summary stats.
    """
    print("Loading existing cost ratio data...")
    df = load_existing_csv(csv_path)

    if df is None:
        return

    print("Creating cost ratio visualization...")
    create_cost_ratio_visualization(df, output_path)

    print("\nSummary Statistics:")
    print("=" * 50)
    for phase in PHASES:
        print(f"\n{phase.replace('_', '-').title()} Phase:")
        for comp in COST_COMPONENTS:
            col_name = f"{comp}_ratio_{phase}_avg"
            mean_ratio = df[col_name].mean()
            std_ratio = df[col_name].std()
            print(f"  {comp.replace('_', ' ').title()}: {mean_ratio:.3f} ± {std_ratio:.3f}")

    print("\nVisualization complete!")

def main():
    """
    Run the complete cost ratio analysis: load, compute, save CSV, plot, summarize.
    """
    print("Starting Token Cost Ratio Analysis...")

    print("\n1. Loading data from all runs...")
    all_data = load_all_data()
    print(f"Loaded data for {len(all_data)} problem instances")

    print("\n2. Calculating cost ratios...")
    df = create_cost_ratio_dataframe(all_data)
    print(f"Created dataframe with shape: {df.shape}")

    csv_path = "token_cost_ratio_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n3. Saved cost ratio data to {csv_path}")

    print("\n4. Creating visualization...")
    create_cost_ratio_visualization(df)

    print("\n5. Summary Statistics:")
    print("=" * 50)
    for phase in PHASES:
        print(f"\n{phase.replace('_', '-').title()} Phase:")
        for comp in COST_COMPONENTS:
            col_name = f"{comp}_ratio_{phase}_avg"
            mean_ratio = df[col_name].mean()
            std_ratio = df[col_name].std()
            print(f"  {comp.replace('_', ' ').title()}: {mean_ratio:.3f} ± {std_ratio:.3f}")

    print("\nAnalysis complete!")

if __name__ == "__main__":
    import os
    if os.path.exists("token_cost_ratio_analysis.csv"):
        print("CSV file found. Creating cost ratio visualization...")
        create_visualization_only()
    else:
        print("CSV file not found. Running full cost ratio analysis...")
        main()


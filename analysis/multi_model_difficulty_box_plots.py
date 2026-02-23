"""
Multi-Model Difficulty Level Box Plots

This script aggregates data from multiple models and creates box plots showing
the distribution of token costs and tool usages across difficulty levels.

Creates 3 box plots:
1. Prompt Tokens by Difficulty
2. Completion Tokens by Difficulty
3. Tool Usages by Difficulty
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent

def discover_model_dataset_dirs() -> dict:
    """Discover model dataset directories from available mean-token CSV files."""
    model_dirs = {}
    pattern = "*/swe_bench_token_cost_aggregated_total_with_accuracy_col_mean.csv"
    for csv_path in sorted(DATASET_ROOT.glob(pattern)):
        model_dir = csv_path.parent.name
        model_dirs[model_dir] = csv_path.parent
    return model_dirs


MODEL_DATASET_DIRS = discover_model_dataset_dirs()

# Difficulty level order (excluding >4 hours)
difficulty_order = ["<15 min fix", "15 min - 1 hour", "1-4 hours"]

# Colors for each metric
colors = {
    "prompt": "#A0BFE0",
    "completion": "#7895CB",
    "tool": "#4A55A2",
}

# Set larger font sizes globally
plt.rcParams.update({'font.size': 14})

print("Loading data from all models...")

# Load data from all models
all_data = []
for model_name, dataset_dir in MODEL_DATASET_DIRS.items():
    csv_path = Path(dataset_dir) / "swe_bench_token_cost_aggregated_total_with_accuracy_col_mean.csv"
    if csv_path.exists():
        df_model = pd.read_csv(csv_path)
        df_model['model'] = model_name
        all_data.append(df_model)
        print(f"  Loaded data from {model_name}: {len(df_model)} instances")
    else:
        print(f"  Warning: File not found for {model_name}: {csv_path}")

if not all_data:
    raise ValueError("No data files found!")

# Combine all models
df_all = pd.concat(all_data, ignore_index=True)
print(f"\nTotal instances across all models: {len(df_all)}")
print(f"Total models: {len(df_all['model'].unique())}")

# Set difficulty as categorical with order
df_all["difficulty"] = pd.Categorical(
    df_all["difficulty"], categories=difficulty_order, ordered=True
)

# Remove rows with missing difficulty or ">4 hours"
df_all = df_all.dropna(subset=['difficulty'])
df_all = df_all[df_all["difficulty"].isin(difficulty_order)]

print("\nNumber of instances by difficulty level (across all models):")
difficulty_counts = df_all['difficulty'].value_counts()
for difficulty in difficulty_order:
    count = difficulty_counts.get(difficulty, 0)
    print(f"  {difficulty}: {count} instances")

# ============================================================================
# Create Box Plots
# ============================================================================

print("\n" + "="*80)
print("Creating Box Plots")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True)

# Consistent numeric positions for ALL subplots
pos = np.arange(1, len(difficulty_order) + 1)

# Prepare data for the box plots
# For each difficulty level, collect all values across all models
prompt_data = []
completion_data = []
tool_data = []

for diff in difficulty_order:
    group_data = df_all[df_all["difficulty"] == diff]
    
    # Prompt tokens
    prompt_values = group_data["total_prompt_tokens_mean"].dropna().values
    prompt_data.append(prompt_values)
    
    # Completion tokens
    completion_values = group_data["total_completion_tokens_mean"].dropna().values
    completion_data.append(completion_values)
    
    # Tool usages
    tool_values = group_data["total_tool_usages_mean"].dropna().values
    tool_data.append(tool_values)

# 1) Prompt tokens (box)
bp1 = axes[0].boxplot(prompt_data, positions=pos, patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor(colors["prompt"])
    patch.set_alpha(0.8)
axes[0].set_title("a) Prompt Tokens", fontsize=22)
axes[0].set_ylabel("Tokens", fontsize=20)
axes[0].set_xticks(pos)
axes[0].set_xticklabels(difficulty_order)

# 2) Completion tokens (box)
bp2 = axes[1].boxplot(completion_data, positions=pos, patch_artist=True)
for patch in bp2['boxes']:
    patch.set_facecolor(colors["completion"])
    patch.set_alpha(0.8)
axes[1].set_title("b) Completion Tokens", fontsize=22)
axes[1].set_ylabel("Tokens", fontsize=20)
axes[1].ticklabel_format(axis='y', style='sci', scilimits=(5, 5))
axes[1].set_xticks(pos)
axes[1].set_xticklabels(difficulty_order)

# 3) Tool usages (box)
bp3 = axes[2].boxplot(tool_data, positions=pos, patch_artist=True)
for patch in bp3['boxes']:
    patch.set_facecolor(colors["tool"])
    patch.set_alpha(0.8)
axes[2].set_title("c) Tool Usages", fontsize=22)
axes[2].set_ylabel("Tool Usages", fontsize=20)
axes[2].set_xticks(pos)
axes[2].set_xticklabels(difficulty_order)

# Common cosmetics
for ax in axes:
    ax.tick_params(axis='x', labelsize=18, rotation=15)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid(True, axis='y', alpha=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

fig.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot
output_path = OUTPUT_DIR / "multi_model_difficulty_box_plots.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nBox plots saved to: {output_path}")
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("Summary Statistics by Difficulty Level")
print("="*80)

for diff in difficulty_order:
    group_data = df_all[df_all["difficulty"] == diff]
    if len(group_data) > 0:
        print(f"\n{diff}:")
        print(f"  Number of instances: {len(group_data)}")
        print(f"  Prompt tokens - Mean: {group_data['total_prompt_tokens_mean'].mean():.2f}, "
              f"Median: {group_data['total_prompt_tokens_mean'].median():.2f}")
        print(f"  Completion tokens - Mean: {group_data['total_completion_tokens_mean'].mean():.2f}, "
              f"Median: {group_data['total_completion_tokens_mean'].median():.2f}")
        print(f"  Tool usages - Mean: {group_data['total_tool_usages_mean'].mean():.2f}, "
              f"Median: {group_data['total_tool_usages_mean'].median():.2f}")

# ============================================================================
# Cross-Group Comparison Analysis
# ============================================================================

print("\n" + "="*80)
print("Cross-Group Comparison: <15 min fix vs 1-4 hours")
print("="*80)

# Compute total tokens per instance
df_all = df_all.copy()
df_all['total_tokens'] = (
    df_all['total_prompt_tokens_mean'].fillna(0)
    + df_all['total_completion_tokens_mean'].fillna(0)
)

# Focus on the two groups of interest
lt15_label = '<15 min fix'
gt1h_label = '1-4 hours'

lt15 = df_all[df_all['difficulty'] == lt15_label].dropna(subset=['total_tokens'])
gt1h = df_all[df_all['difficulty'] == gt1h_label].dropna(subset=['total_tokens'])

n_lt15 = len(lt15)
n_gt1h = len(gt1h)

# If any group is empty, handle gracefully
if n_lt15 == 0 or n_gt1h == 0:
    print(f"Insufficient data: <15 min fix count={n_lt15}, 1-4 hours count={n_gt1h}")
else:
    # Thresholds: use the mean total tokens of the target group
    gt1h_mean = gt1h['total_tokens'].mean()
    lt15_mean = lt15['total_tokens'].mean()

    # 1) How many <15 min fix instances cost more total tokens than the 1-4 hours mean?
    lt15_more_than_gt1h_mean = int((lt15['total_tokens'] > gt1h_mean).sum())
    lt15_more_pct = 100.0 * lt15_more_than_gt1h_mean / n_lt15 if n_lt15 else 0.0

    # 2) How many 1-4 hours instances cost less total tokens than the <15 min fix mean?
    gt1h_less_than_lt15_mean = int((gt1h['total_tokens'] < lt15_mean).sum())
    gt1h_less_pct = 100.0 * gt1h_less_than_lt15_mean / n_gt1h if n_gt1h else 0.0

    # Print results
    print(f"Comparison based on per-group mean total tokens")
    print(f"- {lt15_label} vs {gt1h_label} mean:")
    print(f"  Count: {lt15_more_than_gt1h_mean} out of {n_lt15}  |  Percentage: {lt15_more_pct:.2f}%")
    print(f"- {gt1h_label} vs {lt15_label} mean:")
    print(f"  Count: {gt1h_less_than_lt15_mean} out of {n_gt1h}  |  Percentage: {gt1h_less_pct:.2f}%")

    # (Optional) Show the means for context
    print(f"\nGroup means (total tokens): {lt15_label}={lt15_mean:.2f}, {gt1h_label}={gt1h_mean:.2f}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

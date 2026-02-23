"""
Multi-Model Variance Plot

This script aggregates the "sort and show variance" plot across multiple LLM models.
It shows the overall pooled statistics (mean ± std) across all models, with optional
overlay of individual model lines to show between-model variability.

Approach:
1. Load data from multiple models
2. For each instance, pool all runs across all models
3. Calculate mean and std across the pooled data
4. Sort instances by mean and plot with variance bands
5. Optionally overlay individual model lines with transparency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"
OUTPUT_DIR = Path(__file__).resolve().parent

def discover_model_paths() -> dict:
    """Discover model CSV paths under dataset root."""
    model_paths = {}
    pattern = "*/swe_bench_token_cost_aggregated_total_with_accuracy.csv"
    for csv_path in sorted(DATASET_ROOT.glob(pattern)):
        model_dir = csv_path.parent.name
        model_paths[model_dir] = csv_path
    return model_paths


MODEL_PATHS = discover_model_paths()

# Define metrics and their corresponding run columns
metric_config = {
    "Total Prompt Tokens": [f"total_prompt_tokens_run{i}" for i in range(1, 5)],
    "Total Completion Tokens": [f"total_completion_tokens_run{i}" for i in range(1, 5)],
    "Total Cost": [f"total_cost_run{i}" for i in range(1, 5)],
}

# Define colors for each metric (prompt, completion, cost)
metric_colors = {
    "Total Prompt Tokens": "#A0BFE0",       # light blue
    "Total Completion Tokens": "#7895CB",   # medium blue
    "Total Cost": "#4A55A2",                # dark blue
}

# Set larger font sizes globally (both plots)
plt.rcParams.update({"font.size": 20})
print("Loading data from multiple models...")
# Load and combine data from all models
all_data = []
for model_name, csv_path in MODEL_PATHS.items():
    if Path(csv_path).exists():
        df_model = pd.read_csv(csv_path)
        df_model['model'] = model_name
        all_data.append(df_model)
        print(f"  Loaded {len(df_model)} instances from {model_name}")
    else:
        print(f"  Warning: File not found for {model_name}: {csv_path}")

if not all_data:
    raise ValueError("No data files found!")

# Combine all models
df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal instances across all models: {len(df)}")
print(f"Models included: {df['model'].unique()}\n")

# Create a single figure with 3 subplots stacked vertically
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Process each metric and plot in its own subplot
for idx, (metric_name, cols) in enumerate(metric_config.items()):
    print(f"\nProcessing {metric_name}...")
    
    # Method 1: Pooled approach - combine all runs across all models
    # Group by instance_id to pool data across models
    pooled_data = []
    
    for instance_id in df['instance_id'].unique():
        instance_data = df[df['instance_id'] == instance_id]
        
        # Get all run values across all models for this instance
        all_values = []
        for _, row in instance_data.iterrows():
            for col in cols:
                val = row[col]
                if pd.notna(val):
                    all_values.append(float(val))
        
        if len(all_values) > 0:
            pooled_data.append({
                'instance_id': instance_id,
                'mean': np.mean(all_values),
                'std': np.std(all_values),
                'values': all_values
            })
    
    pooled_df = pd.DataFrame(pooled_data)
    
    if len(pooled_df) == 0:
        print(f"  Warning: No pooled data for {metric_name}")
        continue
    
    # Sort by mean for plotting
    sorted_indices = pooled_df['mean'].argsort()
    sorted_mean = pooled_df['mean'].iloc[sorted_indices].reset_index(drop=True)
    sorted_std = pooled_df['std'].iloc[sorted_indices].reset_index(drop=True)
    
    # Plot in the corresponding subplot
    ax = axes[idx]
    
    # Plot the main pooled mean and std
    ax.plot(sorted_mean, label="Pooled Mean (All Models)", 
           color=metric_colors[metric_name], linewidth=3, zorder=3)
    ax.fill_between(
        x=range(len(sorted_mean)),
        y1=sorted_mean - sorted_std,
        y2=sorted_mean + sorted_std,
        alpha=0.5,
        label="Pooled ±1 Std Dev",
        color=metric_colors[metric_name],
        zorder=2
    )
    
    # Set labels and titles
    ax.set_title(f"{metric_name} (Pooled Across All Models)", 
                fontsize=22, fontweight="bold")
    ax.set_xlabel("Instance Index (sorted by pooled mean)", fontsize=22)
    ax.set_ylabel(metric_name, fontsize=22)
    ax.tick_params(axis="both", labelsize=20)
    
    # Set y-axis limits for better visibility
    if metric_name == "Total Prompt Tokens":
        ax.set_ylim(-0.5e7, 1.5e7)
    elif metric_name == "Total Completion Tokens":
        y_min, _ = ax.get_ylim()
        ax.set_ylim(y_min, 80000)
    elif metric_name == "Total Cost":
        y_min, _ = ax.get_ylim()
        ax.set_ylim(y_min, max(ax.get_ylim()[1], sorted_mean.max() * 1.1))
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=metric_colors[metric_name], linewidth=3, label='Pooled Mean (All Models)'),
        plt.Rectangle((0,0),1,1, facecolor=metric_colors[metric_name], alpha=0.5, label='Pooled ±1 Std Dev'),
    ]
    ax.legend(handles=legend_elements, fontsize=20, loc="best")
    
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # Print summary statistics
    print(f"  Summary statistics:")
    print(f"    Total instances: {len(pooled_df)}")
    print(f"    Mean of means: {sorted_mean.mean():.2e}")
    print(f"    Mean std: {sorted_std.mean():.2e}")
    print(f"    Coefficient of variation: {(sorted_std / sorted_mean).mean():.4f}")

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the first figure (variance plot)
output_path = OUTPUT_DIR / "multi_model_variance_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure 1 saved to: {output_path}")
plt.close()

# ---------------------------------------------------------------------------
# Second plot: Token and cost bar plot (total tokens and dollar cost per model)
# ---------------------------------------------------------------------------
prompt_cols = [f"total_prompt_tokens_run{i}" for i in range(1, 5)]
completion_cols = [f"total_completion_tokens_run{i}" for i in range(1, 5)]
tool_cols = [f"total_tool_usages_run{i}" for i in range(1, 5)]
round_cols = [f"total_interaction_rounds_run{i}" for i in range(1, 5)]
cost_cols = [f"total_cost_run{i}" for i in range(1, 5)]

model_agg = []
for model_name in df["model"].unique():
    df_m = df[df["model"] == model_name]
    # Total token = input + output (mean across instances and runs)
    input_tokens = df_m[prompt_cols].values.flatten()
    output_tokens = df_m[completion_cols].values.flatten()
    input_tokens = input_tokens[~np.isnan(input_tokens)]
    output_tokens = output_tokens[~np.isnan(output_tokens)]
    mean_input = np.mean(input_tokens) if len(input_tokens) else 0
    mean_output = np.mean(output_tokens) if len(output_tokens) else 0
    total_tokens = mean_input + mean_output

    tool_vals = df_m[tool_cols].values.flatten() if all(c in df_m.columns for c in tool_cols) else np.array([np.nan])
    tool_vals = tool_vals[~np.isnan(tool_vals)]
    tool_count = np.mean(tool_vals) if len(tool_vals) else np.nan

    round_vals = df_m[round_cols].values.flatten() if all(c in df_m.columns for c in round_cols) else np.array([np.nan])
    round_vals = round_vals[~np.isnan(round_vals)]
    turns = np.mean(round_vals) if len(round_vals) else np.nan

    # Cost values (USD) across runs
    cost_vals = (
        df_m[cost_cols].values.flatten()
        if all(c in df_m.columns for c in cost_cols)
        else np.array([np.nan])
    )
    cost_vals = cost_vals[~np.isnan(cost_vals)]
    mean_cost = np.mean(cost_vals) if len(cost_vals) else 0

    # Second bar: average total cost per task
    second_bar_value = mean_cost

    model_agg.append({
        "model": model_name,
        "total_tokens": total_tokens,
        "tool_count": tool_count,
        "turns": turns,
        "cost": mean_cost,
        "second_bar": second_bar_value,
    })

agg_df = pd.DataFrame(model_agg)
# Reorder models alphabetically
order = sorted(agg_df["model"].tolist())
agg_df = agg_df.set_index("model").loc[order].reset_index()
models = agg_df["model"].tolist()
x = np.arange(len(models))
width = 0.35

# Cost plot
fig2, ax1 = plt.subplots(figsize=(16, 12))
# Bar colors aligned with token/cost palette
BAR_COLOR_TOKENS = "#A0BFE0"
BAR_COLOR_COST = "#4A55A2"
bars1 = ax1.bar(x - width / 2, agg_df["total_tokens"], width, label="Total tokens", color=BAR_COLOR_TOKENS)
ax1.set_xlabel("Model", fontsize=22)
ax1.set_ylabel("Token count", fontsize=22)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha="right", fontsize=20)
ax1.tick_params(axis="y", labelsize=20)
ax1.legend(loc="upper left", fontsize=20)
ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

# Second bar: average dollar cost on secondary y-axis
ax2 = ax1.twinx()
bars2 = ax2.bar(x + width / 2, agg_df["second_bar"], width, label="Cost", color=BAR_COLOR_COST)
ax2.set_ylabel("Cost ($)", fontsize=22)
ax2.tick_params(axis="y", labelsize=20)
ax2.legend(loc="upper right", fontsize=20)

ax1.set_title("Total token and cost by model (mean across instances and runs)", fontsize=22, fontweight="bold")
fig2.tight_layout()

output_path2 = OUTPUT_DIR / "multi_model_token_cost_bar_plot.png"
fig2.savefig(output_path2, dpi=300, bbox_inches="tight")
print(f"Figure 2 saved to: {output_path2}")
plt.close(fig2)

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
print("\nOutput: 2 plots")
print("  1. Variance plot (pooled mean ± std, no model bars): multi_model_variance_plot.png")
print("  2. Token cost bar plot (total tokens, tool count per model): multi_model_token_cost_bar_plot.png")

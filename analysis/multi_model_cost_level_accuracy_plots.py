"""
Cost Level Analysis (Multi-Model with Regression)

This script creates six plots:
1. Prompt Tokens - Accuracy vs Cost Level
2. Completion Tokens - Accuracy vs Cost Level
3. Prompt Tokens - Repeated Modify vs Cost Level
4. Completion Tokens - Repeated Modify vs Cost Level
5. Prompt Tokens - Repeated View vs Cost Level
6. Completion Tokens - Repeated View vs Cost Level

The cost levels are: MinCost (lowest) -> LowerCost -> UpperCost -> MaxCost (highest)

Uses regression analysis with categorical cost levels to estimate coefficients.
All plots include error bars and p-values are reported.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import string

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
except ImportError:
    raise ImportError("statsmodels is required. Install with: pip install statsmodels")

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT / "analysis"
OUTPUT_DIR = Path(__file__).resolve().parent

def format_pvalue(pval):
    """Format p-value for display, using scientific notation for very small values."""
    if pval < 0.0001:
        return f"{pval:.2e}"
    else:
        return f"{pval:.4f}"

def wrap_label(label, max_chars_per_line=30):
    """Wrap a label into two lines if it's too long."""
    # For ylabels like "Accuracy Coefficient (vs MinCost)" or "Repeated Modify Coefficient (vs MinCost)"
    if "Coefficient (vs MinCost)" in label:
        # Split before "Coefficient" to keep the main part together
        parts = label.split(" Coefficient", 1)
        return f"{parts[0]}\nCoefficient{parts[1]}"
    
    # For titles with ": " pattern like "Prompt Tokens: Accuracy vs Cost Level (Regression Coefficients)"
    if ": " in label:
        parts = label.split(": ", 1)
        # Split the second part further if needed
        second_part = parts[1]
        if " (" in second_part:
            # Split before the parentheses to keep main title together
            subparts = second_part.split(" (", 1)
            return f"{parts[0]}: {subparts[0]}\n({subparts[1]}"
        elif " vs " in second_part:
            subparts = second_part.split(" vs ", 1)
            return f"{parts[0]}:\n{subparts[0]} vs {subparts[1]}"
        else:
            return f"{parts[0]}:\n{second_part}"
    
    # For labels with parentheses
    if " (" in label:
        parts = label.split(" (", 1)
        return f"{parts[0]}\n({parts[1]}"
    
    # For labels with " vs " pattern
    if " vs " in label:
        parts = label.split(" vs ", 1)
        return f"{parts[0]}\nvs {parts[1]}"
    
    # Default: split at the middle if too long
    if len(label) > max_chars_per_line:
        mid = len(label) // 2
        # Try to find a space near the middle
        for i in range(mid - 5, mid + 5):
            if i < len(label) and label[i] == ' ':
                return f"{label[:i]}\n{label[i+1:]}"
    
    return label

def discover_model_result_dirs() -> dict:
    """Discover model result directories from *_results folders."""
    model_dirs = {}
    for result_dir in sorted(ANALYSIS_ROOT.glob("*_results")):
        if not result_dir.is_dir():
            continue
        stem = result_dir.name[:-len("_results")]
        model_dirs[stem] = result_dir
    return model_dirs


MODEL_RESULT_DIRS = discover_model_result_dirs()

# Define metrics of interest
main_metrics = ["Prompt Tokens", "Completion Tokens"]

# Define colors for each metric
metric_colors = {
    "Prompt Tokens": "#ACC2D9",      # light blue
    "Completion Tokens": "#464196",  # dark blue
}

# Cost level order (from lowest to highest cost)
cost_levels = ["MinCost", "LowerCost", "UpperCost", "MaxCost"]
cost_level_labels = ["Min Cost", "Lower Cost", "Upper Cost", "Max Cost"]
cost_level_numeric = [1, 2, 3, 4]  # For plotting

# Set larger font sizes globally
plt.rcParams.update({'font.size': 16})

print("Loading cost level accuracy charts from all models...")

# Load data from all models
all_data = []
for model_name, result_dir in MODEL_RESULT_DIRS.items():
    csv_path = Path(result_dir) / "cost_level_accuracy_chart.csv"
    if csv_path.exists():
        df_model = pd.read_csv(csv_path)
        df_model['model'] = model_name
        all_data.append(df_model)
        print(f"  Loaded data from {model_name}")
    else:
        print(f"  Warning: File not found for {model_name}: {csv_path}")

if not all_data:
    raise ValueError("No data files found!")

# Combine all models
df_all = pd.concat(all_data, ignore_index=True)
print(f"\nTotal models loaded: {len(df_all['model'].unique())}")

# Prepare data for regression analysis
print("\nPreparing data for regression analysis...")

def prepare_regression_data(df, metric_name):
    """Prepare data in long format for regression analysis."""
    metric_data = df[df['Metric'] == metric_name].copy()
    
    # Reshape to long format: each row is one model-cost_level combination
    long_data = []
    for _, row in metric_data.iterrows():
        model = row['model']
        for cost_level, cost_numeric in zip(cost_levels, cost_level_numeric):
            mean_col = f"{cost_level}_Mean"
            std_col = f"{cost_level}_Std"
            
            if mean_col in row.index and pd.notna(row[mean_col]):
                long_data.append({
                    'model': model,
                    'cost_level': cost_level,  # Keep as categorical string
                    'cost_level_numeric': cost_numeric,
                    'value': float(row[mean_col]),
                    'std': float(row[std_col]) if pd.notna(row.get(std_col)) else np.nan
                })
    
    return pd.DataFrame(long_data)

def aggregate_metrics(df, metric_name):
    """Aggregate a specific metric across all models (for display purposes)."""
    metric_data = df[df['Metric'] == metric_name].copy()
    
    aggregated = {}
    for cost_level in cost_levels:
        mean_col = f"{cost_level}_Mean"
        std_col = f"{cost_level}_Std"
        
        if mean_col in metric_data.columns:
            # Average the means across models
            aggregated[f"{cost_level}_mean"] = metric_data[mean_col].mean()
            # Pooled standard deviation: sqrt(mean of variances)
            aggregated[f"{cost_level}_std"] = np.sqrt((metric_data[std_col]**2).mean())
    
    return aggregated

def fit_regression_and_get_coefficients(long_df):
    """Fit regression with cost level as categorical variable and return coefficients, SEs, and p-values."""
    if len(long_df) == 0:
        return None, None, None
    
    # Create dummy variables for cost level (MinCost as reference)
    long_df_reg = long_df.copy()
    long_df_reg['cost_level'] = pd.Categorical(long_df_reg['cost_level'], 
                                                 categories=cost_levels, 
                                                 ordered=True)
    
    # Fit regression: value ~ C(cost_level) + model
    # This controls for model-specific effects
    try:
        formula = 'value ~ C(cost_level, Treatment(reference="MinCost")) + C(model)'
        model = ols(formula, data=long_df_reg).fit()
        
        # Extract coefficients for cost levels (excluding model effects)
        coef_dict = {}
        coef_se_dict = {}
        coef_pval_dict = {}
        
        for cost_level in cost_levels:
            if cost_level == "MinCost":
                # Reference level - coefficient is 0
                coef_dict[cost_level] = 0.0
                coef_se_dict[cost_level] = 0.0
                coef_pval_dict[cost_level] = 1.0  # Reference level has no p-value
            else:
                # Get coefficient for this cost level
                coef_name = f"C(cost_level, Treatment(reference=\"MinCost\"))[T.{cost_level}]"
                if coef_name in model.params.index:
                    coef_dict[cost_level] = model.params[coef_name]
                    coef_se_dict[cost_level] = model.bse[coef_name]
                    # Get p-value from the summary
                    pval = model.pvalues[coef_name]
                    coef_pval_dict[cost_level] = pval
                else:
                    coef_dict[cost_level] = np.nan
                    coef_se_dict[cost_level] = np.nan
                    coef_pval_dict[cost_level] = np.nan
        
        return coef_dict, coef_se_dict, coef_pval_dict
    except Exception as e:
        print(f"    Regression error: {e}")
        return None, None, None

def create_single_plot(metric_name, metric_key, plot_title, ylabel, output_path):
    """Create a single plot for a specific metric and metric key."""
    print(f"\nCreating plot: {plot_title}")
    print("-" * 80)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Prepare data for regression
    long_df = prepare_regression_data(df_all, metric_key)
    
    if len(long_df) == 0:
        print(f"  Warning: No data for {metric_key}")
        plt.close()
        return
    
    # Fit regression and get coefficients
    coef_dict, coef_se_dict, coef_pval_dict = fit_regression_and_get_coefficients(long_df)
    
    if coef_dict is None:
        print(f"  Warning: Could not fit regression for {metric_key}")
        plt.close()
        return
    
    # Extract coefficients in order
    coefs = []
    coef_ses = []
    x_vals = []
    for cost_level, cost_numeric in zip(cost_levels, cost_level_numeric):
        if cost_level in coef_dict and not np.isnan(coef_dict[cost_level]):
            coefs.append(coef_dict[cost_level])
            coef_ses.append(coef_se_dict[cost_level])
            x_vals.append(cost_numeric)
    
    if len(coefs) == 0:
        print(f"  Warning: No valid coefficients for {metric_key}")
        plt.close()
        return
    
    coefs = np.array(coefs)
    coef_ses = np.array(coef_ses)
    x_vals = np.array(x_vals)
    
    # Plot line chart with coefficients and error bars
    ax.errorbar(x_vals, coefs, yerr=coef_ses, marker='o', markersize=10, 
                linewidth=3, capsize=5, capthick=2, elinewidth=2,
                color=metric_colors[metric_name], alpha=0.8, zorder=3)
    
    # Add significance markers based on p-values
    # Use standard thresholds: * p<0.05, ** p<0.01, *** p<0.001
    # Place the marker slightly above the top of each error bar
    pvals = []
    for cost_level in cost_levels:
        pvals.append(coef_pval_dict.get(cost_level, np.nan))
    pvals = np.array(pvals)
    
    # We need p-values only for the cost levels that actually have coefficients
    # Align them with x_vals / coefs arrays
    aligned_pvals = []
    for cost_level in [cl for cl, _ in zip(cost_levels, cost_level_numeric)]:
        if cost_level in coef_dict and not np.isnan(coef_dict[cost_level]):
            aligned_pvals.append(coef_pval_dict.get(cost_level, np.nan))
    aligned_pvals = np.array(aligned_pvals)
    
    # Compute vertical offset for markers
    y_min = min(coefs - coef_ses)
    y_max = max(coefs + coef_ses)
    y_range = y_max - y_min if y_max > y_min else 1.0
    marker_offset = 0.05 * y_range
    
    for x, y, se, pval in zip(x_vals, coefs, coef_ses, aligned_pvals):
        if np.isnan(pval):
            continue
        if pval < 0.001:
            marker = "***"
        elif pval < 0.01:
            marker = "**"
        elif pval < 0.05:
            marker = "*"
        else:
            marker = ""
        if marker:
            ax.text(x, y + se + marker_offset, marker,
                    ha="center", va="bottom", fontsize=18,
                    color="black", fontweight="bold")
    
    # Format plot
    ax.set_xlabel("Cost Level", fontsize=18, fontweight='bold')
    ax.set_ylabel(wrap_label(ylabel), fontsize=18, fontweight='bold')
    ax.set_title(plot_title, fontsize=20, fontweight='bold')
    ax.set_xticks(cost_level_numeric)
    ax.set_xticklabels(cost_level_labels, fontsize=16)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3, zorder=1)
    
    # Scale y-axis to better show the lines and significance markers
    y_min = min(coefs - coef_ses)
    y_max = max(coefs + coef_ses)
    y_range = y_max - y_min
    if y_range > 0:
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.2 * y_range)
    
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Plot saved to: {output_path}")
    plt.close()
    
    # Print regression coefficients and p-values
    print(f"\n  Regression coefficients by cost level (relative to MinCost):")
    for cost_level, cost_numeric, coef, se, pval in zip(cost_levels, cost_level_numeric, 
                                                   [coef_dict.get(cl, np.nan) for cl in cost_levels],
                                                   [coef_se_dict.get(cl, np.nan) for cl in cost_levels],
                                                   [coef_pval_dict.get(cl, np.nan) for cl in cost_levels]):
        if not np.isnan(coef):
            pval_str = format_pvalue(pval) if not np.isnan(pval) else "N/A"
            print(f"    {cost_level:12s}: {coef:8.4f} ± {se:.4f} (p={pval_str})")

# ============================================================================
# PLOT 1: Prompt Tokens - Accuracy vs Cost Level
# ============================================================================

print("\n" + "="*80)
print("Creating Plot 1: Prompt Tokens - Accuracy vs Cost Level")
print("="*80)

create_single_plot(
    metric_name="Prompt Tokens",
    metric_key="Prompt Tokens",
    plot_title="Non-monotonic accuracy–cost pattern",
    ylabel="Accuracy Coefficient",
    output_path=OUTPUT_DIR / "multi_model_prompt_tokens_accuracy_plot.png"
)

# ============================================================================
# PLOT 2: Completion Tokens - Accuracy vs Cost Level
# ============================================================================

print("\n" + "="*80)
print("Creating Plot 2: Completion Tokens - Accuracy vs Cost Level")
print("="*80)

create_single_plot(
    metric_name="Completion Tokens",
    metric_key="Completion Tokens",
    plot_title="Non-monotonic accuracy–cost pattern",
    ylabel="Accuracy Coefficient",
    output_path=OUTPUT_DIR / "multi_model_completion_tokens_accuracy_plot.png"
)

# ============================================================================
# PLOT 3: Prompt Tokens - Repeated Modify vs Cost Level
# ============================================================================

print("\n" + "="*80)
print("Creating Plot 3: Prompt Tokens - Repeated Modify vs Cost Level")
print("="*80)

create_single_plot(
    metric_name="Prompt Tokens",
    metric_key="Prompt Tokens - Repeated Modify Same File",
    plot_title="Modify repetition increases at higher cost",
    ylabel="Repeated Modify Coefficient",
    output_path=OUTPUT_DIR / "multi_model_prompt_tokens_repeated_modify_plot.png"
)

# ============================================================================
# PLOT 4: Completion Tokens - Repeated Modify vs Cost Level
# ============================================================================

print("\n" + "="*80)
print("Creating Plot 4: Completion Tokens - Repeated Modify vs Cost Level")
print("="*80)

create_single_plot(
    metric_name="Completion Tokens",
    metric_key="Completion Tokens - Repeated Modify Same File",
    plot_title="Modify repetition increases at higher cost",
    ylabel="Repeated Modify Coefficient",
    output_path=OUTPUT_DIR / "multi_model_completion_tokens_repeated_modify_plot.png"
)

# ============================================================================
# PLOT 5: Prompt Tokens - Repeated View vs Cost Level
# ============================================================================

print("\n" + "="*80)
print("Creating Plot 5: Prompt Tokens - Repeated View vs Cost Level")
print("="*80)

create_single_plot(
    metric_name="Prompt Tokens",
    metric_key="Prompt Tokens - Repeated View Same File",
    plot_title="View repetition increases at higher cost",
    ylabel="Repeated View Coefficient",
    output_path=OUTPUT_DIR / "multi_model_prompt_tokens_repeated_view_plot.png"
)

# ============================================================================
# PLOT 6: Completion Tokens - Repeated View vs Cost Level
# ============================================================================

print("\n" + "="*80)
print("Creating Plot 6: Completion Tokens - Repeated View vs Cost Level")
print("="*80)

create_single_plot(
    metric_name="Completion Tokens",
    metric_key="Completion Tokens - Repeated View Same File",
    plot_title="View repetition increases at higher cost",
    ylabel="Repeated View Coefficient",
    output_path=OUTPUT_DIR / "multi_model_completion_tokens_repeated_view_plot.png"
)

# ============================================================================
# Print summary statistics
# ============================================================================

print("\n" + "="*80)
print("Summary Statistics")
print("="*80)

for metric_name in main_metrics:
    print(f"\n{metric_name}:")
    print("-" * 60)
    
    # Accuracy data
    agg_data = aggregate_metrics(df_all, metric_name)
    print("  Accuracy by Cost Level:")
    for cost_level, label in zip(cost_levels, cost_level_labels):
        mean_key = f"{cost_level}_mean"
        std_key = f"{cost_level}_std"
        if mean_key in agg_data:
            print(f"    {label:15s}: {agg_data[mean_key]:.4f} ± {agg_data[std_key]:.4f}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

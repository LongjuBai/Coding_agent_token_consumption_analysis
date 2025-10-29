#!/usr/bin/env python3
"""
Script to create bar plots comparing Zero-Shot with Zero-Shot + FG.
"""

import numpy as np
import matplotlib.pyplot as plt

def collect_data():
    """Organize the correlation data with confidence intervals."""
    
    data = {
        'exact_number': {
            'zeroshot': {
                'input': {'r': 0.1983, 'ci_lower': 0.0893, 'ci_upper': 0.3008},
                'output': {'r': 0.2217, 'ci_lower': 0.0669, 'ci_upper': 0.3661}
            },
            'fg': {
                'input': {'r': 0.2018, 'ci_lower': 0.1240, 'ci_upper': 0.2749},
                'output': {'r': 0.2286, 'ci_lower': 0.1050, 'ci_upper': 0.3436}
            }
        },
        'logscale': {
            'zeroshot': {
                'input': {'r': 0.2565, 'ci_lower': 0.1387, 'ci_upper': 0.3651},
                'output': {'r': 0.2624, 'ci_lower': 0.1251, 'ci_upper': 0.3889}
            },
            'fg': {
                'input': {'r': 0.2605, 'ci_lower': 0.1994, 'ci_upper': 0.3168},
                'output': {'r': 0.2669, 'ci_lower': 0.1816, 'ci_upper': 0.3453}
            }
        }
    }
    
    return data

def create_plots(data):
    """Create bar plots for correlation scores with confidence intervals."""
    
    # Setting labels
    setting_labels = ['Zero-Shot', 'Zero-Shot + FG']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for different predictor types
    colors = {'zeroshot': '#1f77b4', 'fg': '#ff7f0e'}
    
    # Plot for INPUT_TOKEN_ESTIMATE
    plot_correlation_bars(ax1, data, 'input', 'Input Token Estimate Correlation', setting_labels, colors, ylabel_flag=True)
    
    # Plot for OUTPUT_TOKEN_ESTIMATE
    plot_correlation_bars(ax2, data, 'output', 'Output Token Estimate Correlation', setting_labels, colors, ylabel_flag=False)
    
    plt.tight_layout()
    plt.savefig('./prediction/fg_comparison_plots.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_bars(ax, data, metric_key, title, setting_labels, colors, ylabel_flag):
    """Create bar plot for a specific metric with confidence intervals."""
    
    # Extract data values
    # Setting 1 (Zero-Shot): exact_number and logscale results
    exact_number_zeroshot = data['exact_number']['zeroshot'][metric_key]['r']
    logscale_zeroshot = data['logscale']['zeroshot'][metric_key]['r']
    
    # Setting 2 (Zero-Shot + FG): exact_number and logscale results
    exact_number_fg = data['exact_number']['fg'][metric_key]['r']
    logscale_fg = data['logscale']['fg'][metric_key]['r']
    
    # X positions for bars (reduced spacing between clusters)
    x_pos = np.arange(len(setting_labels)) * 0.6  # 0.6 spacing between clusters instead of 1.0
    width = 0.2  # Width of bars
    
    # Prepare data arrays: exact_number and logscale for both settings
    exact_values = [exact_number_zeroshot, exact_number_fg]
    logscale_values = [logscale_zeroshot, logscale_fg]
    
    # Calculate error bars (CI range) for exact_number
    # yerr format: [[lower_errors], [upper_errors]]
    exact_lower_err = [
        exact_number_zeroshot - data['exact_number']['zeroshot'][metric_key]['ci_lower'],
        exact_number_fg - data['exact_number']['fg'][metric_key]['ci_lower']
    ]
    exact_upper_err = [
        data['exact_number']['zeroshot'][metric_key]['ci_upper'] - exact_number_zeroshot,
        data['exact_number']['fg'][metric_key]['ci_upper'] - exact_number_fg
    ]
    exact_err = [exact_lower_err, exact_upper_err]
    
    # Calculate error bars for logscale
    logscale_lower_err = [
        logscale_zeroshot - data['logscale']['zeroshot'][metric_key]['ci_lower'],
        logscale_fg - data['logscale']['fg'][metric_key]['ci_lower']
    ]
    logscale_upper_err = [
        data['logscale']['zeroshot'][metric_key]['ci_upper'] - logscale_zeroshot,
        data['logscale']['fg'][metric_key]['ci_upper'] - logscale_fg
    ]
    logscale_err = [logscale_lower_err, logscale_upper_err]
    
    # Plot bars with error bars
    bars1 = ax.bar(x_pos - 0.5*width, exact_values, width, 
                   yerr=exact_err, capsize=3,
                   label='Exact Number', color=colors['zeroshot'], alpha=0.7)
    bars2 = ax.bar(x_pos + 0.5*width, logscale_values, width, 
                   yerr=logscale_err, capsize=3,
                   label='LogScale', color=colors['fg'], alpha=0.5)
    
    # Customize plot (increased font sizes)
    ax.set_xlabel('Settings', fontsize=22)
    if ylabel_flag:
        ax.set_ylabel('Correlation Score', fontsize=22)
    ax.set_title(title, fontsize=26)  # Removed bold
    ax.set_xticks(x_pos)
    ax.set_xticklabels(setting_labels, fontsize=19)
    ax.tick_params(axis='y', labelsize=19)
    ax.legend(fontsize=17)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=15)

if __name__ == "__main__":
    print("Organizing data...")
    data = collect_data()
    
    print("\nCreating plots...")
    create_plots(data)
    
    print("Plots saved as 'fg_comparison_plots.png'")

#!/usr/bin/env python3
"""
Script to create bar plots for correlation scores across different settings.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def collect_metrics_data():
    """Collect all metrics.json files and calculate averages."""
    
    # Base paths
    exact_number_base = "./prediction/llm_predictor_exact_number"
    logscale_base = "./prediction/llm_predictor_logScale"
    
    # Settings mapping
    settings = {
        'exact_number': {
            'zeroshot': {
                'problem_statement': 'ZeroShot_ProblemStatement',
                'problem_statement_toolreasoning': 'ZeroShot_ProblemStatement_ToolReasoning',
                'problem_statement_toolreasoning_difficulty': 'ZeroShot_ProblemStatement_ToolReasoning_Difficulty',
                'problem_statement_toolreasoning_difficulty_repoinfo': 'ZeroShot_ProblemStatement_ToolReasoning_Difficulty_RepoInfo'
            },
            'fewshot': {
                'problem_statement': 'FewShot_ProblemStatement',
                'problem_statement_toolreasoning': 'FewShot_ProblemStatement_ToolReasoning',
                'problem_statement_toolreasoning_difficulty': 'FewShot_ProblemStatement_ToolReasoning_Difficulty',
                'problem_statement_toolreasoning_difficulty_repoinfo': 'FewShot_ProblemStatement_ToolReasoning_Difficulty_RepoInfo'
            }
        },
        'logscale': {
            'zeroshot': {
                'problem_statement': 'ZeroShot_ProblemStatement',
                'problem_statement_toolreasoning': 'ZeroShot_ProblemStatement_ToolReasoning',
                'problem_statement_toolreasoning_difficulty': 'ZeroShot_ProblemStatement_ToolReasoning_Difficulty',
                'problem_statement_toolreasoning_difficulty_repoinfo': 'ZeroShot_ProblemStatement_ToolReasoning_Difficulty_RepoInfo'
            },
            'fewshot': {
                'problem_statement': 'FewShot_ProblemStatement',
                'problem_statement_toolreasoning': 'FewShot_ProblemStatement_ToolReasoning',
                'problem_statement_toolreasoning_difficulty': 'FewShot_ProblemStatement_ToolReasoning_Difficulty',
                'problem_statement_toolreasoning_difficulty_repoinfo': 'FewShot_ProblemStatement_ToolReasoning_Difficulty_RepoInfo'
            }
        }
    }
    
    # Data structure to store results
    data = {
        'exact_number': {
            'zeroshot': {'input_corr': [], 'output_corr': [], 'input_corr_std': [], 'output_corr_std': []},
            'fewshot': {'input_corr': [], 'output_corr': [], 'input_corr_std': [], 'output_corr_std': []}
        },
        'logscale': {
            'zeroshot': {'input_corr': [], 'output_corr': [], 'input_corr_std': [], 'output_corr_std': []},
            'fewshot': {'input_corr': [], 'output_corr': [], 'input_corr_std': [], 'output_corr_std': []}
        }
    }
    
    # Collect data for each setting
    for predictor_type in ['exact_number', 'logscale']:
        base_path = exact_number_base if predictor_type == 'exact_number' else logscale_base
        
        for shot_type in ['zeroshot', 'fewshot']:
            for setting_name, folder_name in settings[predictor_type][shot_type].items():
                correlations_input = []
                correlations_output = []
                
                # Check if this is a repoinfo setting (uses results_two_calls)
                if 'repoinfo' in setting_name:
                    result_dirs = ['results_two_calls1', 'results_two_calls2', 'results_two_calls3', 'results_two_calls4', 'results_two_calls5']
                else:
                    result_dirs = ['results1', 'results2', 'results3', 'results4', 'results5']
                
                for result_dir in result_dirs:
                    metrics_path = os.path.join(base_path, result_dir, folder_name, 'metrics.json')
                    
                    if os.path.exists(metrics_path):
                        try:
                            with open(metrics_path, 'r') as f:
                                metrics = json.load(f)
                                correlations_input.append(metrics['INPUT_TOKEN_ESTIMATE']['corr'])
                                correlations_output.append(metrics['OUTPUT_TOKEN_ESTIMATE']['corr'])
                        except Exception as e:
                            print(f"Error reading {metrics_path}: {e}")
                    else:
                        print(f"File not found: {metrics_path}")
                
                # Calculate average and standard deviation if we have data
                if correlations_input:
                    avg_input_corr = np.mean(correlations_input)
                    avg_output_corr = np.mean(correlations_output)
                    std_input_corr = np.std(correlations_input)
                    std_output_corr = np.std(correlations_output)
                    
                    data[predictor_type][shot_type]['input_corr'].append(avg_input_corr)
                    data[predictor_type][shot_type]['output_corr'].append(avg_output_corr)
                    data[predictor_type][shot_type]['input_corr_std'].append(std_input_corr)
                    data[predictor_type][shot_type]['output_corr_std'].append(std_output_corr)
                    
                    print(f"{predictor_type} - {shot_type} - {setting_name}: Input corr = {avg_input_corr:.4f} ± {std_input_corr:.4f}, Output corr = {avg_output_corr:.4f} ± {std_output_corr:.4f}")
                else:
                    print(f"No data found for {predictor_type} - {shot_type} - {setting_name}")
    
    return data

def create_plots(data):
    """Create bar plots for correlation scores."""
    
    # Setting labels (abbreviated)
    setting_labels = ['P', 'PT', 'PTD', 'PTDR']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for different shot types
    colors = {'zeroshot': '#1f77b4', 'fewshot': '#ff7f0e'}
    
    # Plot for INPUT_TOKEN_ESTIMATE
    plot_correlation_bars(ax1, data, 'input_corr', 'Input Token Estimate Correlation', setting_labels, colors, ylabel_flag=True)
    
    # Plot for OUTPUT_TOKEN_ESTIMATE
    plot_correlation_bars(ax2, data, 'output_corr', 'Output Token Estimate Correlation', setting_labels, colors, ylabel_flag=False)
    
    plt.tight_layout()
    plt.savefig('./prediction/correlation_plots.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_bars(ax, data, metric_key, title, setting_labels, colors, ylabel_flag):
    """Create bar plot for a specific metric."""
    
    # Data preparation
    exact_number_zeroshot = data['exact_number']['zeroshot'][metric_key]
    exact_number_fewshot = data['exact_number']['fewshot'][metric_key]
    logscale_zeroshot = data['logscale']['zeroshot'][metric_key]
    logscale_fewshot = data['logscale']['fewshot'][metric_key]
    
    # Error bar data preparation
    std_key = metric_key.replace('_corr', '_corr_std')
    exact_number_zeroshot_std = data['exact_number']['zeroshot'][std_key]
    exact_number_fewshot_std = data['exact_number']['fewshot'][std_key]
    logscale_zeroshot_std = data['logscale']['zeroshot'][std_key]
    logscale_fewshot_std = data['logscale']['fewshot'][std_key]
    
    # X positions for bars
    x_pos = np.arange(len(setting_labels))
    width = 0.2  # Width of bars
    
    # Plot bars with error bars
    bars1 = ax.bar(x_pos - 1.5*width, exact_number_zeroshot, width, 
                   yerr=exact_number_zeroshot_std, capsize=3,
                   label='Exact Number - ZeroShot', color=colors['zeroshot'], alpha=0.7)
    bars2 = ax.bar(x_pos - 0.5*width, exact_number_fewshot, width, 
                   yerr=exact_number_fewshot_std, capsize=3,
                   label='Exact Number - FewShot', color=colors['fewshot'], alpha=0.7)
    bars3 = ax.bar(x_pos + 0.5*width, logscale_zeroshot, width, 
                   yerr=logscale_zeroshot_std, capsize=3,
                   label='LogScale - ZeroShot', color=colors['zeroshot'], alpha=0.5)
    bars4 = ax.bar(x_pos + 1.5*width, logscale_fewshot, width, 
                   yerr=logscale_fewshot_std, capsize=3,
                   label='LogScale - FewShot', color=colors['fewshot'], alpha=0.5)
    
    # Customize plot (increased font sizes)
    ax.set_xlabel('Settings', fontsize=22)
    if ylabel_flag:
        ax.set_ylabel('Correlation Score', fontsize=22)
    ax.set_title(title, fontsize=26)  # Removed bold
    ax.set_xticks(x_pos)
    ax.set_xticklabels(setting_labels, fontsize=19)
    ax.tick_params(axis='y', labelsize=19)
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=11)

if __name__ == "__main__":
    print("Collecting metrics data...")
    data = collect_metrics_data()
    
    print("\nCreating plots...")
    create_plots(data)
    
    print("Plots saved as 'correlation_plots.png'")

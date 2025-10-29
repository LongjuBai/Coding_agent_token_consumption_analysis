#!/usr/bin/env python3
"""
Analysis script to create grouped bar chart showing accuracy by cost levels.
For each instance, ranks the 4 runs by each metric (prompt tokens, completion tokens, tool usages)
and assigns accuracy scores to MaxCost, UpperCost, LowerCost, MinCost levels.
"""

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_data(filepath):
    """Load the CSV data and extract relevant columns."""
    data = []
    with open(filepath, 'r') as f:
        # Increase field size limit to handle large fields
        csv.field_size_limit(1000000)
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            # Extract total tokens and accuracy for all 4 runs
            total_tokens = [int(row[15]), int(row[28]), int(row[41]), int(row[54])]  # total_total_tokens_run1-4
            accuracies = [int(row[65]), int(row[66]), int(row[67]), int(row[68])]
            
            data.append({
                'total_tokens': total_tokens,
                'accuracies': accuracies
            })
    
    return data

def rank_and_assign_accuracy(data):
    """
    For total tokens, rank the 4 runs and assign accuracy to cost levels.
    Returns: dict with structure {metric: {level: [accuracies]}}
    """
    results = {
        'total_tokens': {'MaxCost': [], 'UpperCost': [], 'LowerCost': [], 'MinCost': []}
    }
    
    for instance in data:
        # Rank the runs by total tokens (highest to lowest)
        values = instance['total_tokens']
        accuracies = instance['accuracies']
        
        # Create list of (value, accuracy, run_index) tuples
        ranked_data = [(values[i], accuracies[i], i) for i in range(4)]
        # Sort by value in descending order (highest cost first)
        ranked_data.sort(key=lambda x: x[0], reverse=True)
        
        # Assign accuracies to cost levels
        results['total_tokens']['MaxCost'].append(ranked_data[0][1])  # Highest cost
        results['total_tokens']['UpperCost'].append(ranked_data[1][1])  # Second highest
        results['total_tokens']['LowerCost'].append(ranked_data[2][1])  # Second lowest
        results['total_tokens']['MinCost'].append(ranked_data[3][1])   # Lowest cost
    
    return results

def calculate_average_accuracies(results):
    """Calculate average accuracy and standard deviation for each metric and cost level."""
    avg_results = {}
    std_results = {}
    
    for metric in results:
        avg_results[metric] = {}
        std_results[metric] = {}
        for level in results[metric]:
            accuracies = results[metric][level]
            avg_accuracy = sum(accuracies) / len(accuracies)
            # Calculate standard deviation
            variance = sum((x - avg_accuracy) ** 2 for x in accuracies) / len(accuracies)
            std_deviation = variance ** 0.5
            
            avg_results[metric][level] = avg_accuracy
            std_results[metric][level] = std_deviation
    
    return avg_results, std_results

def save_results_to_csv(avg_results, std_results, csv_path):
    """Save results to CSV file."""
    # Define the data structure for plotting
    metrics = ['total_tokens']
    levels = ['MaxCost', 'UpperCost', 'LowerCost', 'MinCost']
    
    # Prepare data for plotting
    data_matrix = []
    error_matrix = []
    for metric in metrics:
        metric_data = []
        metric_errors = []
        for level in levels:
            metric_data.append(avg_results[metric][level])
            metric_errors.append(std_results[metric][level])
        data_matrix.append(metric_data)
        error_matrix.append(metric_errors)
    
    # Save as CSV
    with open(csv_path, 'w') as f:
        f.write('Metric,MaxCost_Mean,MaxCost_Std,UpperCost_Mean,UpperCost_Std,LowerCost_Mean,LowerCost_Std,MinCost_Mean,MinCost_Std\n')
        for i, metric in enumerate(['Total Tokens']):
            f.write(f'{metric},{data_matrix[i][0]:.3f},{error_matrix[i][0]:.3f},{data_matrix[i][1]:.3f},{error_matrix[i][1]:.3f},{data_matrix[i][2]:.3f},{error_matrix[i][2]:.3f},{data_matrix[i][3]:.3f},{error_matrix[i][3]:.3f}\n')
    
    print(f"Data saved to: {csv_path}")
    return data_matrix, error_matrix

def load_results_from_csv(csv_path):
    """Load results from existing CSV file."""
    data_matrix = []
    error_matrix = []
    metrics = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        for row in reader:
            metric = row[0]
            metrics.append(metric)
            # Extract mean values (odd indices: 1, 3, 5, 7)
            mean_values = [float(row[i]) for i in range(1, 8, 2)]  # MaxCost_Mean, UpperCost_Mean, LowerCost_Mean, MinCost_Mean
            # Extract std values (even indices: 2, 4, 6, 8)
            std_values = [float(row[i]) for i in range(2, 9, 2)]   # MaxCost_Std, UpperCost_Std, LowerCost_Std, MinCost_Std
            data_matrix.append(mean_values)
            error_matrix.append(std_values)
    
    print(f"Data loaded from: {csv_path}")
    return data_matrix, error_matrix, metrics

def create_grouped_bar_chart(data_matrix, error_matrix, metrics, output_path):
    """Create the bar chart with specified colors and error bars."""
    levels = ['MinCost', 'LowerCost', 'UpperCost', 'MaxCost']  # Reordered from min to max
    
    # Reorder data_matrix and error_matrix to match the new level order
    # Original order: MaxCost, UpperCost, LowerCost, MinCost
    # New order: MinCost, LowerCost, UpperCost, MaxCost
    reordered_data_matrix = []
    reordered_error_matrix = []
    for metric_data, metric_errors in zip(data_matrix, error_matrix):
        # Reorder: [MaxCost, UpperCost, LowerCost, MinCost] -> [MinCost, LowerCost, UpperCost, MaxCost]
        reordered_metric_data = [metric_data[3], metric_data[2], metric_data[1], metric_data[0]]
        reordered_metric_errors = [metric_errors[3], metric_errors[2], metric_errors[1], metric_errors[0]]
        reordered_data_matrix.append(reordered_metric_data)
        reordered_error_matrix.append(reordered_metric_errors)
    
    # Set up the plot with reduced height
    fig, ax = plt.subplots(figsize=(12, 10))  # Reduced width since we only have one metric
    
    # Define color for total tokens
    color = '#464196'  # Dark purple
    
    # Set bar width and positions
    bar_width = 0.6  # Wider bars since we only have one metric
    x_positions = np.arange(len(levels))
    
    # Create bars without error bars
    metric_data = reordered_data_matrix[0]
    bars = ax.bar(x_positions, metric_data, bar_width, 
                 label=metrics[0], color=color, alpha=0.3)
    
    # Add value labels on top of bars
    for bar, value in zip(bars, metric_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Customize the plot
    ax.set_xlabel('Cost Levels', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by Cost Levels Across Four Runs', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.8)
    
    # Removed the red horizontal line at y=0.5
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return reordered_data_matrix

def print_summary_statistics(results, avg_results):
    """Print summary statistics."""
    print("=" * 60)
    print("COST LEVEL ACCURACY ANALYSIS SUMMARY")
    print("=" * 60)
    
    total_instances = len(results['total_tokens']['MaxCost'])
    print(f"Total instances analyzed: {total_instances}")
    print()
    
    for metric in ['total_tokens']:
        print(f"{metric.replace('_', ' ').title()}:")
        print("-" * 30)
        for level in ['MaxCost', 'UpperCost', 'LowerCost', 'MinCost']:
            accuracies = results[metric][level]
            avg_acc = avg_results[metric][level]
            success_count = sum(accuracies)
            print(f"  {level:10}: {avg_acc:.3f} ({success_count}/{total_instances})")
        print()

def main():
    """Main function to run the analysis."""
    # File paths
    input_file = './dataset/swe_bench_token_cost_aggregated_total_with_accuracy.csv'
    output_file = './analysis/cost_level_accuracy_chart.png'
    csv_file = './analysis/cost_level_accuracy_chart.csv'
    
    # Check if CSV file already exists
    if os.path.exists(csv_file):
        print("CSV file already exists. Loading data from CSV...")
        data_matrix, error_matrix, metrics = load_results_from_csv(csv_file)
    else:
        print("CSV file not found. Running full analysis...")
        print("Loading data...")
        data = load_data(input_file)
        print(f"Loaded {len(data)} instances")
        
        print("Ranking runs and assigning accuracies...")
        results = rank_and_assign_accuracy(data)
        
        print("Calculating average accuracies and standard deviations...")
        avg_results, std_results = calculate_average_accuracies(results)
        
        print("Saving results to CSV...")
        data_matrix, error_matrix = save_results_to_csv(avg_results, std_results, csv_file)
        
        print("Printing summary statistics...")
        print_summary_statistics(results, avg_results)
        
        # Verify totals
        print("\nVerification:")
        for metric in results:
            total = sum(len(results[metric][level]) for level in results[metric])
            print(f"{metric}: {total} total assignments (should be {len(data) * 4})")
        
        metrics = ['Total Tokens']
    
    print("Creating visualization...")
    create_grouped_bar_chart(data_matrix, error_matrix, metrics, output_file)
    
    print(f"Chart saved to: {output_file}")

if __name__ == "__main__":
    main()

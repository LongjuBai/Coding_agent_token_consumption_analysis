import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define colors
colors = {
    "prompt tokens": "#ACC2D9",
    "completion tokens": "#464196", 
    "tool usages": "#779ECB"
}

# Read the CSV file
df = pd.read_csv('./analysis/cost_level_accuracy_chart.csv')

# Extract data
metrics = df['Metric'].tolist()
# Sort cost levels from min to max cost
cost_levels = ['MinCost', 'LowerCost', 'UpperCost', 'MaxCost']

# Extract mean values for each metric and cost level
data = {}
for metric in metrics:
    data[metric] = []
    for cost_level in cost_levels:
        mean_col = f"{cost_level}_Mean"
        data[metric].append(df[df['Metric'] == metric][mean_col].iloc[0])

# Create figure with single subplot (square shape)
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

# Set font sizes
plt.rcParams.update({'font.size': 28})
ax.tick_params(axis='both', which='major', labelsize=24)

# Line Chart
for i, metric in enumerate(metrics):
    color = colors.get(metric.lower(), '#000000')  # Default to black if metric not found
    ax.plot(cost_levels, data[metric], marker='o', linewidth=4, label=metric, markersize=8, color=color)

# Customize line chart
ax.set_xlabel('Cost Level', fontsize=32)
ax.set_ylabel('Accuracy', fontsize=32)
ax.set_title('Accuracy Trends Across Cost Levels', fontsize=36)
ax.legend(fontsize=28)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.55, 0.63)  # Set y-axis range to better show differences

# Keep x-axis labels horizontal
plt.setp(ax.get_xticklabels(), rotation=0)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('./analysis/cost_level_accuracy_line_chart.png', 
            dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

print("Line chart created successfully!")
print(f"Saved as: ./analysis/cost_level_accuracy_line_chart_larger.png")

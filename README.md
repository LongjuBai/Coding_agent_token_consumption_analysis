# AI Agent Token Consumption Analysis

This repository contains code and analysis for understanding and predicting LLM token consumption patterns in AI coding agent workflows, specifically on the SWE-bench dataset.

## Overview

AI agents offer substantial opportunities to boost human productivity across many settings. However, their use in complex workflows also drives rapid growth in LLM token consumption. This work presents the first empirical analysis of agent token consumption patterns using agent trajectories on SWE-bench, and explores the possibility of predicting token costs at the beginning of task execution.

### Key Findings

1. **Inherent Randomness**: Agent token consumption has inherent randomness even when executing the same tasks; some runs use up to 10× more tokens than others.

2. **Accuracy and the Inverse Test-time Scaling Paradox**: More token usage does not lead to higher accuracy; tasks and runs that cost more tokens are usually associated with lower accuracy.

3. **Input Token Dominance**: Unlike chat and reasoning tasks, input tokens dominate overall consumption and cost, even with token caching.

4. **Prediction Feasibility**: While predicting total token consumption before execution is very challenging (Pearson's r < 0.15), predicting output-token amounts and the log scale of total consumption appears practical and reasonably accurate.

## Repository Structure

```
.
├── analysis/                    # Analysis scripts and notebooks
│   ├── cost_level_accuracy_analysis.py          # Analysis of accuracy by cost levels
│   ├── data_analysis.ipynb                      # Main exploratory data analysis
│   ├── difficulty_token_analysis.ipynb          # Token consumption vs task difficulty
│   ├── rounds_analysis_cache_tokens_plot.ipynb  # Analysis of token caching effects
│   └── token_cost_corr_analysis/               # Token and cost correlation analysis
│
├── dataset/                     # Dataset processing and preparation
│   ├── extract_llm_completions_with_stats.py    # Extract LLM completions and statistics
│   ├── get_accuracy.ipynb                       # Accuracy calculation
│   └── rounds_analysis.ipynb                    # Analysis of agent execution rounds
│
└── prediction/                  # Token consumption prediction models
    ├── llm_predictor_exact_number/              # Exact token number prediction
    │   └── llm_token_cost_pipeline_*.py
    ├── llm_predictor_logScale/                  # Logarithmic scale prediction
    │   └── llm_token_cost_pipeline_*.py
    ├── create_correlation_plots.py              # Generate correlation visualizations
    └── create_fg_comparison_plots.py            # Feature group comparison plots
```

## Key Insights

### Token Consumption Patterns

- **High Variability**: Token consumption varies significantly across runs of the same task
- **Input-Dominant**: Input tokens (prompt tokens) account for the majority of costs
- **Negative Correlation**: Higher token usage correlates with lower task accuracy

### Prediction Challenges

- Total token prediction is extremely difficult (r < 0.15)
- Output token prediction shows promise (more accurate)
- Predictions on log scale are more feasible than exact number predictions

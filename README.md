# AI Agent Token Consumption Analysis

This repository contains code and analysis for understanding and predicting LLM token consumption patterns in AI agent workflows, specifically on the SWE-bench dataset.

## Overview

AI agents offer substantial opportunities to boost human productivity across many settings. However, their use in complex workflows also drives rapid growth in LLM token consumption. This work presents the first empirical analysis of agent token consumption patterns using agent trajectories on SWE-bench, and explores the possibility of predicting token costs at the beginning of task execution.

### Key Findings

1. **Inherent Randomness**: Agent token consumption has inherent randomness even when executing the same tasks; some runs use up to 10× more tokens than others.

2. **Token Usage vs. Accuracy Trade-off**: More token usage does not lead to higher accuracy; tasks and runs that cost more tokens are usually associated with lower accuracy.

3. **Input Token Dominance**: Unlike chat and reasoning tasks, input tokens dominate overall consumption and cost, even with token caching.

4. **Prediction Feasibility**: While predicting total token consumption before execution is very challenging (Pearson's r < 0.15), predicting output-token amounts and the range of total consumption appears practical and reasonably accurate.

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

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required packages: pandas, numpy, matplotlib, scikit-learn

### Installation

```bash
# Clone the repository
git clone https://github.com/LongjuBai/Coding_agent_token_consumption_analysis.git
cd Coding_agent_token_consumption_analysis

# Install dependencies (if requirements.txt is available)
pip install -r requirements.txt
```

### Usage

#### 1. Data Analysis

Explore the main findings through the analysis notebooks:

```bash
# Open Jupyter notebook
jupyter notebook analysis/data_analysis.ipynb
```

Key analysis files:
- `analysis/cost_level_accuracy_analysis.py` - Analyze the relationship between token costs and accuracy
- `analysis/difficulty_token_analysis.ipynb` - Examine how task difficulty affects token consumption
- `analysis/rounds_analysis_cache_tokens_plot.ipynb` - Study the impact of token caching

#### 2. Token Prediction

Run token consumption prediction models:

```bash
# Exact number prediction
python prediction/llm_predictor_exact_number/llm_token_cost_pipeline_exact_number.py

# Logarithmic scale prediction
python prediction/llm_predictor_logScale/llm_token_cost_pipeline_logScale.py
```

#### 3. Generate Visualizations

Create correlation and comparison plots:

```bash
python prediction/create_correlation_plots.py
python prediction/create_fg_comparison_plots.py
```

## Key Insights

### Token Consumption Patterns

The analysis reveals several important patterns in agent token consumption:

- **High Variability**: Token consumption varies significantly across runs of the same task
- **Input-Dominant**: Input tokens (prompt tokens) account for the majority of costs
- **Negative Correlation**: Higher token usage correlates with lower task accuracy

### Prediction Challenges

Our experiments demonstrate that:
- Total token prediction is extremely difficult (r < 0.15)
- Output token prediction shows promise (more accurate)
- Range-based predictions are more feasible than exact number predictions

## Results

Key visualizations are included in the repository:
- Cost vs. accuracy analysis charts
- Difficulty button and bar plots
- Correlation plots comparing different prediction approaches
- Token caching effect visualizations

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Understanding and Predicting Agent Token Consumption},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Authors

- **Longju Bai** - [GitHub](https://github.com/LongjuBai)

## Acknowledgments

This work was conducted as part of research on AI agent efficiency and cost analysis. Special thanks to the SWE-bench dataset creators for providing the benchmarking environment.


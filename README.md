# Coding Agent Token Consumption Analysis

This repository contains data-processing and analysis scripts for studying LLM token/cost behavior in coding-agent trajectories (SWE-bench style runs), including multi-model comparisons, phase-level token decomposition, and self-prediction evaluation.

## TL;DR

AI coding agents can consume very large token budgets on long-horizon repository tasks.  
Using OpenHands on SWE-bench as a case study, this project analyzes token consumption (input + output) to answer three core questions: where tokens are spent, which models are more token-efficient, and whether token usage can be predicted before execution.

## Main Findings

1. Agentic coding is substantially more expensive than code reasoning and code chat settings.
2. Token usage is highly variable across models and even across repeated runs of the same task.
3. Higher token usage usually does not improve accuracy; higher-cost runs are often less accurate.
4. Human difficulty labels only weakly align with model cost: some “easy” tasks are expensive, and many “hard” tasks are solved cheaply.
5. Input tokens dominate overall consumption and cost, even with caching.
6. Phase-level dynamics shift over time: early phases are context-construction heavy, while later phases become more generation-heavy.
7. Self-prediction provides a weak-to-moderate signal (especially for output tokens), useful for rough budgeting but not precise forecasting.

## Scope

The current codebase is organized into:

- `dataset/`: build and enrich per-instance datasets from raw run artifacts.
- `analysis/`: multi-model analysis scripts (summary tables and figures).
- `analysis/phase_level_analysis/`: phase-based analysis from per-round interaction traces.
- `self_prediction_analysis/`: prediction-correlation computation and prompt templates.

This repo intentionally tracks scripts and templates, not generated CSV/PNG result artifacts.

## Repository Layout

```text
.
├── analysis/
│   ├── create_success_subset_csv.py
│   ├── model_token_usage_comparison.py
│   ├── multi_model_cost_level_accuracy_plots.py
│   ├── multi_model_difficulty_box_plots.py
│   ├── multi_model_group_accuracy_analysis.py
│   ├── multi_model_variance_plot.py
│   ├── unified_model_comparison.py
│   └── phase_level_analysis/
│       ├── token_corr_analysis.py
│       ├── token_cost_ratio_analysis.py
│       ├── token_count_analysis.py
│       └── token_ratio_analysis.py
├── dataset/
│   ├── add_tool_token_count_to_total.py
│   ├── extract_llm_completions_with_stats.py
│   ├── get_accuracy.py
│   ├── get_usage_average.py
│   ├── rounds_analysis.py
│   └── repo_stats_analysis/
│       ├── github_repo_analyzer_simple.py
│       └── repo_analyses/
│           ├── generate_file_trees.py
│           └── integrate_repo_stats.py
└── self_prediction_analysis/
    ├── calculate_correlations.py
    └── prompts/
        ├── in_context_learning_example.j2
        ├── swe_token_estimate.j2
        └── system_prompt.j2
```

## Python Dependencies

Install the common analysis stack:

```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels tqdm PyGithub
```

## Data Pipeline (Dataset Scripts)

1. `dataset/extract_llm_completions_with_stats.py`
   Extracts per-interaction message files and per-instance `summary.json` from one run’s `llm_completions`.
   Current script uses in-file variables (`input_root`, `run_name`), so update them before running.

2. `dataset/rounds_analysis.py`
   Unified round-level generator (replaces separate basic/with-cache versions).
   Produces `summary_rounds.json`, `summary_rounds_withCache.json`, or both.

3. `dataset/get_accuracy.py`
   Adds `acc_runX` columns by checking each run’s `report.json` `resolved_ids`.

4. `dataset/add_tool_token_count_to_total.py`
   Adds per-tool token-cost columns per run and row-wise averages/per-call averages.

5. `dataset/get_usage_average.py`
   Adds `total_prompt_tokens_mean`, `total_completion_tokens_mean`, and `total_tool_usages_mean`.

### Example Commands

```bash
python dataset/rounds_analysis.py --base ./sonet_openhands --run-ids 1,2,3,4 --mode both
python dataset/get_accuracy.py --input-csv dataset.csv --output-csv dataset_with_acc.csv
python dataset/add_tool_token_count_to_total.py --input-csv dataset_with_acc.csv --output-csv dataset_with_tool_avgs.csv
python dataset/get_usage_average.py --input dataset_with_acc.csv --output dataset_with_means.csv
```

## Optional Repo-Stats Augmentation

- `dataset/repo_stats_analysis/github_repo_analyzer_simple.py`: pull repository file/line stats via GitHub API to JSON.
- `dataset/repo_stats_analysis/repo_analyses/generate_file_trees.py`: render ASCII file trees from analysis JSON.
- `dataset/repo_stats_analysis/repo_analyses/integrate_repo_stats.py`: merge repository stats into a CSV keyed by repo name.

### Example Commands

```bash
python dataset/repo_stats_analysis/github_repo_analyzer_simple.py --repo owner/name --output repo_analysis.json
python dataset/repo_stats_analysis/repo_analyses/generate_file_trees.py --input-dir dataset/repo_stats_analysis/repo_analyses
python dataset/repo_stats_analysis/repo_analyses/integrate_repo_stats.py --input-csv input.csv --output-csv output.csv
```

## Multi-Model Analysis Scripts

- `analysis/create_success_subset_csv.py`: builds a success-subset table where every model has at least one successful run, averaging only successful runs.
- `analysis/multi_model_group_accuracy_analysis.py`: computes grouped cost/accuracy rows across models and exports mixed-effects summary tables.
- `analysis/unified_model_comparison.py`: creates one unified per-model summary table with all-instance means, success-subset means, and prediction metrics.
- `analysis/model_token_usage_comparison.py`: compares model token and dollar cost usage on all instances versus a shared success subset.
- `analysis/multi_model_variance_plot.py`: pools run-level metrics across models and visualizes per-instance variance plus model-level token/cost bars.
- `analysis/multi_model_difficulty_box_plots.py`: analyzes token and tool distributions across task difficulty levels with box plots and summary stats.
- `analysis/multi_model_cost_level_accuracy_plots.py`: fits cost-level regressions across models and plots coefficients for accuracy and repeated actions.

These scripts run standalone and write outputs under `analysis/`.

### Example Commands

```bash
python analysis/create_success_subset_csv.py --dataset-root dataset
python analysis/multi_model_group_accuracy_analysis.py --dataset-root dataset
python analysis/unified_model_comparison.py --dataset-root dataset --prediction-root self_prediction_analysis
python analysis/model_token_usage_comparison.py
python analysis/multi_model_variance_plot.py
```

## Phase-Level Analysis Scripts

Scripts in `analysis/phase_level_analysis/` compute token/cost behavior across five phases:
`early`, `early_mid`, `mid`, `later_mid`, `later`.

- `token_ratio_analysis.py`: token-type proportion by phase.
- `token_count_analysis.py`: absolute token counts by phase.
- `token_cost_ratio_analysis.py`: cost-component proportion by phase using pricing rules.
- `token_corr_analysis.py`: per-phase token-type vs cost correlations.

Important: these scripts use top-level config constants (`BASE`, `RUN_DIR_TMPL`, `EXTRACT_DIR_TMPL`) and expect `summary_rounds_withCache.json` inputs. Set these constants to your local extracted-run layout before execution.

## Self-Prediction Analysis

- `self_prediction_analysis/calculate_correlations.py`
  Computes correlation metrics from prediction CSVs (regular + `_log` variants) and writes `correlations_results.csv`.

- `self_prediction_analysis/prompts/*.j2`
  Prompt templates used for token-estimation style self-prediction workflows.

### Example Command

```bash
python self_prediction_analysis/calculate_correlations.py --root self_prediction_analysis --output self_prediction_analysis/correlations_results.csv
```

## Notes

- Most scripts default to relative paths and can be redirected via CLI arguments where supported.
- Several analysis scripts auto-discover model folders under `dataset/` by looking for expected CSV filenames.

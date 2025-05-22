# MOFGPT: Metal-Organic Framework Generation with Reinforcement Learning

## Overview

![MOFGPT Icon](https://github.com/srivathsanb14/MOFGPT/MOFGPT_icon.png)

MOFGPT is a comprehensive pipeline for generating novel Metal-Organic Frameworks (MOFs) with targeted properties using reinforcement learning. The framework enables fine-tuning of language models on MOF representations and subsequently optimizing them toward specific property targets through reinforcement learning.

This project provides a complete workflow for:
1. Fine-tuning generative models on MOF datasets
2. Generating new MOF candidates from fine-tuned models
3. Training reinforcement learning models to target specific properties
4. Analyzing and comparing the generated MOFs against training data
5. Visualizing distributions of MOF properties

## Project Structure

```
MOFGPT/
├── config/                 # Configuration files for training and generation
│   ├── config_finetune.yaml    # Fine-tuning configuration
│   └── rl/                     # RL-specific configurations
│       ├── config.yaml         # Main RL configuration
│       └── config_generation.yaml # Generation parameters
├── tokenizer/             # MOF tokenization
│   └── mof_tokenizer_gpt.py # Tokenizer for MOF representations
├── dataset/               # Dataset handling
│   └── dataset_gpt.py      # Dataset classes for MOF data
├── models/               # Core execution scripts
│   ├── train_finetune.py   # Fine-tuning script
│   ├── finetune_inference.py # Inference with fine-tuned model
│   ├── train_rl.py         # RL training script
│   ├── rl_inference.py     # Inference with RL model
│   ├── data_analysis.py    # Data analysis script
│   ├── compare_mof_results.py # Results comparison script
│   ├── run_full_pipeline.py # End-to-end pipeline execution
│   └── modules/                # Core functionality
│       ├── pipeline_metrics_logger.py # Performance logging
│       ├── model_utils.py      # Model handling utilities
│       ├── data_utils.py       # Data processing utilities
│       ├── reward.py           # RL reward functions
│       ├── generation.py       # MOF generation code
│       ├── experience_replay.py # Experience replay buffer for RL
│       └── models.py           # Model architectures
└── benchmark_datasets/    # Example datasets for training
    └── finetune/          # Fine-tuning datasets
        └── hMOF_CH4_*_small_mofid_finetune/ # Various MOF datasets
```

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/srivathsanb14/mofgpt.git
cd mofgpt
```

2. Create a conda environment and install dependencies:
```bash
conda create -n mofgpt python=3.8
conda activate mofgpt
pip install -r requirements.txt
```

## Usage

### End-to-End Pipeline

Run the full pipeline on multiple datasets:

```bash
python run_full_pipeline.py \
    --dataset-folders ./benchmark_datasets/finetune/hMOF_CH4_0.5_small_mofid_finetune ./benchmark_datasets/finetune/hMOF_CH4_0.05_small_mofid_finetune \
    --finetune-config ./config/config_finetune.yaml \
    --rl-config ./config/rl/config.yaml \
    --targets mean mean_plus_1std mean_plus_2std \
    --num-generations 100 \
    --continue-on-failure
```

### Data Analysis

Analyze MOF datasets to calculate statistics and set RL targets:

```bash
python data_analysis.py \
    --config config/rl/config.yaml \
    --property-col 1 \
    --property-name "CH4 adsorption" \
    --output-dir ./analysis \
    --target-types mean mean_plus_1std mean_plus_2std \
    --optimization-mode higher
```

### Fine-Tuning

Fine-tune a base pretrained language model on specific MOF data:

```bash
python train_finetune.py \
    --config_filename config/config_finetune.yaml
```

### Fine-Tuned Model Inference

Generate MOFs using the fine-tuned model:

```bash
python finetune_inference.py \
    --model ./models/mofgpt_model_best.pt \
    --config config/config_finetune.yaml \
    --generation-config config/rl/config_generation.yaml \
    --num-generations 100 \
    --batch-size 20 \
    --output-dir ./results
```

### RL Training

Train an RL model to generate MOFs with targeted properties:

```bash
python train_rl.py \
    --config config/rl/config.yaml \
    --output-dir ./models \
    --wandb-run-name "RL_mean_plus_1std" \
    --temperature 0.8 \
    --diversity-factor 0.1
```

### RL Model Inference

Generate MOFs using the trained RL model:

```bash
python rl_inference.py \
    --model ./models/mofgpt_RL_best.pt \
    --config config/rl/config.yaml \
    --generation-config config/rl/config_generation.yaml \
    --num-generations 100 \
    --batch-size 20 \
    --property-name "CH4 adsorption" \
    --output-dir ./results
```

### Compare Results

Compare the distributions of original and generated MOFs:

```bash
python compare_mof_results.py \
    --original-data ./benchmark_datasets/finetune/hMOF_CH4_0.5_small_mofid_finetune/train.csv \
    --finetune-data ./results/finetune/mofgpt_best_generations.csv \
    --rl-data ./results/rl_mean/mofgpt_RL_best_generations.csv ./results/rl_mean_plus_1std/mofgpt_RL_best_generations.csv \
    --rl-names "mean" "mean_plus_1std" \
    --property-col 1 \
    --property-name "CH4 adsorption at 0.5 bar" \
    --stats-json ./analysis/stats.json \
    --output-dir ./comparison_results \
    --include-no-finetune
```


## Reinforcement Learning Approach

The RL approach in this codebase uses:

1. **Pretrained or Fine-tuned LLM as Starting Point**: The RL model starts from a language model either just pretrained on MOF data or fine-tuned on specific MOF datasets based on configuration.
2. **Policy Gradient Method**: Optimizes policy to maximize reward.
3. **Multi-component Reward Function**: Combines rewards for:
   - **Novelty**: Encourages generating MOFs not in the training data
   - **Validity**: Ensures generated MOFs have valid chemical structures
   - **Diversity**: Promotes structural diversity among generated MOFs
   - **Target Achievement**: Rewards proximity to desired property targets

4. **Experience Replay**: Stores and reuses successful MOF generations to stabilize training.

## Analysis & Metrics

The codebase provides comprehensive analysis tools:

- **Property Distribution Analysis**: Visualizes how generated MOFs compare to original data
- **Statistical Comparisons**: Generates tables of key statistics for all datasets
- **Target Achievement**: Quantifies how well generated MOFs match target properties
- **Visualization Tools**: Creates normalized distributions, violin plots, ECDFs, and boxplots

This codebase represents significant work in applying reinforcement learning to the generation of MOFs with targeted properties.

## License

[Insert appropriate license information here]

## Citation

If you use this code, please cite:

```
[Citation information]
```

## Contact

[Your contact information]

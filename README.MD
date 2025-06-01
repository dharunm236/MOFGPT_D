# MOFGPT: Metal-Organic Framework Generation with Reinforcement Learning

## Overview


<table style="border: none;">
  <tr>
    <td style="vertical-align: middle; padding-right: 20px;">
      <p><strong>MOFGPT</strong> is a comprehensive pipeline for generating novel Metal-Organic Frameworks (MOFs) with targeted properties using reinforcement learning. The framework enables fine-tuning of language models on MOF representations and subsequently optimizing them toward specific property targets through reinforcement learning.</p>
    </td>
    <td style="vertical-align: middle;">
      <img src="https://github.com/srivathsanb14/MOFGPT/blob/main/MOFGPT_icon.png" width="300">
    </td>
  </tr>
</table>



This project provides a complete workflow for:
1. Fine-tuning generative models on MOF datasets
2. Generating new MOF candidates from fine-tuned models
3. Training reinforcement learning models to target specific properties
4. Analyzing and comparing the generated MOFs against training data
5. Visualizing distributions of MOF properties


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

3. Download the pretrained model from huggingface

```bash
# Create the saved_models directory
mkdir -p saved_models

# Download the model using git lfs or huggingface_hub
# Option 1: Using git (requires git-lfs)
cd saved_models
git clone https://huggingface.co/srivathsanb14/MOFGPT
cd ..

# Option 2: Using huggingface_hub (recommended)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='srivathsanb14/MOFGPT', local_dir='./saved_models/MOFGPT')
"
```

Note: Make sure to install huggingface_hub if using Option 2:

```bash
pip install huggingface_hub
```

The pre-trained MOFGPT model from [https://huggingface.co/srivathsanb14/MOFGPT](https://huggingface.co/srivathsanb14/MOFGPT) should be saved in the saved_models directory for use with the pipeline.

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



## Citation

If you use this code, please cite these two associated works:

```
@article{doi:10.1021/jacs.2c11420,
    author = {Cao, Zhonglin and Magar, Rishikesh and Wang, Yuyang and Barati Farimani, Amir},
    title = {MOFormer: Self-Supervised Transformer Model for Metal–Organic Framework Property Prediction},
    journal = {Journal of the American Chemical Society},
    volume = {145},
    number = {5},
    pages = {2958-2967},
    year = {2023},
    doi = {10.1021/jacs.2c11420},
    URL = {https://doi.org/10.1021/jacs.2c11420}
}
```

## Contact

Please contact srivathb@andrew.cmu.edu or barati@cmu.edu

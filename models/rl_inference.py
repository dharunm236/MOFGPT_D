#!/usr/bin/env python
# rl_inference.py - Improved version
"""
MOF-RL Inference Module

This module handles inference with trained MOF-RL models, generating
MOFs with target properties and visualizing results.
"""

import os
import sys
import torch
import yaml
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

sys.path.append("../") 
from modules.model_utils import (
    get_model, clear_memory, log_gpu_memory
)
from tokenizer.mof_tokenizer_gpt import MOFTokenizerGPT
from modules.models import LLMModel, LLMFineTuneModel
from modules.data_utils import (
    load_all_mofs, process_sequence_to_str, 
    verify_rdkit, check_for_existence
)
from modules.generation import generate
import random

def calculate_diversity(mof_strings):
    """Calculate diversity metrics for generated MOFs"""
    if len(mof_strings) <= 1:
        return {
            'unique_count': len(mof_strings),
            'diversity_ratio': 0.0,
            'unique_smiles': []
        }
    
    # Get unique SMILES
    unique_smiles = list(set(mof_strings))
    
    return {
        'unique_count': len(unique_smiles),
        'diversity_ratio': len(unique_smiles) / len(mof_strings),
        'unique_smiles': unique_smiles
    }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate and analyze MOFs using trained model")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to trained model")
    parser.add_argument("--config", type=str, default="../config/rl/config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--generation-config", type=str, default="../config/rl/config_generation.yaml", 
                        help="Path to generation configuration file")
    parser.add_argument("--output-dir", type=str, default="./results", 
                        help="Directory to save results")
    parser.add_argument("--output-filename", type=str, default=None, 
                        help="Explicit filename for output CSV") 
    parser.add_argument("--num-generations", type=int, default=100, 
                        help="Number of MOFs to generate")
    parser.add_argument("--batch-size", type=int, default=50, 
                        help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=None, 
                        help="Sampling temperature (overrides config)")
    parser.add_argument("--cuda-device", type=int, default=None, 
                        help="CUDA device ID to use (overrides config)")
    parser.add_argument("--property-name", type=str, default="Property", 
                        help="Name of the property for plotting")
    parser.add_argument("--model-type", type=str, choices=["rl", "finetune"], default="rl",
                        help="Type of model to use for inference (RL or fine-tune)")
    parser.add_argument("--target-value", type=float, default=None,
                        help="Target property value that the model was trained for (overrides config)")
    
    return parser.parse_args()


def load_model_and_configs(model_path, config_path, generation_config_path, device, model_type="rl"):
    """
    Load model and configurations with improved handling of model types
    """
    print(f"\nLoading configurations for {model_type.upper()} model...")
    
    # Load configuration files
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(config["model"]["model_config_filename"], "r") as f:
        model_config = yaml.safe_load(f)
    with open(generation_config_path, "r") as f:
        generation_config = yaml.safe_load(f)

    # Extract configuration sections
    training_config = config["training"]
    data_config = config["data"]
    base_model_config = model_config["base_model"]
    fine_tune_config = model_config["fine_tune"]
    
    # Initialize tokenizer
    tokenizer = MOFTokenizerGPT(
        vocab_file=data_config["vocab_path"],
        add_special_tokens=data_config["tokenizer"]["add_special_tokens"],
        pad_token=data_config["tokenizer"]["pad_token"],
        mask_token=data_config["tokenizer"]["mask_token"],
        bos_token=data_config["tokenizer"]["bos_token"],
        eos_token=data_config["tokenizer"]["eos_token"],
        unk_token=data_config["tokenizer"]["unk_token"],
        max_len=data_config["tokenizer"]["max_seq_len"],
        use_topology=data_config["tokenizer"]["use_topology"],
    )
    
    # Update model configuration with tokenizer information
    base_model_config["vocab_size"] = tokenizer.vocab_size
    base_model_config["pad_token_id"] = tokenizer.pad_token_id
    base_model_config["bos_token_id"] = tokenizer.bos_token_id
    base_model_config["eos_token_id"] = tokenizer.eos_token_id
    base_model_config["ignore_index"] = data_config["ignore_index"]
    base_model_config["max_position_embeddings"] = data_config["tokenizer"]["max_seq_len"]

    # Initialize base model
    print(f"Initializing base model for {model_type.upper()}...")
    clear_memory()
    base_model = get_model(config=base_model_config, device=device)
    
    # Load the model based on type
    print(f"Loading {model_type.upper()} model from: {model_path}")

    if model_type == "rl":
        print("\nTarget values configuration:")
        target_values = training_config.get("target_values", [0.0])
        print(f"Using target values: {target_values}")
        
        # Print model path to help debug
        print(f"Model path: {model_path}")
        
        # Try to extract target from model path
        target_type = "unknown"
        if "mean_plus_1std" in model_path:
            target_type = "mean+1std"
        elif "mean_plus_2std" in model_path:
            target_type = "mean+2std"
        elif "mean_plus_3std" in model_path:
            target_type = "mean+3std"
        elif "mean_minus" in model_path:
            target_type = "mean-std"
        elif "_mean" in model_path and not any(x in model_path for x in ["plus", "minus"]):
            target_type = "mean"
        
        print(f"Inferred target type from model path: {target_type}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if model_type == "finetune":
            # For fine-tune model, we need to wrap it in LLMFineTuneModel
            # First try loading the model state dict directly
            loaded = False
            
            # Try different possible keys for the state dict
            possible_keys = ["model_state_dict", "state_dict", "llm_network.state_dict"]
            
            for key in possible_keys:
                if key in checkpoint:
                    try:
                        # Try loading into the base model
                        base_model.load_state_dict(checkpoint[key])
                        print(f"Loaded base model weights from key: {key}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load base model with key {key}: {str(e)}")
            
            if not loaded:
                print("Could not load model weights directly. Trying to create fine-tune model wrapper...")
            
            # Create a fine-tune model wrapper
            model = LLMFineTuneModel(
                llm_network=base_model,
                llm_config=base_model_config,
                fine_tune_config=fine_tune_config,
                is_fp16=training_config['fp16']
            ).to(device)
            
            # Try to load the full model state dict if direct loading failed
            if not loaded:
                for key in possible_keys:
                    if key in checkpoint:
                        try:
                            model.load_state_dict(checkpoint[key])
                            print(f"Loaded full fine-tune model with key: {key}")
                            loaded = True
                            break
                        except Exception as e:
                            print(f"Failed to load fine-tune model with key {key}: {str(e)}")
            
            if not loaded:
                print("Warning: Could not load model weights properly. Results may be incorrect.")
        else:
            # For RL model, use standard loading approach
            loaded = False
            
            # Try different possible keys for the state dict
            possible_keys = ["model_state_dict", "state_dict", "llm_network.state_dict"]
            
            for key in possible_keys:
                if key in checkpoint:
                    try:
                        base_model.load_state_dict(checkpoint[key])
                        print(f"Loaded RL model weights from key: {key}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load RL model with key {key}: {str(e)}")
            
            # Print model info for debugging
            print(f"\nDEBUG - Loaded Model Info:")
            print(f"Model path: {model_path}")
            print(f"Model type: {model_type}")
            
            if 'rl_mean_plus_1std' in model_path:
                print("This appears to be a Mean+1std target RL model")
            elif 'rl_mean_plus_2std' in model_path:
                print("This appears to be a Mean+2std target RL model")
            elif 'rl_mean' in model_path:
                print("This appears to be a Mean target RL model")

            if not loaded:
                print("Warning: Could not load model weights properly. Results may be incorrect.")
            
            # Set the model to base_model (no wrapper needed for RL)
            model = base_model
        
        print(f"{model_type.upper()} model loaded successfully")
        
        # Print model info
        if "epoch" in checkpoint:
            epoch = checkpoint.get("epoch", "unknown")
            print(f"Model trained for {epoch} epochs")
        
        if model_type == "rl" and "max_val_reward" in checkpoint:
            reward = checkpoint.get("max_val_reward", "unknown")
            print(f"Best reward: {reward}")
        
        del checkpoint
        clear_memory()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, tokenizer, data_config, training_config, generation_config, None
    
    # Initialize energy predictor for RL model only
    energy_predictor = None
    if model_type == "rl" and training_config["do_energy_reward"]:
        print("\nInitializing energy predictor model...")
        try:
            import copy
            
            # Create a copy of the model config and modify the dimensions to match the checkpoint
            target_model_config = copy.deepcopy(base_model_config)
            target_model_config["hidden_size"] = 768          # Update to match checkpoint
            target_model_config["n_embd"] = 768              # Additional parameter that might be needed
            target_model_config["intermediate_size"] = 3072  # Update intermediate size (4x hidden_size)
            
            # Create a fresh model with the correct dimensions
            target_base_model = get_model(config=target_model_config, device=device)
            
            # Load the checkpoint
            target_model_saved_dict = torch.load(fine_tune_config["pretrained_model_path"], map_location=device)
            
            # Create the target predictor model
            energy_predictor = LLMFineTuneModel(
                llm_network=target_base_model,  # Use the correctly sized model
                llm_config=target_model_config, # Use the modified config
                fine_tune_config=fine_tune_config,
                is_fp16=training_config['fp16']
            ).to(device)
            
            # Try to load the energy predictor state dict
            loaded = False
            possible_keys = ["model_state_dict", "state_dict", "llm_network.state_dict"]
            
            for key in possible_keys:
                if key in target_model_saved_dict:
                    try:
                        energy_predictor.load_state_dict(target_model_saved_dict[key])
                        print(f"Loaded energy predictor model with key: {key}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load energy predictor with key {key}: {str(e)}")
            
            if not loaded:
                print("Warning: Could not load energy predictor properly. Target predictions may be incorrect.")
                
            del target_model_saved_dict, target_base_model
            clear_memory()
        except Exception as e:
            print(f"Error loading energy predictor: {str(e)}")
            print("Continuing without energy prediction")
            training_config["do_energy_reward"] = False
    
    return model, tokenizer, data_config, training_config, generation_config, energy_predictor


def generate_mofs(model,
                 tokenizer,
                 generation_config,
                 training_config,
                 device,
                 metals_list,
                 all_mof_strs,
                 topology_labels_key,
                 energy_predictor,
                 output_filename,
                 num_generations=100,
                 batch_size=50,
                 property_name="Property",
                 model_type="rl"):
    """
    Generate MOF structures using trained model and save to CSV.
    Continues generating until exactly num_generations valid and novel MOFs are produced.
    """
    print(f"\n{'='*50}")
    print(f"Generating {num_generations} valid and novel MOF structures using {model_type.upper()} model")
    print(f"{'='*50}")
    
    ## ADDED: Print target values being used for generation
    target_values = training_config.get('target_values', [0.0])
    print(f"Using target values for generation: {target_values}")

    # Set model to evaluation mode
    if hasattr(model, 'eval'):
        model.eval()
    
    # Use a moderate temperature for generation
    original_temperature = generation_config["temperature"]
    generation_config["temperature"] = max(0.8, original_temperature)
    
    # Generate all MOFs in one or more batches
    all_mofs = []
    all_targets = []
    
    # Track metrics for reporting
    total_generated = 0
    valid_count = 0
    novel_count = 0
    
    # Keep track of unique MOFs
    unique_mofs = set()
    
    # Initialize metrics tracking
    metrics = {
        'total_attempts': 0,
        'valid_count': 0,
        'novel_count': 0,
        'unique_mofs': set(),
        'all_generated': []
    }
    
    print("Generating MOFs...")
    
    batch_idx = 0
    with tqdm(total=num_generations, desc="Generating valid & novel MOFs") as pbar:
        while len(all_mofs) < num_generations:
            batch_idx += 1
            current_batch_size = batch_size
                
            print(f"Generating batch {batch_idx} (size: {current_batch_size}). Current progress: {len(all_mofs)}/{num_generations} valid & novel MOFs")
            
            # Handle different generation approaches based on model type
            if model_type == "rl":
                # For RL model, we use the standard generation function
                generated_sequences, _, generated_targets = generate(
                    model=model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    training_config=training_config,
                    device=device,
                    num_return_sequences=current_batch_size,
                    energy_predictor=energy_predictor
                )
            else:
                # For fine-tune model, we need a different approach
                # First generate sequences (without property prediction)
                generated_sequences, _, _ = generate(
                    model=model.llm_network if hasattr(model, 'llm_network') else model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    training_config=training_config,
                    device=device,
                    num_return_sequences=current_batch_size,
                    energy_predictor=None  # No energy predictor for generation
                )
                generated_targets = []
            
            # Convert to MOF strings
            generated_mof_strs = []
            for sequence in generated_sequences:
                sequence_str = process_sequence_to_str(
                    sequence=sequence,
                    tokenizer=tokenizer,
                    training_config=training_config,
                    generation_config=generation_config
                )
                generated_mof_strs.append(sequence_str)
            
            # Predict properties if using fine-tune model
            if model_type == "finetune":
                print("Predicting properties using fine-tune model...")
                # Use the fine-tune model to predict properties
                with torch.no_grad():
                    for mof_str in generated_mof_strs:
                        # Tokenize the MOF string
                        token_ids = tokenizer.encode(mof_str)
                        token_tensor = torch.tensor([token_ids]).to(device)
                        
                        # Create mask
                        mask = torch.ones_like(token_tensor).to(device)
                        
                        # Get prediction directly from the model
                        try:
                            # Check if the model has specific prediction method or use forward
                            if hasattr(model, 'predict'):
                                output = model.predict(token_ids=token_tensor, mask_ids=mask)
                            else:
                                output = model(token_ids=token_tensor, mask_ids=mask)
                            
                            # Extract prediction value
                            if isinstance(output, torch.Tensor):
                                prediction = output.item()
                            else:
                                prediction = output
                                
                            generated_targets.append(prediction)
                            print(f"Predicted property: {prediction:.4f}")
                        except Exception as e:
                            print(f"Error during property prediction: {str(e)}")
                            generated_targets.append(0.0)  # Default value on error
            
            # Check validity
            valid_mofs_list, valid_bool = verify_rdkit(
                curr_mofs=generated_mof_strs,
                metal_atom_list=metals_list,
                generation_config=generation_config,
                training_config=training_config,
                topology_labels_key=topology_labels_key
            )
            
            # Process results
            for i, mof_str in enumerate(generated_mof_strs):
                total_generated += 1
                
                # Add to metrics
                metrics['total_attempts'] += 1
                metrics['all_generated'].append(mof_str)
                
                # Check if valid
                is_valid = valid_bool[i] if i < len(valid_bool) else False
                if is_valid:
                    valid_count += 1
                    metrics['valid_count'] += 1
                    
                    # Check if novel compared to dataset
                    # is_novel = not check_for_existence(mof_str, all_mof_strs)
                    is_novel = not check_for_existence([mof_str], all_mof_strs)[1][0]
                    if is_novel:
                        novel_count += 1
                        metrics['novel_count'] += 1
                    
                    # Add to our collection if not a duplicate
                    if is_valid and is_novel and mof_str not in unique_mofs:
                        unique_mofs.add(mof_str)
                        metrics['unique_mofs'].add(mof_str)
                        all_mofs.append(mof_str)
                        
                        # Get corresponding target
                        if generated_targets and i < len(generated_targets):
                            all_targets.append(generated_targets[i])
                        else:
                            # Default if no prediction
                            if training_config.get("num_targets", 1) > 1:
                                all_targets.append([0.0] * training_config.get("num_targets", 1))
                            else:
                                all_targets.append(0.0)
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'valid': f"{valid_count}/{total_generated} ({valid_count/max(1, total_generated)*100:.1f}%)",
                            'novel': f"{novel_count}/{valid_count} ({novel_count/max(1, valid_count)*100:.1f}%)"
                        })
                
                # If we have enough, break the inner loop
                if len(all_mofs) >= num_generations:
                    break
            
            # Free memory
            del generated_sequences, generated_mof_strs, generated_targets
            del valid_mofs_list, valid_bool
            clear_memory()
            
            # If we have enough, break the outer loop
            if len(all_mofs) >= num_generations:
                break
    
    # Restore original temperature
    generation_config["temperature"] = original_temperature
    
    # Ensure we have exactly num_generations MOFs by truncating if needed
    all_mofs = all_mofs[:num_generations]
    all_targets = all_targets[:num_generations]
    
    # Create DataFrame for saving
    results_data = []
    
    # Fix for Issue #1: Ensure target values are properly saved to CSV
    for i, (mof, target_val) in enumerate(zip(all_mofs, all_targets)):
        # Create a row dict
        row = {"id": i, "mof": mof}
        
        # Add target predictions
        if isinstance(target_val, (list, tuple)):
            for j, t_val in enumerate(target_val):
                row[f"target_{j+1}"] = float(t_val)  # Ensure it's a float
        else:
            row["target_1"] = float(target_val)  # Ensure it's a float
        
        results_data.append(row)
    
    # Print a sample of the data before saving
    print("\nSample of data to be saved:")
    for i in range(min(3, len(results_data))):
        print(results_data[i])
    
    # Save to CSV
    output_df = pd.DataFrame(results_data)
    output_df.to_csv(output_filename, index=False)
    
    # Verify the CSV was written correctly
    try:
        test_read = pd.read_csv(output_filename)
        print(f"\nSuccessfully verified CSV file. Shape: {test_read.shape}")
        
        # Check the first few rows to make sure targets were saved correctly
        print("\nFirst few rows of saved CSV:")
        print(test_read.head(3))
    except Exception as e:
        print(f"Error verifying CSV file: {str(e)}")
    
    print(f"\nSaved {len(all_mofs)} MOF structures to {output_filename}")
    print(f"Generated {total_generated} total MOFs to get {len(all_mofs)} valid and novel structures")
    print(f"Validity rate: {valid_count/max(1, total_generated)*100:.1f}%")
    print(f"Novelty rate: {novel_count/max(1, valid_count)*100:.1f}%")
    print(f"Overall efficiency: {len(all_mofs)/max(1, total_generated)*100:.1f}%")
    
    # Print model type in the statistics
    print(f"Model type: {model_type.upper()}")
    
    # Sample of generated MOFs
    print("\nSample of generated MOFs:")
    for i in range(min(5, len(all_mofs))):
        target_str = ""
        if all_targets and i < len(all_targets):
            if isinstance(all_targets[i], (list, tuple)):
                target_str = f" | Targets: {[round(float(t), 4) for t in all_targets[i]]}"
            else:
                target_str = f" | Target: {round(float(all_targets[i]), 4)}"
        print(f"MOF {i+1}: {all_mofs[i]}{target_str}")
    
    # Calculate diversity from metrics
    diversity_stats = calculate_diversity(metrics['all_generated'])
    
    # Final stats dictionary
    stats_dict = {
        'total_attempts': metrics['total_attempts'],
        'valid_count': metrics['valid_count'],
        'validity_rate': metrics['valid_count'] / max(1, metrics['total_attempts']),
        'novel_count': metrics['novel_count'],
        'novelty_rate': metrics['novel_count'] / max(1, metrics['valid_count']),
        'unique_count': len(metrics['unique_mofs']),
        'diversity_ratio': diversity_stats['diversity_ratio'] if 'diversity_ratio' in diversity_stats else 0.0,
        'efficiency': len(metrics['unique_mofs']) / max(1, metrics['total_attempts'])
    }
    
    # Save statistics to CSV
    stats_path = save_generation_stats(stats_dict, os.path.dirname(output_filename))
    print(f"Saved statistics to {stats_path}")
    
    return all_mofs, all_targets, output_df

def save_generation_stats(stats_dict, output_dir, filename="generation_stats.csv"):
    """Save generation statistics to a CSV file"""
    stats_df = pd.DataFrame([stats_dict])
    stats_path = os.path.join(output_dir, filename)
    stats_df.to_csv(stats_path, index=False)
    print(f"Generation statistics saved to: {stats_path}")
    return stats_path


def debug_model_targets(model_path, generated_values, output_dir, target_value=None):
    """Debug function to print model target info and generated distribution stats"""
    # Extract target from model path
    target_type = "unknown"
    if "mean_plus_1std" in model_path:
        target_type = "mean+1std"
    elif "mean_plus_2std" in model_path:
        target_type = "mean+2std"
    elif "mean_plus_3std" in model_path:
        target_type = "mean+3std"
    elif "mean_minus" in model_path:
        target_type = "mean-std"
    elif "_mean" in model_path and not any(x in model_path for x in ["plus", "minus"]):
        target_type = "mean"
    
    # Calculate statistics of generated values
    mean_val = np.mean(generated_values)
    median_val = np.median(generated_values)
    std_val = np.std(generated_values)
    min_val = np.min(generated_values)
    max_val = np.max(generated_values)
    
    # Print debug info
    print("\n===== MODEL TARGET DEBUG INFO =====")
    print(f"Model path: {model_path}")
    print(f"Inferred target type: {target_type}")
    if target_value is not None:
        print(f"Actual target value used: {target_value:.6f}")
    print(f"Generated distribution stats:")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Median: {median_val:.6f}")
    print(f"  Std Dev: {std_val:.6f}")
    print(f"  Range: {min_val:.6f} to {max_val:.6f}")
    
    # Save to a debug file
    debug_info = {
        "model_path": model_path,
        "target_type": target_type,
        "actual_target_value": target_value,
        "generated_mean": mean_val,
        "generated_median": median_val,
        "generated_std": std_val,
        "generated_min": min_val,
        "generated_max": max_val
    }
    
    debug_path = os.path.join(output_dir, "target_debug_info.json")
    with open(debug_path, 'w') as f:
        json.dump(debug_info, f, indent=4)
    
    print(f"Debug info saved to: {debug_path}")
    print("=====================================")


def analyze_data_distribution(data_df, property_col, property_name, target_values=None, output_dir=None):
    """
    Analyze and plot the distribution of property values
    
    Args:
        data_df: DataFrame with property values
        property_col: Column name for the property
        property_name: Display name for the property
        target_values: List of target values to highlight
        output_dir: Directory to save plots (if None, displays instead)
        
    Returns:
        stats: Dictionary with statistics
    """
    print(f"\nAnalyzing distribution of {property_name}...")
    
    # Extract property values
    values = data_df[property_col].dropna()
    
    # Calculate statistics
    stats = {
        "count": len(values),
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "q1": values.quantile(0.25),
        "q3": values.quantile(0.75),
    }
    
    # Print statistics
    print(f"\n----- {property_name} Statistics -----")
    print(f"Count: {stats['count']}")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"Standard Deviation: {stats['std']:.4f}")
    print(f"Range: {stats['min']:.4f} to {stats['max']:.4f}")
    print(f"Interquartile Range: {stats['q1']:.4f} to {stats['q3']:.4f}")
    
    # Create histogram plot
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with KDE
    sns.histplot(values, kde=True, color='skyblue', edgecolor='black')
    
    # Add vertical lines for key statistics
    plt.axvline(stats["mean"], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {stats["mean"]:.4f}')
    plt.axvline(stats["median"], color='green', linestyle='dashed', linewidth=2, label=f'Median: {stats["median"]:.4f}')
    
    # Add target value lines if provided
    if target_values:
        for i, target in enumerate(target_values):
            color = ['purple', 'brown', 'orange', 'magenta', 'cyan'][i % 5]
            plt.axvline(target, color=color, linestyle=':', linewidth=2, label=f'Target {i+1}: {target:.4f}')
    
    # Set labels and title
    plt.xlabel(property_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of {property_name} Values', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    
    # Add text box with statistics
    stats_text = '\n'.join([
        f"Count: {stats['count']}",
        f"Mean: {stats['mean']:.4f}",
        f"Std Dev: {stats['std']:.4f}",
        f"Min: {stats['min']:.4f}",
        f"Max: {stats['max']:.4f}"
    ])
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                 ha='left', va='top', fontsize=10)
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{property_name.lower().replace(' ', '_')}_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return stats


def plot_target_achievement(generated_df, target_values, property_cols, property_names, output_dir=None):
    """
    Plot how well the generated MOFs achieve the target values
    
    Args:
        generated_df: DataFrame with generated MOFs
        target_values: List of target values
        property_cols: List of column names for the properties
        property_names: List of display names for the properties
        output_dir: Directory to save plots (if None, displays instead)
    """
    print("\nAnalyzing target achievement...")
    
    # Number of targets
    num_targets = len(target_values)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Track achievement rates
    achievement_rates = []
    
    # Plot for each target
    for i in range(num_targets):
        target = target_values[i]
        property_col = property_cols[i]
        property_name = property_names[i]
        
        # Extract values
        values = generated_df[property_col].dropna()
        
        # Calculate statistics
        mean_val = values.mean()
        median_val = values.median()
        std_val = values.std()
        
        # Calculate target achievement
        if mean_val <= target:
            # Target is higher than mean
            achievement = mean_val / target * 100
            achievement_msg = f"Mean is {100 - achievement:.1f}% below target"
        else:
            # Mean is higher than target
            achievement = target / mean_val * 100  
            achievement_msg = f"Mean is {100 - achievement:.1f}% above target"
        
        achievement_rates.append(achievement)
        
        # Calculate percentage within X% of target
        within_5pct = sum(abs(v - target) / target <= 0.05 for v in values) / len(values) * 100
        within_10pct = sum(abs(v - target) / target <= 0.10 for v in values) / len(values) * 100
        within_20pct = sum(abs(v - target) / target <= 0.20 for v in values) / len(values) * 100
        
        # Create subplot
        plt.subplot(num_targets, 1, i+1)
        
        # Plot histogram
        sns.histplot(values, kde=True, color='skyblue', edgecolor='black')
        
        # Add vertical lines
        plt.axvline(target, color='red', linestyle='dashed', linewidth=2, label=f'Target: {target:.4f}')
        plt.axvline(mean_val, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.4f}')
        
        # Add shaded region for "close to target"
        plt.axvspan(target * 0.95, target * 1.05, alpha=0.2, color='green')
        
        # Set labels
        plt.xlabel(property_name, fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title(f'Target Achievement for {property_name}', fontsize=12)
        plt.legend(loc='best', fontsize=9)
        
        # Add statistics text
        stats_text = '\n'.join([
            f"Mean: {mean_val:.4f}",
            f"Target: {target:.4f}",
            achievement_msg,
            f"Within 5% of target: {within_5pct:.1f}%",
            f"Within 10% of target: {within_10pct:.1f}%",
            f"Within 20% of target: {within_20pct:.1f}%"
        ])
        
        plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                     ha='left', va='top', fontsize=9)
    
    plt.tight_layout()
    
    # Save or display the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "target_achievement.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Target achievement plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print overall target achievement summary
    print("\nTarget Achievement Summary:")
    for i in range(num_targets):
        print(f"Target {i+1} ({property_names[i]}): {achievement_rates[i]:.1f}% achievement")


def main():
    """Main function for MOF-RL inference"""
    # Parse arguments
    args = parse_arguments()

    # Set seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Print banner
    print("\n" + "="*70)
    print("MOF-RL INFERENCE AND ANALYSIS".center(70))
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Model type: {args.model_type}")
    print(f"Configuration: {args.config}")
    print(f"Generation config: {args.generation_config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of generations: {args.num_generations}")
    print(f"Property name: {args.property_name}")
    print("="*70 + "\n")



    # Set device for inference
    if torch.cuda.is_available():
        if args.cuda_device is not None:
            if args.cuda_device >= torch.cuda.device_count():
                print(f"Warning: Specified CUDA device {args.cuda_device} not available. Using device 0 instead.")
                device = torch.device('cuda:0')
            else:
                device = torch.device(f'cuda:{args.cuda_device}')
            print(f"Using CUDA device {args.cuda_device}: {torch.cuda.get_device_name(args.cuda_device)}")
        else:
            device = torch.device('cuda:0')
            print(f"Using CUDA device 0: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU for inference")
    
    # Load model and configurations
    model, tokenizer, data_config, training_config, generation_config, energy_predictor = load_model_and_configs(
        model_path=args.model,
        config_path=args.config,
        generation_config_path=args.generation_config,
        device=device,
        model_type=args.model_type
    )
    
    if model is None:
        print("Error loading model, exiting.")
        return 1
    
    # Override temperature if specified
    if args.temperature is not None:
        generation_config["temperature"] = args.temperature
        print(f"Overriding temperature: {args.temperature}")
    
    # Load topology labels and metals list
    topology_labels_filename = data_config["topology_labels_map_filename"]
    with open(topology_labels_filename, "r") as f:
        topology_labels_map = json.load(f)
    topology_labels_key = list(topology_labels_map.keys())

    # Load metals list
    with open(generation_config["metals_list_file"], "r") as f:
        metals_list = [line.strip() for line in f.readlines()]
    
    # Load training MOF datasets
    print("\nLoading MOF datasets for novelty check...")
    all_mof_strs_train, all_mof_strs_val, all_mof_strs_test, all_mof_strs = load_all_mofs(
        data_config=data_config,
        training_config=training_config
    )
    
    # Get number of targets and target values
    num_targets = training_config.get("num_targets", 1)
    target_values = training_config.get("target_values", [0.0] * num_targets)
    


    # if args.target_value is not None:
    #     training_config["target_values"] = [float(args.target_value)]
    #     print(f"Using target value from command line: {args.target_value}")

    if args.target_value is not None:
        original_target = training_config.get("target_values", [0.0])[0]
        training_config["target_values"] = [float(args.target_value)]
        print(f"Overriding target value: {original_target} -> {args.target_value}")
    
    # Determine output CSV filename
    if args.output_filename:
        output_csv = args.output_filename
    else:
        # Generate a filename based on model type and path
        model_basename = os.path.basename(args.model).replace(".pt", "")
        output_csv = os.path.join(args.output_dir, f"{model_basename}_generations.csv")
    
    # Generate MOFs and save to CSV
    _, _, generated_df = generate_mofs(
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        training_config=training_config,
        device=device,
        metals_list=metals_list,
        all_mof_strs=all_mof_strs,
        topology_labels_key=topology_labels_key,
        energy_predictor=energy_predictor,
        output_filename=output_csv,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        property_name=args.property_name,
        model_type=args.model_type
    )
    
    # Determine property columns and names for visualization
    if num_targets == 1:
        property_cols = ["target_1"]
        property_names = [args.property_name]
    else:
        property_cols = [f"target_{i+1}" for i in range(num_targets)]
        property_names = [f"{args.property_name} {i+1}" for i in range(num_targets)]
    
    # Update property names to include model type
    property_names = [f"{name} ({args.model_type.upper()})" for name in property_names]

    if generated_df is not None and "target_1" in generated_df.columns:
        actual_property_values = generated_df["target_1"].values
        
        # Now debug with the actual generated values and target value
        target_val = training_config.get("target_values", [0.0])[0]
        debug_model_targets(model_path=args.model, 
                        generated_values=actual_property_values,
                        output_dir=args.output_dir,
                        target_value=target_val)


    # # Analyze distributions
    # for i in range(num_targets):
    #     analyze_data_distribution(
    #         data_df=generated_df,
    #         property_col=property_cols[i],
    #         property_name=property_names[i],
    #         target_values=[target_values[i]],
    #         output_dir=args.output_dir
    #     )
    
    # # Plot target achievement
    # plot_target_achievement(
    #     generated_df=generated_df,
    #     target_values=target_values,
    #     property_cols=property_cols,
    #     property_names=property_names,
    #     output_dir=args.output_dir
    # )
    
    print("\nGeneration and analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
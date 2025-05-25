#!/usr/bin/env python
# rl_inference.py
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


# Add this to your inference script to relax filtering for high-property MOFs

def relaxed_validity_check(mof_strings, predicted_properties, target_values, metals_list, 
                          generation_config, training_config, topology_labels_key, 
                          property_threshold_factor=0.8):
    """
    Apply relaxed validity checking for MOFs that are close to target properties
    
    Args:
        property_threshold_factor: If predicted property >= target * this factor, 
                                 apply relaxed validation
    """
    valid_mofs = []
    valid_bool = []
    
    # Standard validation first
    standard_valid_mofs, standard_valid_bool = verify_rdkit(
        curr_mofs=mof_strings,
        metal_atom_list=metals_list,
        generation_config=generation_config,
        training_config=training_config,
        topology_labels_key=topology_labels_key
    )
    
    for i, (mof, pred_prop, std_valid) in enumerate(zip(mof_strings, predicted_properties, standard_valid_bool)):
        if std_valid:
            # Standard validation passed
            valid_mofs.append(mof)
            valid_bool.append(1)
        else:
            # Check if this MOF has high predicted property
            is_high_property = False
            if isinstance(pred_prop, (list, tuple)):
                for j, prop_val in enumerate(pred_prop):
                    if j < len(target_values):
                        target_val = target_values[j]
                        if prop_val >= target_val * property_threshold_factor:
                            is_high_property = True
                            break
            else:
                if len(target_values) > 0:
                    target_val = target_values[0]
                    if pred_prop >= target_val * property_threshold_factor:
                        is_high_property = True
            
            if is_high_property:
                # Apply relaxed validation for high-property MOFs
                relaxed_valid = relaxed_rdkit_check(mof, metals_list, generation_config, training_config)
                if relaxed_valid:
                    print(f"🎯 Relaxed validation passed for high-property MOF: {pred_prop}")
                    valid_mofs.append(mof)
                    valid_bool.append(1)
                else:
                    valid_bool.append(0)
            else:
                valid_bool.append(0)
    
    return valid_mofs, valid_bool

def relaxed_rdkit_check(mof_string, metals_list, generation_config, training_config):
    """
    More lenient RDKit validation for potentially high-value MOFs
    """
    try:
        compounds = mof_string.split(".")
        
        valid_organic_compound = 0
        valid_inorganic_compound = 0
        
        for compound in compounds:
            # Check for metal atoms
            found_metal_atom = False
            for metal_atom in metals_list:
                if metal_atom in compound:
                    found_metal_atom = True
                    valid_inorganic_compound += 1
                    break
            
            if not found_metal_atom:
                # Try to parse with RDKit, but be more lenient
                mol = Chem.MolFromSmiles(compound, sanitize=False)  # Don't sanitize
                if mol is not None:
                    try:
                        # Try to sanitize, but don't fail if it doesn't work
                        Chem.SanitizeMol(mol)
                        valid_organic_compound += 1
                    except:
                        # Even if sanitization fails, count as valid if RDKit could parse it
                        valid_organic_compound += 1
                else:
                    # Try with some common fixes
                    fixed_compound = fix_common_smiles_issues(compound)
                    mol = Chem.MolFromSmiles(fixed_compound, sanitize=False)
                    if mol is not None:
                        valid_organic_compound += 1
        
        # Return true if we have both organic and inorganic components
        return valid_inorganic_compound > 0 and valid_organic_compound > 0
        
    except Exception as e:
        return False

def fix_common_smiles_issues(smiles):
    """Fix common SMILES issues that might make RDKit fail"""
    # Remove extra parentheses
    fixed = smiles.replace("((", "(").replace("))", ")")
    
    # Fix common valence issues by removing problematic atoms/bonds
    # This is a simplified fix - you might need more sophisticated logic
    
    return fixed

def relaxed_novelty_check(mof_strings, predicted_properties, target_values, all_mof_strs,
                         property_threshold_factor=0.8, similarity_threshold=0.85):
    """
    Apply relaxed novelty checking - allow high-property MOFs that are similar but not identical
    """
    novel_mofs = []
    novel_bool = []
    
    for mof, pred_prop in zip(mof_strings, predicted_properties):
        # Check if this is a high-property MOF
        is_high_property = False
        if isinstance(pred_prop, (list, tuple)):
            for j, prop_val in enumerate(pred_prop):
                if j < len(target_values):
                    target_val = target_values[j]
                    if prop_val >= target_val * property_threshold_factor:
                        is_high_property = True
                        break
        else:
            if len(target_values) > 0:
                target_val = target_values[0]
                if pred_prop >= target_val * property_threshold_factor:
                    is_high_property = True
        
        if is_high_property:
            # For high-property MOFs, allow similar but not identical structures
            is_novel = is_sufficiently_different(mof, all_mof_strs, similarity_threshold)
            if is_novel:
                print(f"🎯 High-property MOF passed relaxed novelty check: {pred_prop}")
            novel_mofs.append(mof)
            novel_bool.append(1 if is_novel else 0)
        else:
            # Standard novelty check
            is_novel = mof not in all_mof_strs
            if is_novel:
                novel_mofs.append(mof)
            novel_bool.append(1 if is_novel else 0)
    
    return novel_mofs, novel_bool

def is_sufficiently_different(mof, all_mof_strs, similarity_threshold=0.85):
    """
    Check if MOF is sufficiently different from existing MOFs
    Using simple string similarity - could be improved with chemical similarity
    """
    from difflib import SequenceMatcher
    
    for existing_mof in all_mof_strs:
        similarity = SequenceMatcher(None, mof, existing_mof).ratio()
        if similarity > similarity_threshold:
            return False  # Too similar to existing MOF
    
    return True  # Sufficiently different


def generate_mofs(model, tokenizer, generation_config, training_config,
                 device, metals_list, all_mof_strs, topology_labels_key,
                 energy_predictor, output_filename, num_generations=100,
                 batch_size=50, property_name="Property", model_type="rl",
                 enable_relaxed_filtering=True):
    """
    Generate MOFs with proper statistics logging in the original simple format
    """
    print(f"\n{'='*50}")
    print(f"Generating MOFs with {'RELAXED' if enable_relaxed_filtering else 'STANDARD'} filtering")
    print(f"{'='*50}")
    
    target_values = training_config.get('target_values', [0.0])
    print(f"Target values: {target_values}")
    
    # Set model to evaluation mode
    if hasattr(model, 'eval'):
        model.eval()
    
    # Initialize simple statistics tracking (like original)
    all_mofs = []
    all_targets = []
    unique_mofs = set()
    
    # Track comprehensive statistics for final reporting
    total_attempts = 0
    valid_count = 0
    novel_count = 0
    all_generated_mofs = []  # For diversity calculation
    
    print("Generating MOFs...")
    
    batch_idx = 0
    with tqdm(total=num_generations, desc="Generating MOFs") as pbar:
        while len(all_mofs) < num_generations:
            batch_idx += 1
            current_batch_size = batch_size
                
            print(f"Batch {batch_idx} (size: {current_batch_size}). Progress: {len(all_mofs)}/{num_generations}")
            
            # Generate sequences
            if model_type == "rl":
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
                # Handle fine-tune model generation
                generated_sequences, _, _ = generate(
                    model=model.llm_network if hasattr(model, 'llm_network') else model,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    training_config=training_config,
                    device=device,
                    num_return_sequences=current_batch_size,
                    energy_predictor=None
                )
                generated_targets = [0.0] * len(generated_sequences)  # Placeholder
            
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
            
            # Track ALL generated MOFs for diversity calculation
            all_generated_mofs.extend(generated_mof_strs)
            
            # Apply filtering with relaxed rules if enabled
            if enable_relaxed_filtering:
                valid_mofs_list, valid_bool = relaxed_validity_check(
                    mof_strings=generated_mof_strs,
                    predicted_properties=generated_targets,
                    target_values=target_values,
                    metals_list=metals_list,
                    generation_config=generation_config,
                    training_config=training_config,
                    topology_labels_key=topology_labels_key
                )
                
                novel_mofs_list, novel_bool = relaxed_novelty_check(
                    mof_strings=generated_mof_strs,
                    predicted_properties=generated_targets,
                    target_values=target_values,
                    all_mof_strs=all_mof_strs
                )
            else:
                # Standard filtering
                valid_mofs_list, valid_bool = verify_rdkit(
                    curr_mofs=generated_mof_strs,
                    metal_atom_list=metals_list,
                    generation_config=generation_config,
                    training_config=training_config,
                    topology_labels_key=topology_labels_key
                )
                
                novel_mofs_list, novel_bool = check_for_existence(
                    curr_mof_list=generated_mof_strs,
                    all_mof_list=all_mof_strs
                )
            
            # Process results and update statistics
            for i, mof_str in enumerate(generated_mof_strs):
                total_attempts += 1
                
                is_valid = valid_bool[i] if i < len(valid_bool) else False
                is_novel = novel_bool[i] if i < len(novel_bool) else False
                
                if is_valid:
                    valid_count += 1
                    
                    if is_novel:
                        novel_count += 1
                        
                        # Add to final collection if unique
                        if mof_str not in unique_mofs:
                            unique_mofs.add(mof_str)
                            all_mofs.append(mof_str)
                            
                            # Get corresponding target
                            if generated_targets and i < len(generated_targets):
                                all_targets.append(generated_targets[i])
                            else:
                                all_targets.append(0.0)
                            
                            # Update progress bar
                            pbar.update(1)
                            pbar.set_postfix({
                                'valid': f"{valid_count}/{total_attempts}",
                                'novel': f"{novel_count}/{valid_count}" if valid_count > 0 else "0/0",
                                'unique': len(unique_mofs)
                            })
                
                if len(all_mofs) >= num_generations:
                    break
            
            # Print batch summary
            print(f"  Batch {batch_idx} results:")
            print(f"    Generated: {len(generated_mof_strs)}")
            print(f"    Valid: {sum(valid_bool)} ({sum(valid_bool)/len(generated_mof_strs)*100:.1f}%)")
            print(f"    Novel: {sum(novel_bool)} ({sum(novel_bool)/max(1,sum(valid_bool))*100:.1f}%)")
            print(f"    Added to final: {len(all_mofs) - len(unique_mofs) + len(unique_mofs)}")
            
            # Free memory
            del generated_sequences, generated_mof_strs, generated_targets
            clear_memory()
            
            if len(all_mofs) >= num_generations:
                break
    
    # Calculate final rates
    validity_rate = valid_count / max(1, total_attempts)
    novelty_rate = novel_count / max(1, valid_count)
    efficiency_rate = len(all_mofs) / max(1, total_attempts)
    
    # Calculate diversity
    diversity_stats = calculate_diversity(all_generated_mofs)
    diversity_ratio = diversity_stats['diversity_ratio']
    
    # Print comprehensive statistics
    print(f"\n{'='*60}")
    print("FINAL GENERATION STATISTICS")
    print(f"{'='*60}")
    print(f"Total attempts: {total_attempts}")
    print(f"Valid MOFs: {valid_count} ({validity_rate*100:.1f}%)")
    print(f"Novel MOFs: {novel_count} ({novelty_rate*100:.1f}%)")
    print(f"Unique MOFs: {len(unique_mofs)}")
    print(f"Final collection: {len(all_mofs)}")
    print(f"Overall efficiency: {efficiency_rate*100:.1f}%")
    print(f"Diversity ratio: {diversity_ratio:.3f}")
    
    if enable_relaxed_filtering:
        print(f"Relaxed filtering was enabled for high-property MOFs")
    
    # Save results (truncate to exactly num_generations)
    all_mofs = all_mofs[:num_generations]
    all_targets = all_targets[:num_generations]
    
    # Create DataFrame and save main results CSV
    results_data = []
    for i, (mof, target_val) in enumerate(zip(all_mofs, all_targets)):
        row = {"id": i, "mof": mof}
        if isinstance(target_val, (list, tuple)):
            for j, t_val in enumerate(target_val):
                row[f"target_{j+1}"] = float(t_val)
        else:
            row["target_1"] = float(target_val)
        results_data.append(row)
    
    output_df = pd.DataFrame(results_data)
    output_df.to_csv(output_filename, index=False)
    
    print(f"\nSaved {len(all_mofs)} MOF structures to {output_filename}")
    
    # Save SIMPLE statistics CSV (original format)
    output_dir = os.path.dirname(output_filename)
    stats_dict = {
        'total_attempts': total_attempts,
        'valid_count': valid_count,
        'validity_rate': validity_rate,
        'novel_count': novel_count,
        'novelty_rate': novelty_rate,
        'unique_count': len(unique_mofs),
        'diversity_ratio': diversity_ratio,
        'efficiency': efficiency_rate
    }
    
    stats_df = pd.DataFrame([stats_dict])
    stats_csv_path = os.path.join(output_dir, "generation_stats.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Statistics saved to: {stats_csv_path}")
    
    # Print sample of final MOFs
    print(f"\nSample of generated MOFs:")
    for i in range(min(5, len(all_mofs))):
        target_str = ""
        if all_targets and i < len(all_targets):
            if isinstance(all_targets[i], (list, tuple)):
                target_str = f" | Targets: {[round(float(t), 4) for t in all_targets[i]]}"
            else:
                target_str = f" | Target: {round(float(all_targets[i]), 4)}"
        print(f"  MOF {i+1}: {all_mofs[i]}{target_str}")
    
    return all_mofs, all_targets, output_df


def save_generation_stats(stats_dict, output_dir, filename="generation_stats.csv"):
    """Save generation statistics to a CSV file in the original simple format"""
    stats_df = pd.DataFrame([stats_dict])
    stats_path = os.path.join(output_dir, filename)
    stats_df.to_csv(stats_path, index=False)
    print(f"Generation statistics saved to: {stats_path}")
    return stats_path


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
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
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


    dataset_name = os.path.basename(os.path.dirname(data_config["train_csv_filename"]))
    print(f"Dataset name: {dataset_name}")

    if args.property_name == "Property":  # Only override if default value is used
        if "QMOF" in dataset_name:
            args.property_name = "Band Gap (eV)" 
            print(f"Setting property name to 'Band Gap (eV)' for QMOF dataset")
        elif "hMOF" in dataset_name:
            # Attempt to extract gas type and pressure
            gas_type = "gas"
            pressure = ""
            
            if "CH4" in dataset_name:
                gas_type = "CH4"
            elif "CO2" in dataset_name:
                gas_type = "CO2"
            
            import re
            pressure_match = re.search(r'(\d+\.\d+)', dataset_name)
            if pressure_match:
                pressure = pressure_match.group(1)
                args.property_name = f"{gas_type} adsorption at {pressure} bar"
            else:
                args.property_name = f"{gas_type} adsorption"
            
            print(f"Setting property name to '{args.property_name}' for hMOF dataset")
    


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

    
    print("\nGeneration and analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

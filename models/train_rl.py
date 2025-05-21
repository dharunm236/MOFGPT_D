import os
import sys
import torch
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path
import gc
from torch import autocast
import torch.nn.functional as F

sys.path.append("../")  
from modules.model_utils import (
    get_model, save_model, clear_memory, log_gpu_memory, get_cosine_schedule_with_warmup
)
from modules.data_utils import (
    load_all_mofs, check_for_existence, verify_rdkit, 
    process_sequence_to_str, verify_topology
)
from tokenizer.mof_tokenizer_gpt import MOFTokenizerGPT
from modules.models import LLMModel, LLMFineTuneModel
from modules.generation import generate
from modules.reward import RewardFunction
from modules.experience_replay import ExperienceReplayBuffer
import random



def train_one_epoch(base_model_class,
                    batch_tokens,
                    batch_reward,
                    optimizer,
                    training_config,
                    device,
                    scheduler,
                    epoch,
                    experience_buffer=None,
                    scaler=None):
    """
    Memory-optimized training for one epoch with batch processing and experience replay
    
    Args:
        base_model_class: Model class wrapper
        batch_tokens: Batch of token sequences
        batch_reward: Batch of rewards
        optimizer: Optimizer
        training_config: Training configuration
        device: Device to use
        scheduler: Learning rate scheduler
        epoch: Current epoch
        experience_buffer: Experience replay buffer (optional)
        scaler: Gradient scaler for mixed precision (optional)
        
    Returns:
        batch_expected_reward: Expected reward for the batch
        batch_loss: Loss value for the batch
    """
    # Set model to training mode
    base_model_class.network.train()
    
    # Log memory before training
    # log_gpu_memory("Before training")
    
    # Determine if we should use experience replay
    # Only start after a certain number of epochs (prevent catastrophic forgetting)
    replay_start_epoch = training_config.get("replay_start_epoch", 100)
    use_replay = (experience_buffer is not None and 
                training_config.get("use_experience_replay", True) and 
                epoch >= replay_start_epoch and 
                len(experience_buffer.buffer) > 0)
    
    # Update experience buffer with current batch samples
    if experience_buffer is not None:
        added_count = 0
        for tokens, reward in zip(batch_tokens, batch_reward):
            if experience_buffer.add_experience(tokens, reward):  # This moves tokens to CPU
                added_count += 1
        
        if added_count > 0:
            print(f"Added {added_count}/{len(batch_tokens)} experiences to buffer")
    
    # Combine current batch with experiences from buffer
    combined_tokens = batch_tokens.copy()
    combined_rewards = batch_reward.copy()
    
    if use_replay:
        # Use 30% of batch size for replay samples
        replay_ratio = training_config.get("replay_ratio", 0.3)
        replay_size = max(1, int(len(batch_tokens) * replay_ratio))
        
        # Sample from buffer (tokens are on CPU)
        replay_tokens, replay_rewards = experience_buffer.sample_batch(replay_size)
        
        # Add to combined batch if samples were returned
        if replay_tokens:
            # Move replay samples to GPU
            replay_tokens_gpu = [tokens.to(device) for tokens in replay_tokens]
            combined_tokens.extend(replay_tokens_gpu)
            combined_rewards.extend(replay_rewards)
            
            print(f"Using experience replay: {len(replay_tokens)} samples from buffer")
            
            # Print buffer stats
            stats = experience_buffer.get_stats()
            print(f"Buffer stats: size={stats['size']}/{experience_buffer.max_size}, " 
                  f"mean_reward={stats['mean_reward']:.4f}, "
                  f"max_reward={stats['max_reward']:.4f}")
    elif experience_buffer is not None and epoch < replay_start_epoch:
        print(f"Experience replay will start at epoch {replay_start_epoch} (current: {epoch})")
    
    # Initialize loss and reward accumulators
    batch_loss = 0.0
    batch_expected_reward = 0.0
    batch_size = len(combined_tokens)
    
    # Skip if batch is empty
    if batch_size == 0:
        return 0, 0
    
    # Process in smaller mini-batches to manage memory better
    mini_batch_size = 8  # Adjust based on your GPU memory
    optimizer.zero_grad()
    
    for mini_batch_idx in range(0, batch_size, mini_batch_size):
        mini_batch_end = min(mini_batch_idx + mini_batch_size, batch_size)
        curr_mini_batch_size = mini_batch_end - mini_batch_idx
        
        # Process each sample in the mini-batch
        curr_batch_loss = 0.0
        curr_batch_reward = 0.0
        
        for i in range(mini_batch_idx, mini_batch_end):
            tokens = combined_tokens[i]
            reward = combined_rewards[i]
            
            # Calculate discounted returns
            # Power calculates how much to discount future rewards
            # Higher powers discount future rewards more
            discounted_returns = (torch.pow(
                training_config["discount_factor"],
                torch.arange(tokens.shape[1]-1, 0, -1).to(device)
            ) * reward)
            
            # Prepare inputs for model
            attention_mask = torch.ones_like(tokens).to(device)
            
            # Ensure tokens are on the correct device and type
            if tokens.device != device:
                tokens = tokens.to(device)
            tokens = tokens.type(torch.LongTensor).to(device)
            
            target_token_ids = tokens.clone()
            
            # Forward pass
            # with autocast(enabled=training_config['fp16']):
            with autocast(device_type='cuda', enabled=training_config['fp16']):
                y_hat = base_model_class.forward(
                    tokens,
                    attention_mask,
                    target_token_ids,
                    training_config['fp16']
                )
                
                # Get logits and apply softmax
                logits = y_hat.logits
                logits = logits[:, :-1, :]  # Remove last token prediction
                logits = logits.squeeze(0)
                log_preds = F.log_softmax(logits, dim=1)
                
                # Get indices of chosen tokens
                idxs = tokens[0, 1:].view(-1, 1)
                
                # Get log probabilities of chosen tokens
                action_values = log_preds.gather(dim=1, index=idxs).view(-1, 1)
                
                # Calculate expected reward (policy gradient)
                expected_reward = -torch.sum(action_values * discounted_returns.view(-1, 1))
                
                # Scale the loss by batch size for gradient accumulation
                current_loss = expected_reward / batch_size
            
            # Add to batch totals
            curr_batch_reward += reward
            curr_batch_loss += expected_reward.item()
            
            # Backward pass for this sample
            if training_config["fp16"] and scaler is not None:
                # Mixed precision training
                scaler.scale(current_loss).backward()
            else:
                current_loss.backward()
            
            # Free memory for next iteration
            del y_hat, logits, log_preds, action_values, expected_reward
            del attention_mask, target_token_ids, discounted_returns
        
        # Update batch accumulators
        batch_loss += curr_batch_loss
        batch_expected_reward += curr_batch_reward
        
        # Clear memory after each mini-batch
        clear_memory()
    
    # Average the accumulated values
    batch_loss /= batch_size
    batch_expected_reward /= batch_size
    
    # Apply gradient clipping to prevent large updates
    gradient_clip = training_config.get("gradient_clip", 0.0)
    
    if training_config["fp16"] and scaler is not None:
        # Mixed precision training
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base_model_class.network.parameters(), gradient_clip)
            
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard training
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(base_model_class.network.parameters(), gradient_clip)
            
        optimizer.step()
    
    # Update learning rate
    if scheduler is not None:
        scheduler.step()
    
    # Clean up memory
    clear_memory()
    # log_gpu_memory("After training")
    
    # Log training stats
    print(f"Training stats: loss={batch_loss:.6f}, reward={batch_expected_reward:.4f}")
    
    return batch_expected_reward, batch_loss



# def final_test_evaluation(num_targets,
#                          base_model,
#                          tokenizer,
#                          generation_config,
#                          training_config,
#                          device,
#                          topology_labels_key,
#                          all_mof_strs,
#                          all_mof_strs_train,
#                          all_mof_strs_val,
#                          all_mof_strs_test,
#                          metals_list,
#                          energy_predictor,
#                          reward_function,
#                          output_filename):
#     """Run final test evaluation and save generations"""
#     print("\n" + "="*70)
#     print("FINAL TEST EVALUATION")
#     print("="*70)
    
#     # First, do a standard evaluation on the test set
#     test_reward, \
#     novelty_reward, \
#     validity_reward, \
#     diversity_reward, \
#     target_reward, \
#     test_avg_predictions = evaluate(
#         num_targets=num_targets,
#         base_model=base_model,
#         tokenizer=tokenizer,
#         generation_config=generation_config,
#         training_config=training_config,
#         device=device,
#         topology_labels_key=topology_labels_key,
#         all_mof_strs=all_mof_strs,
#         all_mof_strs_train=all_mof_strs_train,
#         all_mof_strs_val=all_mof_strs_val,
#         all_mof_strs_test=all_mof_strs_test,
#         metals_list=metals_list,
#         energy_predictor=energy_predictor,
#         reward_function=reward_function,
#         eval_set="test"  # Explicitly use test set
#     )
    
#     # No need to log to WandB for final test - just print results
    
#     print("\n" + "="*50)
#     print("FINAL TEST RESULTS SUMMARY")
#     print("="*50)
#     print(f"Total reward: {test_reward:.4f}")
#     print(f"Novelty score: {novelty_reward:.4f}")
#     print(f"Validity score: {validity_reward:.4f}")
#     print(f"Diversity score: {diversity_reward:.4f}")
#     print(f"Target reward: {target_reward:.4f}")
    
#     for i, avg_pred in enumerate(test_avg_predictions):
#         if avg_pred is not None:
#             target_val = training_config.get("target_values", [0.0] * num_targets)[i]
#             print(f"Target {i+1}: value={target_val:.4f}, prediction={avg_pred:.4f}")
 
#     return test_reward


def evaluate(num_targets,
            base_model,
            tokenizer,
            generation_config,
            training_config,
            device,
            topology_labels_key,
            all_mof_strs,
            all_mof_strs_train,
            all_mof_strs_val,
            all_mof_strs_test,
            metals_list,
            energy_predictor,
            reward_function,  # Add the reward_function as a parameter
            eval_set="val"):
    """
    Memory-optimized function to evaluate model performance
    
    Args:
        num_targets: Number of target properties
        base_model: Model to evaluate
        tokenizer: Tokenizer for MOF strings
        generation_config: Configuration for generation
        training_config: Training configuration
        device: Device to use
        topology_labels_key: List of valid topology keys
        all_mof_strs: List of all MOFs
        all_mof_strs_train: List of training MOFs
        all_mof_strs_val: List of validation MOFs
        all_mof_strs_test: List of test MOFs
        metals_list: List of metal atoms
        energy_predictor: Model for predicting targets
        reward_function: Instance of RewardFunction
        eval_set: Which dataset to evaluate on ("train", "val", or "test")
        
    Returns:
        batch_reward_mean: Mean total reward
        batch_novelty_reward_mean: Mean novelty reward
        batch_validity_reward_mean: Mean validity reward
        batch_diversity_reward_mean: Mean diversity reward
        batch_target_reward_mean: Mean target reward
        avg_predictions: List of predicted target values
    """
    print("\n" + "="*50)
    print("EVALUATION")
    print("="*50)
    
    # Clear memory before evaluation
    clear_memory()

    data_to_check_against = all_mof_strs_train
    if eval_set == "val":
        data_to_check_against = all_mof_strs_val
    elif eval_set == "test":
        data_to_check_against = all_mof_strs_test
    
    # Generate sequences
    generated_sequences, generated_scores, generated_targets = generate(
        model=base_model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        training_config=training_config,
        device=device,
        num_return_sequences=generation_config["test_num_return_sequences"],
        energy_predictor=energy_predictor
    )
    
    # Convert sequences to MOF strings (process in batches to manage memory)
    generated_mof_strs = []
    batch_size = min(10, len(generated_sequences))
    
    for batch_idx in range(0, len(generated_sequences), batch_size):
        batch_end = min(batch_idx + batch_size, len(generated_sequences))
        batch_sequences = generated_sequences[batch_idx:batch_end]
        
        for sequence in batch_sequences:
            # Process sequence to string
            sequence_str = process_sequence_to_str(
                sequence=sequence,
                tokenizer=tokenizer,
                training_config=training_config,
                generation_config=generation_config
            )
            generated_mof_strs.append(sequence_str)
        
        # Clear batch memory
        del batch_sequences
        clear_memory()
    
    # Print sample of generated MOFs and targets
    print("\nSample generated MOFs:")
    for i in range(min(5, len(generated_mof_strs))):
        target_str = ""
        if generated_targets and i < len(generated_targets):
            target_str = f" | Targets: {generated_targets[i]}"
        print(f"MOF {i+1}: {generated_mof_strs[i]}{target_str}")
    
    # Calculate rewards using the RewardFunction
    batch_reward, novelty_reward, validity_reward, diversity_reward, target_rewards_list = reward_function.calculate_rewards(
        new_mofs=generated_mof_strs,
        metals_list=metals_list,
        all_mof_strs=all_mof_strs if training_config['check_in_test'] else data_to_check_against,
        predicted_targets=generated_targets,
        generation_config=generation_config,
        training_config=training_config,
        topology_labels_key=topology_labels_key
    )
    
    # Calculate mean rewards with safeguards against division by zero
    batch_reward_mean = sum(batch_reward) / max(1, len(batch_reward))
    
    # Calculate novelty metrics
    novelty_factor = training_config["novelty_factor"]
    novelty_reward_raw = [r/novelty_factor if novelty_factor != 0 else 0 for r in novelty_reward]
    batch_novelty_reward_mean = sum(novelty_reward_raw) / max(1, len(novelty_reward_raw))
    
    # Calculate validity metrics
    validity_factor = training_config["validity_factor"]
    validity_reward_raw = [r/validity_factor if validity_factor != 0 else 0 for r in validity_reward]
    batch_validity_reward_mean = sum(validity_reward_raw) / max(1, len(validity_reward_raw))
    
    # Calculate diversity metrics
    diversity_factor = training_config["diversity_factor"]
    diversity_reward_raw = [r/diversity_factor if diversity_factor != 0 else 0 for r in diversity_reward]
    batch_diversity_reward_mean = sum(diversity_reward_raw) / max(1, len(diversity_reward_raw))
    
    # Calculate target metrics for each target
    target_reward_means = []
    avg_predictions = []
    
    for i, target_rewards in enumerate(target_rewards_list):
        if target_rewards:
            target_mean = sum(target_rewards) / max(1, len(target_rewards))
            target_reward_means.append(target_mean)
            
            # Calculate average prediction
            if generated_targets and len(generated_targets) > 0:
                if isinstance(generated_targets[0], (list, tuple)):
                    pred_vals = [p[i] if i < len(p) else None for p in generated_targets]
                else:
                    pred_vals = generated_targets if i == 0 else []
                
                if pred_vals:
                    valid_preds = [p for p in pred_vals if p is not None]
                    if valid_preds:
                        avg_pred = sum(valid_preds) / len(valid_preds)
                        avg_predictions.append(avg_pred)
                    else:
                        avg_predictions.append(None)
                else:
                    avg_predictions.append(None)
            else:
                avg_predictions.append(None)
    
    # Calculate overall target mean
    batch_target_reward_mean = sum(target_reward_means) / max(1, len(target_reward_means))
    
    # Print detailed evaluation results
    print("\n" + "-"*50)
    print("EVALUATION RESULTS")
    print("-"*50)
    print(f"Total reward: {batch_reward_mean:.4f}")
    
    # Calculate weighted components for clarity
    nov_component = sum(novelty_reward) / max(1, len(novelty_reward))
    val_component = sum(validity_reward) / max(1, len(validity_reward))
    div_component = sum(diversity_reward) / max(1, len(diversity_reward))
    tgt_component = sum([sum(tgt) for tgt in target_rewards_list]) / max(1, len(generated_mof_strs))
    
    # Print component breakdown
    if batch_reward_mean > 0:
        print(f"  → Novelty component: {nov_component:.4f}")
        print(f"  → Validity component: {val_component:.4f}")
        print(f"  → Diversity component: {div_component:.4f}")
        print(f"  → Target component: {tgt_component:.4f}")
    else:
        print(f"  → Novelty component: {nov_component:.4f}")
        print(f"  → Validity component: {val_component:.4f}")
        print(f"  → Diversity component: {div_component:.4f}")
        print(f"  → Target component: {tgt_component:.4f}")
    
    print(f"\nNovelty score: {batch_novelty_reward_mean:.4f} (factor: {novelty_factor:.2f})")
    print(f"Validity score: {batch_validity_reward_mean:.4f} (factor: {validity_factor:.2f})")
    print(f"Diversity score: {batch_diversity_reward_mean:.4f} (factor: {diversity_factor:.2f})")
    
    # Print target-specific results with enhanced information
    print("\nTarget-specific results:")
    for i, target_mean in enumerate(target_reward_means):
        target_weight = training_config.get("target_weights", [1.0] * num_targets)[i]
        target_value = training_config.get("target_values", [0.0] * num_targets)[i]
        opt_mode = training_config.get("optimization_modes", ["higher"] * num_targets)[i]
        
        avg_pred = avg_predictions[i] if i < len(avg_predictions) and avg_predictions[i] is not None else "N/A"
        
        print(f"Target {i+1}:")
        print(f"  → Target value: {target_value:.4f}, Weight: {target_weight:.2f}, Mode: {opt_mode}")
        print(f"  → Avg prediction: {avg_pred if isinstance(avg_pred, str) else f'{avg_pred:.4f}'}")
        print(f"  → Reward: {target_mean:.4f}")
    
    print("-"*50)
    
    # Clean up memory before returning
    del generated_sequences, generated_scores, generated_mof_strs
    del novelty_reward, validity_reward, diversity_reward, target_rewards_list, batch_reward
    clear_memory()
    
    return batch_reward_mean, batch_novelty_reward_mean, batch_validity_reward_mean, batch_diversity_reward_mean, batch_target_reward_mean, avg_predictions

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train MOF-RL model")
    parser.add_argument("--config", type=str, default="../config/rl/config.yaml", 
                        help="Path to main configuration file")
    parser.add_argument("--generation-config", type=str, default="../config/rl/config_generation.yaml", 
                        help="Path to generation configuration file")
    parser.add_argument("--output-dir", type=str, default="./models", 
                        help="Directory to save trained models")
    parser.add_argument("--wandb-project", type=str, default=None, 
                        help="WandB project name (if None, uses config value)")
    parser.add_argument("--wandb-run-name", type=str, default=None, 
                        help="WandB run name (if None, auto-generated)")
    parser.add_argument("--cuda-device", type=int, default=None, 
                        help="CUDA device ID to use (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, 
                        help="Number of epochs to train (overrides config)")
    parser.add_argument("--temperature", type=float, default=None, 
                        help="Sampling temperature (overrides config)")
    parser.add_argument("--diversity-factor", type=float, default=0.1, 
                        help="Factor for promoting diversity")
    parser.add_argument("--no-fp16", action="store_true", 
                        help="Disable mixed precision training")
    parser.add_argument("--final-generation-count", type=int, default=None,
                        help="Number of final generations (overrides config)")
    
    return parser.parse_args()


def main():
    """Main function for MOF-RL training"""
    # Parse arguments
    args = parse_arguments()

    # Set seeds for reproducibility
    seed = 42  # You can choose any integer
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print("\n" + "="*70)
    print("MOF-RL Training with Enhanced Exploration and Target Matching")
    print("="*70 + "\n")
    
    # Load configuration files
    print(f"Loading configuration from:")
    print(f"  - {args.config}")
    print(f"  - {args.generation_config}")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(config["model"]["model_config_filename"], "r") as f:
        model_config = yaml.safe_load(f)
    with open(args.generation_config, "r") as f:
        generation_config = yaml.safe_load(f)

    # Extract configuration sections
    training_config = config["training"]
    data_config = config["data"]
    base_model_config = model_config["base_model"]
    fine_tune_config = model_config["fine_tune"]
    
    # Override configurations with command line arguments
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
        print(f"Overriding epochs: {args.epochs}")
    
    if args.cuda_device is not None:
        training_config["cuda_device"] = args.cuda_device
        print(f"Overriding CUDA device: {args.cuda_device}")
    
    if args.temperature is not None:
        generation_config["temperature"] = args.temperature
        print(f"Overriding temperature: {args.temperature}")
    
    if args.no_fp16:
        training_config["fp16"] = False
        print("Disabled mixed precision training (FP16)")
    
    if args.final_generation_count is not None:
        training_config["final_generation_count"] = args.final_generation_count
        print(f"Overriding final generation count: {args.final_generation_count}")
    
    # Add diversity factor to training config
    training_config["diversity_factor"] = args.diversity_factor
    print(f"Setting diversity factor: {args.diversity_factor}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    training_config["save_dir"] = args.output_dir
    
    # Get number of targets and enhance print output
    num_targets = training_config.get("num_targets", 1)
    
    # Print training configuration in a more organized way
    print("\n" + "="*50)
    print("CONFIGURATION SUMMARY")
    print("="*50)
    
    print("\nTraining parameters:")
    print(f"  • Epochs: {training_config['epochs']}")
    print(f"  • Learning rate: {training_config['optimizer']['lr']}")
    print(f"  • Batch size: {generation_config['num_return_sequences']}")
    print(f"  • Device: {'CUDA' if training_config['device'] == 'cuda' else 'CPU'}")
    if training_config['device'] == 'cuda':
        print(f"  • CUDA device: {training_config.get('cuda_device', 0)}")
    print(f"  • Mixed precision: {'Enabled' if training_config.get('fp16', False) else 'Disabled'}")
    
    print("\nGeneration parameters:")
    print(f"  • Temperature: {generation_config['temperature']}")
    print(f"  • Top-k: {generation_config['top_k']}")
    print(f"  • Max sequence length: {generation_config['max_seq_len']}")
    print(f"  • Diversity factor: {training_config['diversity_factor']}")
    
    if num_targets > 0:
        print("\nTarget properties:")
        target_values = training_config.get("target_values", [0.0] * num_targets)
        target_weights = training_config.get("target_weights", [1.0] * num_targets)
        optimization_modes = training_config.get("optimization_modes", ["higher"] * num_targets)
        
        for i in range(num_targets):
            print(f"  • Target {i+1}: value={target_values[i]:.4f}, weight={target_weights[i]:.2f}, mode={optimization_modes[i]}")
    
    # Experience replay configuration
    use_replay = training_config.get("use_experience_replay", True)
    replay_buffer_size = training_config.get("replay_buffer_size", 1000)
    replay_start_epoch = training_config.get("replay_start_epoch", 100)
    
    if use_replay:
        print("\nExperience replay:")
        print(f"  • Buffer size: {replay_buffer_size}")
        print(f"  • Start epoch: {replay_start_epoch}")
        print(f"  • Replay ratio: {training_config.get('replay_ratio', 0.3):.2f}")
        print(f"  • Reward threshold: {training_config.get('replay_reward_threshold', 0.0):.2f}")
    
    # Check for CUDA and available memory
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"\nGPU information:")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  • Device {i}: {device_name}, {total_memory:.2f} GB total memory")
    else:
        print("\nNo CUDA devices available, using CPU")

    # Load topology labels
    topology_labels_filename = data_config["topology_labels_map_filename"]
    with open(topology_labels_filename, "r") as f:
        topology_labels_map = json.load(f)
    topology_labels_key = list(topology_labels_map.keys())

    # Load metals list
    with open(generation_config["metals_list_file"], "r") as f:
        metals_list = [line.strip() for line in f.readlines()]
    
    print(f"\nData:")
    print(f"  • Loaded {len(metals_list)} valid metal atoms")
    print(f"  • Loaded {len(topology_labels_key)} valid topologies")

    # Set device for training with better error handling
    if training_config["device"] == "cuda":
        if torch.cuda.is_available():
            # Get specified CUDA device or default to the first one
            specified_device = training_config.get("cuda_device", 0)
            if specified_device >= torch.cuda.device_count():
                print(f"Warning: Specified CUDA device {specified_device} not available. Using device 0 instead.")
                specified_device = 0
            device = torch.device(f'cuda:{specified_device}')
            print(f"Using CUDA device {specified_device}: {torch.cuda.get_device_name(specified_device)}")
        else:
            print("Warning: CUDA requested but not available. Using CPU instead.")
            device = torch.device('cpu')
            training_config["fp16"] = False  # Disable FP16 on CPU
    else:
        device = torch.device('cpu')
        print("Using CPU for training (as specified in config)")
    
    # Initialize tokenizer
    print("\nInitializing tokenizer...")
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
    print("Initializing base model...")
    clear_memory()  # Ensure clean GPU state before model initialization
    try:
        base_model = get_model(config=base_model_config, device=device)
        print("Base model initialized successfully")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        return
    
    log_gpu_memory("After model initialization")
    
    # Initialize optimizer with proper parameter filtering to avoid None gradients
    print("Initializing optimizer...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, base_model.parameters()),
        lr=training_config["optimizer"]["lr"],
        weight_decay=training_config["optimizer"]["weight_decay"]
    )
    
    # Initialize learning rate scheduler
    if training_config["scheduler"]["type"] == "cosine":
        num_training_steps = training_config["epochs"] 
        num_warm_up_steps = training_config["scheduler"]["warmup_ratio"] * num_training_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warm_up_steps,
            num_training_steps=num_training_steps
        )
        print(f"Using cosine scheduler with warmup: {num_warm_up_steps} warmup steps, {num_training_steps} total steps")
    elif training_config["scheduler"]["type"] == "none":
        scheduler = None
        print("No learning rate scheduler used")

    # Initialize experience replay buffer if enabled
    experience_buffer = None
    if training_config.get("use_experience_replay", True):
        experience_buffer = ExperienceReplayBuffer(
            max_size=replay_buffer_size, 
            reward_threshold=training_config.get("replay_reward_threshold", 0.0)
        )
        print(f"Initialized experience replay buffer with size {replay_buffer_size}")
    else:
        print("Experience replay disabled in configuration")

    # Training state initialization
    start_epoch = 1
    max_val_reward = -1e9  # Best validation reward so far

    clear_memory()
    
    # Create or set model output path
    model_basename = os.path.basename(base_model_config["pretrained_model_path"]).split(".")[0]
    output_model_name = f"{model_basename}_RL.pt"
    final_model_path = os.path.join(args.output_dir, output_model_name)
    print(f"Model will be saved as: {final_model_path}")
    
    # Load saved model if resuming training
    if training_config["resume_training"]:
        print(f"\nResuming training from {training_config['resume_model_path']}")
        try:
            checkpoint = torch.load(training_config["resume_model_path"], map_location=device)
            
            # Load model weights
            base_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load scheduler state if available
            if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            # Set training state
            start_epoch = checkpoint["epoch"] + 1
            if "max_val_reward" in checkpoint:
                max_val_reward = checkpoint["max_val_reward"]
            elif "max_test_reward" in checkpoint:  # For backward compatibility
                max_val_reward = checkpoint["max_test_reward"]
            
            print(f"Loaded model from epoch {checkpoint['epoch']} with validation reward {max_val_reward:.4f}")
            del checkpoint
            clear_memory()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Starting from scratch instead")
            start_epoch = 1
            max_val_reward = -1e9
    else:
        print(f"\nStarting training from pretrained model: {base_model_config['pretrained_model_path']}")
        try:
            base_model_dict = torch.load(base_model_config["pretrained_model_path"], map_location=device)
            
            # Try different possible keys for the state dict
            loaded = False
            possible_keys = ["model_state_dict", "state_dict", "llm_network.state_dict"]
            
            for key in possible_keys:
                if key in base_model_dict:
                    try:
                        base_model.load_state_dict(base_model_dict[key], strict=False)
                        print(f"Base model loaded successfully using key: {key}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load with key {key}: {str(e)}")
            
            if not loaded:
                print("Could not find compatible state dict in the model checkpoint. Starting with random weights.")
                
            del base_model_dict
            clear_memory()
            max_val_reward = -1e9
        except Exception as e:
            print(f"Error loading pretrained model: {str(e)}")
            print("Starting from randomly initialized weights")

    # Initialize energy predictor if needed
    energy_predictor = None
    if training_config["do_energy_reward"]:
        print("\nInitializing target predictor model...")
        try:
            import copy
            
            # Create a copy of the model config and modify the dimensions to match the checkpoint
            target_model_config = copy.deepcopy(base_model_config)
            # These dimensions are typical for fine-tuned models - adjust if your models use different sizes
            target_model_config["hidden_size"] = 768
            target_model_config["n_embd"] = 768
            target_model_config["intermediate_size"] = 3072
            
            # Create a fresh model with the correct dimensions
            target_base_model = get_model(config=target_model_config, device=device)
            
            # Load the checkpoint
            target_model_path = fine_tune_config["pretrained_model_path"]
            print(f"Loading target predictor from: {target_model_path}")
            target_model_saved_dict = torch.load(target_model_path, map_location=device)
            
            # Create the target predictor model
            energy_predictor = LLMFineTuneModel(
                llm_network=target_base_model,
                llm_config=target_model_config,
                fine_tune_config=fine_tune_config,
                is_fp16=training_config['fp16']
            ).to(device)
            
            # Try different possible keys for the state dict
            loaded = False
            possible_keys = ["model_state_dict", "state_dict"]
            
            for key in possible_keys:
                if key in target_model_saved_dict:
                    try:
                        energy_predictor.load_state_dict(target_model_saved_dict[key])
                        print(f"Target predictor model loaded successfully using key: {key}")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load target predictor with key {key}: {str(e)}")
            
            if not loaded:
                print("Could not find compatible state dict in the target model checkpoint")
                if training_config.get("target_predictor_required", True):
                    raise ValueError("Failed to load target predictor model state dictionary")
                else:
                    print("Continuing without target prediction (optional)")
                    training_config["do_energy_reward"] = False
                
            del target_model_saved_dict, target_base_model
            clear_memory()
        except Exception as e:
            print(f"Error loading target predictor: {str(e)}")
            if training_config.get("target_predictor_required", True):
                print("Target predictor is required but failed to initialize. Exiting.")
                sys.exit(1)
            else:
                print("Continuing without target prediction (optional)")
                training_config["do_energy_reward"] = False

    # Load MOF datasets
    print("\nLoading MOF datasets...")
    all_mof_strs_train, all_mof_strs_val, all_mof_strs_test, all_mof_strs = load_all_mofs(
        data_config=data_config,
        training_config=training_config
    )
    
    # Create model wrapper
    base_model_class = LLMModel(network=base_model, config=base_model_config)
    
    # Initialize mixed precision training if enabled
    if training_config["fp16"]:
        if device.type == 'cuda':
            print("\nUsing mixed precision training (FP16)")
            scaler = torch.cuda.amp.GradScaler()
        else:
            print("\nFP16 requested but not supported on CPU. Using FP32 instead.")
            training_config["fp16"] = False
            scaler = None
    else:
        print("\nUsing full precision training (FP32)")
        scaler = None
    
    # Initialize improved reward function
    reward_function = RewardFunction(
        novelty_factor=training_config["novelty_factor"],
        validity_factor=training_config["validity_factor"],
        diversity_factor=training_config["diversity_factor"],
        num_targets=num_targets,
        target_values=training_config.get("target_values", [0.0] * num_targets),
        target_weights=training_config.get("target_weights", [1.0] * num_targets),
        optimization_modes=training_config.get("optimization_modes", ["higher"] * num_targets),
        reward_tolerance=training_config.get("reward_tolerance", 0.1)
    )
    
    # Initialize WandB if requested
    wandb_project = args.wandb_project or config.get("project_name", None)
    # Extract dataset name from train CSV path
    dataset_name = config["data"]["train_csv_filename"].split("/")[-2]  # Get the folder name which should be the dataset name
    print(f"Dataset name: {dataset_name}")
    config["project_name"] = f"mofgpt_{dataset_name}"

    if "QMOF" in dataset_name:
        training_config["optimization_modes"] = ["lower"] * num_targets
        property_name = "Band Gap (eV)"
        print(f"Setting optimization mode to 'lower' for QMOF dataset (Band Gap)")
    elif "hMOF" in dataset_name:
        training_config["optimization_modes"] = ["higher"] * num_targets
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
            property_name = f"{gas_type} adsorption at {pressure} bar"
        else:
            property_name = f"{gas_type} adsorption"
        
        print(f"Setting optimization mode to 'higher' for hMOF dataset ({property_name})")

    # Store property name in config for consistent reference across modules
    training_config["property_name"] = property_name


    if wandb_project and not wandb_project.startswith("#"):
        try:
            print(f"\nInitializing WandB logging with project name: {wandb_project}")
            wandb_config = {
                "model": model_basename,
                "num_targets": num_targets,
                "target_values": training_config.get("target_values", [0.0] * num_targets),
                "temperature": generation_config["temperature"],
                "diversity_factor": training_config["diversity_factor"],
                "epochs": training_config["epochs"],
                "batch_size": generation_config["num_return_sequences"],
                "learning_rate": training_config["optimizer"]["lr"],
                "fp16": training_config["fp16"],
            }

            # Get the target name based on its value from stats
            target_description = "unknown"
            
            # Check if we're in the output directory for a specific target
            output_dir = args.output_dir
            if "rl_mean" in output_dir:
                target_description = "mean"
            elif "rl_mean_plus_1std" in output_dir:
                target_description = "mean_plus_1std"
            elif "rl_mean_plus_2std" in output_dir:
                target_description = "mean_plus_2std"
            elif "rl_mean_plus_3std" in output_dir:
                target_description = "mean_plus_3std"
            elif "rl_mean_minus_1std" in output_dir:
                target_description = "mean_minus_1std"
            elif "rl_mean_minus_2std" in output_dir:
                target_description = "mean_minus_2std"
            elif "rl_mean_minus_3std" in output_dir:
                target_description = "mean_minus_3std"
            elif "rl_custom" in output_dir:
                target_description = "custom"
                
            # Set the run name based on target
            wandb_run_name = args.wandb_run_name or f"RL_{target_description}"
            
            # wandb_run_name = args.wandb_run_name or f"mof-rl-training-{model_basename}"
            wandb.init(project=config["project_name"], config=wandb_config, name=wandb_run_name)
        except Exception as e:
            print(f"Error initializing WandB: {str(e)}")
            print("Continuing without WandB logging")
    
    # Create directory for saving models if it doesn't exist
    if not os.path.exists(training_config['save_dir']):
        os.makedirs(training_config['save_dir'])
        print(f"Created directory for saving models: {training_config['save_dir']}")
    
    # Main training loop
    print("\n" + "="*70)
    print(f"Starting training from epoch {start_epoch} to {training_config['epochs']+start_epoch-1}")
    print("="*70 + "\n")
    
    for epoch in range(start_epoch, training_config["epochs"] + start_epoch):
        print(f"\n{'='*30} Epoch {epoch}/{training_config['epochs']+start_epoch-1} {'='*30}")
        
        # Clear memory at the start of each epoch
        clear_memory()
        log_gpu_memory(f"Start of epoch {epoch}")
        
        # ---- Training ----
        with autocast(device_type='cuda', enabled=training_config['fp16']):
            generated_sequences, \
            generated_scores, \
            generated_targets = generate(
                model=base_model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                training_config=training_config,
                device=device,
                num_return_sequences=generation_config["num_return_sequences"],
                energy_predictor=energy_predictor
            )
        
        # Convert sequences to MOF strings (similar to evaluate function, use batching)
        generated_mof_strs = []
        batch_size = min(10, len(generated_sequences))
        
        for batch_idx in range(0, len(generated_sequences), batch_size):
            batch_end = min(batch_idx + batch_size, len(generated_sequences))
            batch_sequences = generated_sequences[batch_idx:batch_end]
            
            for sequence in batch_sequences:
                # Process sequence to string
                sequence_str = process_sequence_to_str(
                    sequence=sequence,
                    tokenizer=tokenizer,
                    training_config=training_config,
                    generation_config=generation_config
                )
                generated_mof_strs.append(sequence_str)
            
            # Clear batch memory
            del batch_sequences
            clear_memory()
        
        # Calculate rewards using the improved reward function
        batch_reward, \
        novelty_reward, \
        validity_reward, \
        diversity_reward, \
        target_rewards = reward_function.calculate_rewards(
            new_mofs=generated_mof_strs,
            metals_list=metals_list,
            all_mof_strs=all_mof_strs if training_config['check_in_test'] else all_mof_strs_train,
            predicted_targets=generated_targets,
            generation_config=generation_config,
            training_config=training_config,
            topology_labels_key=topology_labels_key
        )

        # Calculate raw novelty, validity, and diversity scores for logging
        novelty_factor = training_config["novelty_factor"]
        validity_factor = training_config["validity_factor"]
        diversity_factor = training_config["diversity_factor"]
        
        novelty_reward_raw = [r/novelty_factor if novelty_factor != 0 else 0 for r in novelty_reward]
        validity_reward_raw = [r/validity_factor if validity_factor != 0 else 0 for r in validity_reward]
        diversity_reward_raw = [r/diversity_factor if diversity_factor != 0 else 0 for r in diversity_reward]

        batch_novelty_score = sum(novelty_reward_raw) / max(1, len(novelty_reward_raw))
        batch_validity_score = sum(validity_reward_raw) / max(1, len(validity_reward_raw))
        batch_diversity_score = sum(diversity_reward_raw) / max(1, len(diversity_reward_raw))

        # Print scores for each epoch
        print(f"Novelty score: {batch_novelty_score:.4f}")
        print(f"Validity score: {batch_validity_score:.4f}")
        print(f"Diversity score: {batch_diversity_score:.4f}")
        
        # Free memory of unused variables before training
        del generated_mof_strs, novelty_reward, validity_reward, diversity_reward
        clear_memory()
        
        # Train for one epoch
        train_reward, train_loss = train_one_epoch(
            base_model_class=base_model_class,
            batch_tokens=generated_sequences,
            batch_reward=batch_reward,
            optimizer=optimizer,
            training_config=training_config,
            device=device,
            scaler=scaler,
            epoch=epoch,
            scheduler=scheduler,
            experience_buffer=experience_buffer
        )
        
        # Free large training tensors
        del generated_sequences, generated_scores, batch_reward
        clear_memory()
        
        # Calculate average predictions for each target for logging
        avg_predictions = []
        for target_idx in range(num_targets):
            if generated_targets and len(generated_targets) > 0:
                if isinstance(generated_targets[0], (list, tuple)):
                    pred_vals = [p[target_idx] if target_idx < len(p) else None for p in generated_targets]
                else:
                    pred_vals = generated_targets if target_idx == 0 else []
                
                if pred_vals:
                    valid_preds = [p for p in pred_vals if p is not None]
                    if valid_preds:
                        avg_pred = sum(valid_preds) / len(valid_preds)
                        avg_predictions.append(avg_pred)
                    else:
                        avg_predictions.append(None)
                else:
                    avg_predictions.append(None)
            else:
                avg_predictions.append(None)
        
        # Log to WandB for every epoch
        if wandb.run is not None:
            try:
                # Create dictionary for wandb logging
                wandb_log = {
                    "train/loss": train_loss,
                    "train/reward": train_reward,
                    "train/novelty_score": batch_novelty_score,
                    "train/validity_score": batch_validity_score,
                    "train/diversity_score": batch_diversity_score,
                }
                
                # Add target-specific metrics
                for i in range(num_targets):
                    target_val = training_config.get("target_values", [0.0] * num_targets)[i]
                    
                    # Add predicted value
                    if i < len(avg_predictions) and avg_predictions[i] is not None:
                        wandb_log[f"train/target_{i+1}_prediction"] = avg_predictions[i]
                        # wandb_log[f"train/target_{i+1}_error"] = abs(avg_predictions[i] - target_val)
                    
                    # Add target value as reference
                    wandb_log[f"reference/target_{i+1}_value"] = target_val

                    # Add target reward component if available
                    if i < len(target_rewards):
                        target_reward_mean = sum(target_rewards[i]) / max(1, len(target_rewards[i]))
                        wandb_log[f"train/target_{i+1}_reward"] = target_reward_mean
                
                # Log the metrics
                wandb.log(wandb_log, step=epoch)
            except Exception as e:
                print(f"Error logging to WandB: {str(e)}")
        
        # Free memory
        del generated_targets, avg_predictions, target_rewards
        clear_memory()
        
        # Evaluation
        if epoch % training_config["eval_interval"] == 0:
            print("\nRunning evaluation...")
            val_reward, \
            val_novelty_reward, \
            val_validity_reward, \
            val_diversity_reward, \
            val_target_reward, \
            val_avg_predictions = evaluate(
                num_targets=num_targets,
                base_model=base_model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                training_config=training_config,
                device=device,
                topology_labels_key=topology_labels_key,
                all_mof_strs=all_mof_strs,
                all_mof_strs_train=all_mof_strs_train,
                all_mof_strs_val=all_mof_strs_val,
                all_mof_strs_test=all_mof_strs_test,
                metals_list=metals_list,
                energy_predictor=energy_predictor,
                reward_function=reward_function,
                eval_set="val"  # Explicitly use validation set
            )
            
            # Log evaluation metrics to WandB
            if wandb.run is not None:
                try:
                    eval_log = {
                        "val/reward": val_reward,
                        "val/novelty_score": val_novelty_reward,
                        "val/validity_score": val_validity_reward,
                        "val/diversity_score": val_diversity_reward,
                        "val/target_reward": val_target_reward,
                    }
                    
                    # Add target-specific validation metrics
                    for i in range(num_targets):
                        target_val = training_config.get("target_values", [0.0] * num_targets)[i]
                        if i < len(val_avg_predictions) and val_avg_predictions[i] is not None:
                            eval_log[f"val/target_{i+1}_prediction"] = val_avg_predictions[i]
                            # eval_log[f"val/target_{i+1}_error"] = abs(val_avg_predictions[i] - target_val)
                    
                    wandb.log(eval_log, step=epoch, commit=True)
                except Exception as e:
                    print(f"Error logging evaluation to WandB: {str(e)}")
            
            # Save model if it's the best so far
            if val_reward > max_val_reward:
                max_val_reward = val_reward
                checkpoint_filename = os.path.join(args.output_dir, f"{model_basename}_RL_best.pt")

                save_model(
                    model=base_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    max_val_reward=max_val_reward,
                    epoch=epoch,
                    experience_buffer=experience_buffer,
                    filename=checkpoint_filename
                )
                
                print(f"New best model with reward {val_reward:.4f}!")
            
            # Also save a periodic checkpoint regardless of performance
            if epoch % training_config.get("checkpoint_interval", 10) == 0:
                checkpoint_filename = os.path.join(args.output_dir, f"{model_basename}_RL_checkpoint.pt")

                save_model(
                    model=base_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    max_val_reward=max_val_reward,
                    epoch=epoch,
                    experience_buffer=experience_buffer,
                    filename=checkpoint_filename
                )
                print(f"Saved checkpoint at epoch {epoch}")
            
            # Free memory
            del val_avg_predictions
            clear_memory()
        
        # Ensure memory is clear at the end of each epoch
        clear_memory()
        log_gpu_memory(f"End of epoch {epoch}")
    
    # Save final model
    save_model(
        model=base_model,
        optimizer=optimizer,
        scheduler=scheduler,
        max_val_reward=max_val_reward,
        epoch=epoch,
        experience_buffer=experience_buffer,
        filename=final_model_path
    )
    
    print("\n" + "="*70)
    print(f"Training complete! Best validation reward: {max_val_reward:.4f}")
    print("="*70)

    # Generate and save final MOF structures
    # final_generations_filename = os.path.join(args.output_dir, f"{model_basename}_RL_generations.csv")
    # final_test_reward = final_test_evaluation(
    #     num_targets=num_targets,
    #     base_model=base_model,
    #     tokenizer=tokenizer,
    #     generation_config=generation_config,
    #     training_config=training_config,
    #     device=device,
    #     topology_labels_key=topology_labels_key,
    #     all_mof_strs=all_mof_strs,
    #     all_mof_strs_train=all_mof_strs_train,
    #     all_mof_strs_val=all_mof_strs_val,
    #     all_mof_strs_test=all_mof_strs_test,
    #     metals_list=metals_list,
    #     energy_predictor=energy_predictor,
    #     reward_function=reward_function,
    #     output_filename=final_generations_filename
    # )

    # print("\n" + "="*70)
    # print(f"Final test reward: {final_test_reward:.4f}")
    # print(f"Final generations saved to: {final_generations_filename}")
    # print("="*70)
    
    # Clean up WandB if it was used
    if wandb.run is not None:
        wandb.finish()
    
    return


if __name__ == "__main__":
    main()
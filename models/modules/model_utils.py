# rl_modules/model_utils.py

"""
Model utilities for MOF-RL

This module provides utility functions for model management.
"""

import os
import gc
import sys
import torch
import numpy as np
import random
import torch.nn as nn
from torch.amp import autocast
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
from transformers import LlamaForCausalLM, \
                         LlamaConfig, \
                         GPT2Config, \
                         GPT2LMHeadModel


def log_gpu_memory(message="", detailed=False):
    """Log GPU memory usage without forcing synchronization unless requested"""
    if torch.cuda.is_available():
        # Only synchronize if detailed stats requested
        if detailed:
            torch.cuda.synchronize()
        
        # Get current device
        current_device = torch.cuda.current_device()
        
        # Get memory information in bytes and convert to GB
        mem_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
        
        print(f"GPU Memory [{message}]: Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
        
        # Only print detailed stats if requested
        if detailed:
            max_allocated = torch.cuda.max_memory_allocated(current_device) / 1024**3
            mem_cached = mem_reserved - mem_allocated
            print(f"  - Cached: {mem_cached:.2f} GB")
            print(f"  - Max Allocated: {max_allocated:.2f} GB")


def clear_memory(force=False):
    """Clear memory only when necessary to avoid expensive operations"""
    gc.collect()
    
    # Only empty cache if we're using more than 90% of available memory or if forced
    if force or (torch.cuda.is_available() and 
                torch.cuda.max_memory_allocated() > 0 and
                torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.9):
        torch.cuda.empty_cache()


def save_model(model, optimizer, scheduler, max_val_reward, epoch, experience_buffer, filename):
    """
    Save model checkpoint with all necessary components
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        max_val_reward: Best validation reward achieved so far
        epoch: Current epoch
        experience_buffer: Experience replay buffer
        filename: Path to save the model
    """
    save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "max_val_reward": max_val_reward,
        "epoch": epoch,
    }
    
    # Save experience buffer stats if available
    if experience_buffer is not None and hasattr(experience_buffer, 'buffer'):
        buffer_stats = experience_buffer.get_stats()
        save_dict["buffer_stats"] = buffer_stats
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Save checkpoint to disk
    torch.save(save_dict, filename)
    print(f"Model saved to {filename}")
    print(f"  - Epoch: {epoch}")
    print(f"  - Max validation reward: {max_val_reward:.4f}")
    
    # Clear memory after saving
    clear_memory()


def load_model(model, optimizer=None, scheduler=None, filename=None, device=None):
    """
    Load model checkpoint
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load state (optional)
        scheduler: Scheduler to load state (optional)
        filename: Path to checkpoint file
        device: Device to load to
        
    Returns:
        model: Loaded model
        optimizer: Updated optimizer if provided
        scheduler: Updated scheduler if provided
        epoch: Last saved epoch
        max_val_reward: Best validation reward
    """
    if not os.path.exists(filename):
        print(f"Checkpoint file not found: {filename}")
        return model, optimizer, scheduler, 0, -float('inf')
    
    try:
        # Load checkpoint with device mapping
        if device is not None:
            checkpoint = torch.load(filename, map_location=device)
        else:
            checkpoint = torch.load(filename)
        
        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state if provided
        if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Get training state
        epoch = checkpoint.get("epoch", 0)
        max_val_reward = checkpoint.get("max_val_reward", checkpoint.get("max_test_reward", -float('inf')))
        
        # Print checkpoint info
        print(f"Loaded checkpoint from {filename}")
        print(f"  - Epoch: {epoch}")
        print(f"  - Max validation reward: {max_val_reward:.4f}")
        
        # Print buffer stats if available
        if "buffer_stats" in checkpoint:
            stats = checkpoint["buffer_stats"]
            print(f"  - Experience buffer: size={stats.get('size', 0)}, mean_reward={stats.get('mean_reward', 0):.4f}")
        
        del checkpoint
        clear_memory()
        
        return model, optimizer, scheduler, epoch, max_val_reward
    
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return model, optimizer, scheduler, 0, -float('inf')


def get_model(config,
              device):
    model_config = GPT2Config(vocab_size=config["vocab_size"],
                              n_embd=config["n_embd"],
                              n_layer=config["n_layer"],
                              n_head=config["n_head"],
                              n_inner=config["n_inner"],
                              activation_function=config["activation_function"],
                              max_position_embeddings=config["max_position_embeddings"],
                              use_cache=config["use_cache"],
                              bos_token_id=config["bos_token_id"],
                              eos_token_id=config["eos_token_id"],
                              resid_pdrop=config["resid_pdrop"],
                              embd_pdrop=config["embd_pdrop"],
                              attn_pdrop=config["attn_pdrop"],
                              layer_norm_epsilon=config["layer_norm_epsilon"],
                              initializer_range=config["initializer_range"],)
    model = GPT2LMHeadModel(model_config).to(device)
    return model


def create_optimizer(model, lr=1e-4, weight_decay=0.01, beta1=0.9, beta2=0.999):
    """
    Create an optimizer for the model
    
    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay factor
        beta1: Adam beta1
        beta2: Adam beta2
        
    Returns:
        optimizer: Configured optimizer
    """
    # Filter out parameters that don't require gradients
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )
    
    return optimizer


def create_scheduler(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a learning rate scheduler
    
    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch (for resuming training)
        
    Returns:
        scheduler: Learning rate scheduler
    """
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        last_epoch=last_epoch
    )
    
    return scheduler


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0.0):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function.
    Adds warm-up steps and a minimum learning rate.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # During warm-up: linear increase from 0 to 1
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # After warm-up: cosine decay from 1 to min_lr
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr + (1 - min_lr) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
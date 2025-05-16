# rl_modules/experience_replay.py

"""
Experience Replay Buffer for MOF-RL training.

This module provides memory-efficient experience replay functionality 
for reinforcement learning of MOF generation.
"""

import torch
import random
import numpy as np
from collections import deque


class ExperienceReplayBuffer:
    """
    Memory-optimized experience replay buffer for storing and sampling past generations.
    Stores tensors on CPU to save GPU memory and implements efficient buffer management.
    """
    def __init__(self, max_size=1000, reward_threshold=0.0, device='cuda', 
                prioritized_replay=True, alpha=0.6, beta=0.4):
        """
        Initialize experience replay buffer
        
        Args:
            max_size: Maximum number of experiences to store
            reward_threshold: Minimum reward for experiences to be stored
            device: Device to use ('cuda' or 'cpu')
            prioritized_replay: Whether to use prioritized experience replay
            alpha: Prioritization exponent (0 = uniform sampling, 1 = fully prioritized)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
        """
        self.buffer = []
        self.max_size = max_size
        self.reward_threshold = reward_threshold
        self.device = device
        
        # Prioritized replay parameters
        self.prioritized_replay = prioritized_replay
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros(max_size, dtype=np.float32)
        self.position = 0
        self.is_full = False
        
        print(f"Initialized Experience Replay Buffer:")
        print(f"  - Max size: {max_size}")
        print(f"  - Reward threshold: {reward_threshold}")
        print(f"  - Prioritized replay: {prioritized_replay}")
        if prioritized_replay:
            print(f"  - Alpha: {alpha}")
            print(f"  - Beta: {beta}")
        
    def add_experience(self, tokens, reward):
        """
        Add a new experience to the buffer if it exceeds the reward threshold
        
        Args:
            tokens: Token sequence (tensor)
            reward: Reward value (float)
            
        Returns:
            True if experience was added, False otherwise
        """
        # Only add experiences that meet the minimum reward threshold
        if reward >= self.reward_threshold:
            # Clone tokens to avoid reference issues and move to CPU to save GPU memory
            tokens_clone = tokens.clone().detach().cpu()
            
            if self.prioritized_replay:
                # For prioritized replay, we always add the experience
                # and calculate its priority based on the reward
                
                # Calculate priority (reward + small epsilon to ensure non-zero)
                priority = (reward + 0.01) ** self.alpha
                
                if not self.is_full:
                    # Buffer not full yet, simply append
                    self.buffer.append((tokens_clone, reward))
                    self.priorities[self.position] = priority
                    self.position = (self.position + 1) % self.max_size
                    if self.position == 0:
                        self.is_full = True
                    return True
                else:
                    # Buffer full, replace element at current position
                    self.buffer[self.position] = (tokens_clone, reward)
                    self.priorities[self.position] = priority
                    self.position = (self.position + 1) % self.max_size
                    return True
            else:
                # For uniform sampling, keep higher reward experiences
                # If buffer is full, remove the lowest-reward experience
                if len(self.buffer) >= self.max_size:
                    # Find the lowest reward in the buffer
                    min_reward_idx = min(range(len(self.buffer)), key=lambda i: self.buffer[i][1])
                    
                    # Only replace if this new reward is higher
                    if reward > self.buffer[min_reward_idx][1]:
                        self.buffer.pop(min_reward_idx)
                        self.buffer.append((tokens_clone, reward))
                        return True
                    return False
                
                # If buffer isn't full yet, just append
                self.buffer.append((tokens_clone, reward))
                return True
                
        return False
        
    def sample_batch(self, batch_size):
        """
        Sample a batch of experiences from the buffer
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            tokens: List of token sequences
            rewards: List of corresponding rewards
            weights: List of importance sampling weights (for prioritized replay)
        """
        if len(self.buffer) == 0:
            return [], [], []
            
        effective_size = len(self.buffer) if not self.prioritized_replay else self.max_size if self.is_full else self.position
        
        if self.prioritized_replay:
            # Prioritized experience replay
            
            # Get valid priorities (for filled buffer elements)
            if self.is_full:
                priorities = self.priorities
            else:
                priorities = self.priorities[:self.position]
            
            # Calculate sampling probabilities
            probabilities = priorities / np.sum(priorities)
            
            # Sample indices based on probabilities
            indices = np.random.choice(
                effective_size, 
                size=min(batch_size, effective_size), 
                replace=False, 
                p=probabilities
            )
            
            # Calculate importance sampling weights
            weights = (effective_size * probabilities[indices]) ** (-self.beta)
            weights /= np.max(weights)  # Normalize weights
            
            # Collect sampled experiences
            tokens = [self.buffer[i][0] for i in indices]
            sampled_rewards = [self.buffer[i][1] for i in indices]
            
            return tokens, sampled_rewards, weights.tolist()
        else:
            # Uniform experience replay with bias toward higher rewards
            
            # Calculate sampling weights based on rewards
            rewards = [exp[1] for exp in self.buffer]
            
            # Add a small constant to avoid zero probabilities
            weights = [max(r, 0.01) for r in rewards]
            
            # Normalize weights
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            
            # Sample indices based on probabilities
            indices = random.choices(range(len(self.buffer)), weights=probs, k=min(batch_size, len(self.buffer)))
            
            # Collect sampled experiences 
            tokens = [self.buffer[i][0] for i in indices]
            sampled_rewards = [self.buffer[i][1] for i in indices]
            
            # Return uniform weights (since this is not prioritized replay)
            weights = [1.0] * len(indices)
            
            return tokens, sampled_rewards, weights
    
    def update_beta(self, fraction):
        """
        Update beta parameter for importance sampling
        
        Args:
            fraction: Fraction of training complete (0 to 1)
        """
        # Anneal beta from initial value to 1.0
        self.beta = min(1.0, self.beta + fraction * (1.0 - self.beta))
    
    def get_stats(self):
        """Return statistics about the buffer"""
        if not self.buffer:
            return {"size": 0, "mean_reward": 0, "max_reward": 0, "min_reward": 0, "memory_usage": 0}
            
        rewards = [exp[1] for exp in self.buffer]
        
        # Estimate memory usage
        mem_usage = sum(tokens.element_size() * tokens.nelement() for tokens, _ in self.buffer)
        mem_usage_mb = mem_usage / (1024 * 1024)
        
        return {
            "size": len(self.buffer),
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "memory_usage_mb": mem_usage_mb
        }


class DiversityBuffer:
    """
    Buffer that tracks diversity of generated samples to help prevent mode collapse.
    """
    def __init__(self, max_size=1000, ngram_size=4):
        """
        Initialize diversity buffer
        
        Args:
            max_size: Maximum number of MOF strings to track
            ngram_size: Size of n-grams to use for diversity analysis
        """
        self.buffer = deque(maxlen=max_size)
        self.ngram_size = ngram_size
        self.ngram_counter = {}
        
    def add_mof(self, mof_str):
        """
        Add a MOF string to the diversity buffer
        
        Args:
            mof_str: MOF string to add
        """
        # If buffer is full, remove oldest entry and its n-grams
        if len(self.buffer) == self.buffer.maxlen:
            old_mof = self.buffer[0]
            old_ngrams = self._extract_ngrams(old_mof)
            
            # Update counter (decrement or remove)
            for ng in old_ngrams:
                self.ngram_counter[ng] -= 1
                if self.ngram_counter[ng] == 0:
                    del self.ngram_counter[ng]
        
        # Add new MOF
        self.buffer.append(mof_str)
        
        # Add its n-grams to counter
        new_ngrams = self._extract_ngrams(mof_str)
        for ng in new_ngrams:
            self.ngram_counter[ng] = self.ngram_counter.get(ng, 0) + 1
    
    def _extract_ngrams(self, text):
        """Extract n-grams from text"""
        return [text[i:i+self.ngram_size] for i in range(len(text)-self.ngram_size+1)]
    
    def calculate_diversity_score(self, mof_str):
        """
        Calculate how diverse a MOF is compared to the buffer
        
        Args:
            mof_str: MOF string to evaluate
            
        Returns:
            diversity_score: Score from 0 (very similar) to 1 (very diverse)
        """
        if not self.buffer:
            return 1.0  # First entry is maximally diverse
        
        # Extract n-grams
        ngrams = self._extract_ngrams(mof_str)
        
        if not ngrams:
            return 0.5  # Default score for very short MOFs
        
        # Calculate average frequency of n-grams
        total_freq = 0
        for ng in ngrams:
            freq = self.ngram_counter.get(ng, 0)
            total_freq += freq
        
        avg_freq = total_freq / len(ngrams)
        
        # Convert to diversity score (lower frequency = higher diversity)
        buffer_size = len(self.buffer)
        if avg_freq == 0:
            return 1.0  # Completely novel n-grams
        
        # Normalize by buffer size and invert (so higher is more diverse)
        diversity = 1.0 - (avg_freq / buffer_size)
        
        return diversity
    
    def get_stats(self):
        """Return diversity statistics"""
        if not self.buffer:
            return {"size": 0, "unique_ngrams": 0, "diversity": 0}
        
        return {
            "size": len(self.buffer),
            "unique_ngrams": len(self.ngram_counter),
            "diversity": len(self.ngram_counter) / (sum(self.ngram_counter.values()) / len(self.ngram_counter))
        }


class MultiObjectiveBuffer:
    """
    Advanced buffer for multi-objective reinforcement learning.
    Maintains separate experiences for different objectives and
    supports dynamic sampling based on current training needs.
    """
    def __init__(self, num_objectives, max_size_per_objective=500):
        """
        Initialize buffer with multiple objectives
        
        Args:
            num_objectives: Number of objectives/targets
            max_size_per_objective: Maximum size for each objective buffer
        """
        self.num_objectives = num_objectives
        self.max_size = max_size_per_objective
        
        # Create separate buffers for each objective
        self.buffers = [[] for _ in range(num_objectives)]
        
        # Track overall performance on each objective
        self.objective_scores = [0.0] * num_objectives
        
        # Track success rate for each objective to adjust sampling
        self.objective_weights = [1.0] * num_objectives
    
    def add_experience(self, tokens, rewards):
        """
        Add experience to appropriate objective buffers
        
        Args:
            tokens: Token sequence
            rewards: List of rewards for each objective
            
        Returns:
            List of objectives this experience was added to
        """
        if len(rewards) != self.num_objectives:
            raise ValueError(f"Expected {self.num_objectives} rewards, got {len(rewards)}")
        
        tokens_clone = tokens.clone().detach().cpu()
        added_to = []
        
        # Add to each objective buffer if reward is significant
        for i, reward in enumerate(rewards):
            # Use a dynamic threshold based on this objective's current score
            threshold = max(0.5 * self.objective_scores[i], 0.1)
            
            if reward >= threshold:
                # If buffer is full, replace lowest reward experience
                if len(self.buffers[i]) >= self.max_size:
                    min_idx = min(range(len(self.buffers[i])), key=lambda j: self.buffers[i][j][1])
                    if reward > self.buffers[i][min_idx][1]:
                        self.buffers[i][min_idx] = (tokens_clone, reward)
                        added_to.append(i)
                else:
                    # Just append if not full
                    self.buffers[i].append((tokens_clone, reward))
                    added_to.append(i)
                
                # Update objective score (moving average)
                alpha = 0.1  # Learning rate for score update
                self.objective_scores[i] = (1 - alpha) * self.objective_scores[i] + alpha * reward
        
        return added_to
    
    def sample_batch(self, batch_size):
        """
        Sample a batch across all objectives, with weighting toward
        objectives that are currently performing worse
        
        Args:
            batch_size: Total number of experiences to sample
            
        Returns:
            tokens: List of token sequences
            rewards: List of reward values
            objectives: List of which objective each sample came from
        """
        tokens = []
        rewards = []
        objectives = []
        
        # Calculate inverse scores for sampling (lower score = higher sampling probability)
        if sum(self.objective_scores) > 0:
            inv_scores = [1.0 / (score + 0.1) for score in self.objective_scores]
            total = sum(inv_scores)
            obj_probs = [s / total for s in inv_scores]
        else:
            # Uniform if no scores yet
            obj_probs = [1.0 / self.num_objectives] * self.num_objectives
        
        # Determine how many samples to take from each buffer
        samples_per_obj = {}
        for _ in range(batch_size):
            obj = np.random.choice(self.num_objectives, p=obj_probs)
            samples_per_obj[obj] = samples_per_obj.get(obj, 0) + 1
        
        # Sample from each buffer
        for obj, count in samples_per_obj.items():
            if not self.buffers[obj]:
                continue
                
            # Sample with replacement if we need more than buffer size
            indices = np.random.choice(
                len(self.buffers[obj]), 
                size=min(count, len(self.buffers[obj])), 
                replace=(count > len(self.buffers[obj]))
            )
            
            for idx in indices:
                tokens.append(self.buffers[obj][idx][0])
                rewards.append(self.buffers[obj][idx][1])
                objectives.append(obj)
        
        return tokens, rewards, objectives
    
    def update_weights(self, objective_performances):
        """
        Update objective weights based on recent performance
        
        Args:
            objective_performances: List of performance metrics for each objective
        """
        if len(objective_performances) != self.num_objectives:
            return
            
        # Normalize performances to sum to 1
        total = sum(objective_performances)
        if total > 0:
            normalized = [p / total for p in objective_performances]
            
            # Inverse weighting - focus more on objectives performing worse
            self.objective_weights = [1.0 - p + 0.1 for p in normalized]
            
            # Re-normalize weights
            weight_sum = sum(self.objective_weights)
            self.objective_weights = [w / weight_sum for w in self.objective_weights]
    
    def get_stats(self):
        """Return statistics about the buffer"""
        return {
            "objective_sizes": [len(b) for b in self.buffers],
            "objective_scores": self.objective_scores,
            "objective_weights": self.objective_weights
        }
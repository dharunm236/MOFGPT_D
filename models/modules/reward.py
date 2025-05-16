# rl_modules/reward.py

"""
Reward functions for MOF-RL training.

This module provides both the original reward function and an improved
version that addresses target prediction accuracy and mode collapse issues.
"""

import math
import numpy as np
from collections import Counter
from rdkit import Chem
import torch
import random

def check_for_existence(curr_mof_list, all_mof_list):
    """Check if MOFs already exist in the dataset (novelty check)"""
    all_mof_set = set(all_mof_list)
    novel_mof_list = []
    bool_val = []
    for mof in curr_mof_list:
        if mof not in all_mof_set:
            novel_mof_list.append(mof)
            bool_val.append(int(1))
        else:
            bool_val.append(int(0))
    return novel_mof_list, bool_val


def verify_topology(curr_mof_with_topology, topology_labels_key, sep_token):
    """Verify if a MOF has a valid topology token"""
    valid_mof = False
    topology_labels_key_set = set(topology_labels_key)
    if sep_token not in curr_mof_with_topology:
        return valid_mof
    curr_mof = curr_mof_with_topology.split(sep_token)[0]
    topology_token = curr_mof_with_topology.split(sep_token)[1]
    if topology_token in topology_labels_key_set:
        valid_mof = True
    return valid_mof


def verify_rdkit(curr_mofs, metal_atom_list, generation_config, training_config, topology_labels_key=None):
    """Verify MOFs using RDKit and check for valid structures"""
    valid_mofs = []
    bool_val = []
    
    for curr_mof in curr_mofs:
        valid_mof = True
        valid_organic_compound = 0
        valid_inorganic_compound = 0

        if training_config["topology_verification"]: 
            valid_topology = verify_topology(curr_mof_with_topology=curr_mof,
                                            topology_labels_key=topology_labels_key,
                                            sep_token=generation_config["sep_token"])
            valid_mof = valid_topology
            mof_part = curr_mof.split(generation_config["sep_token"])[0]
            compounds = mof_part.split(".")            
        else:
            compounds = curr_mof.split(".")
            
        if not training_config["topology_verification"] or valid_topology:
            for compound in compounds:
                found_metal_atom = False
                for metal_atom in metal_atom_list:
                    if metal_atom in compound:
                        found_metal_atom = True
                        valid_inorganic_compound += 1
                        break
                
                if not found_metal_atom:
                    # check if this is a valid compound
                    mol = Chem.MolFromSmiles(compound)
                    if mol is None:
                        valid_mof = False
                        bool_val.append(int(0))
                        break
                    else:
                        valid_organic_compound += 1
                        
        if valid_mof:
            if valid_inorganic_compound > 0 and valid_organic_compound > 0:
                valid_mofs.append(curr_mof)
                bool_val.append(int(1))
        else:
            bool_val.append(int(0))
            
    return valid_mofs, bool_val


def calculate_diversity_score(mof_list, ngram_size=4):
    """
    Calculate diversity score based on n-gram frequency analysis
    
    Args:
        mof_list: List of MOF strings
        ngram_size: Size of character n-grams to analyze (default: 4)
        
    Returns:
        diversity_score: Normalized diversity score (0-1)
        bool_val: Binary value for each MOF indicating if it's diverse
    """
    if not mof_list or len(mof_list) <= 1:
        return 0.0, [0] * len(mof_list)
    
    # Extract n-grams from all MOFs
    all_ngrams = []
    for mof in mof_list:
        # Create character n-grams
        ngrams = [mof[i:i+ngram_size] for i in range(len(mof)-ngram_size+1)]
        all_ngrams.extend(ngrams)
    
    # Count n-gram frequencies
    ngram_counts = Counter(all_ngrams)
    
    # Calculate diversity score for each MOF
    diversity_scores = []
    for mof in mof_list:
        if len(mof) < ngram_size:
            # If MOF is shorter than n-gram size, give minimal diversity score
            diversity_scores.append(0.1)
            continue
            
        # Calculate n-grams for this MOF
        mof_ngrams = [mof[i:i+ngram_size] for i in range(len(mof)-ngram_size+1)]
        
        # Calculate average frequency of this MOF's n-grams in the entire set
        avg_freq = sum(ngram_counts[ng] for ng in mof_ngrams) / len(mof_ngrams)
        
        # Convert to diversity score (lower frequency = higher diversity)
        # Normalize by total number of MOFs
        diversity = 1.0 / (avg_freq / len(mof_list))
        
        # Scale to 0-1 range with a soft cap
        diversity = min(1.0, diversity)
        
        diversity_scores.append(diversity)
    
    # Calculate binary diversity values based on threshold
    diversity_threshold = 0.5  # Adjust as needed
    bool_val = [1 if score >= diversity_threshold else 0 for score in diversity_scores]
    
    # Return average diversity score
    avg_diversity = sum(diversity_scores) / len(diversity_scores)
    
    return avg_diversity, bool_val


class RewardFunction:
    """
    Enhanced reward function for MOF-RL with better target matching and diversity promotion.
    This class provides an improved reward calculation that helps prevent mode collapse
    and encourages models to generate MOFs that better match target properties.
    """
    
    def __init__(self, novelty_factor, validity_factor, diversity_factor,
                 num_targets=1, target_values=None, target_weights=None, 
                 optimization_modes=None, reward_tolerance=0.05):
        """
        Initialize the improved reward function
        
        Args:
            novelty_factor: Weight for novelty rewards
            validity_factor: Weight for validity rewards
            diversity_factor: Weight for diversity rewards
            num_targets: Number of property targets
            target_values: List of target property values
            target_weights: List of weights for each target
            optimization_modes: List of optimization modes ('higher' or 'lower')
            reward_tolerance: Tolerance for target proximity rewards
        """
        self.novelty_factor = novelty_factor
        self.validity_factor = validity_factor
        self.diversity_factor = diversity_factor
        self.num_targets = num_targets
        
        # Set defaults for optional parameters
        self.target_values = target_values or [0.0] * num_targets
        self.target_weights = target_weights or [1.0] * num_targets
        self.optimization_modes = optimization_modes or ["higher"] * num_targets
        self.reward_tolerance = reward_tolerance
        
        # Track the history of generated MOFs for diversity calculation
        self.generation_history = set()
        self.max_history_size = 500
        
        print(f"Initialized Improved Reward Function:")
        print(f"  - Novelty factor: {self.novelty_factor}")
        print(f"  - Validity factor: {self.validity_factor}")
        print(f"  - Diversity factor: {self.diversity_factor}")
        for i in range(self.num_targets):
            print(f"  - Target {i+1}: value={self.target_values[i]}, weight={self.target_weights[i]}, mode={self.optimization_modes[i]}")
    

    
    def calculate_proximity_reward(self, pred_val, target_idx):
        """
        Improved proximity reward calculation with better guidance toward targets
        
        Args:
            pred_val: Predicted property value
            target_idx: Index of the target property
            
        Returns:
            proximity_reward: Reward value based on proximity to target
        """
        target_val = self.target_values[target_idx]
        target_weight = self.target_weights[target_idx]
        opt_mode = self.optimization_modes[target_idx]
        
        # Prevent division by zero
        epsilon = 1e-6
        
        # Calculate normalized deviation from target
        normalized_diff = abs(pred_val - target_val) / (abs(target_val) + epsilon)
        
        # Tolerance window (percentage of target)
        tolerance = self.reward_tolerance
        
        # Define a steeper penalty curve for large deviations
        if normalized_diff <= tolerance:
            # Within tolerance window - high reward
            proximity_factor = 1.0
        elif normalized_diff <= 2 * tolerance:
            # Small deviation - gradual decrease
            t = (normalized_diff - tolerance) / tolerance
            proximity_factor = 1.0 - 0.5 * t
        else:
            # Large deviation - exponential penalty
            proximity_factor = 0.5 * math.exp(-3 * (normalized_diff - 2 * tolerance))
        
        # Apply direction-specific adjustments based on optimization mode
        direction_bonus = 1.0
        
        if opt_mode == "higher":
            if pred_val > target_val:
                # Above target in "higher" mode - small bonus decreasing with distance
                excess = (pred_val - target_val) / (target_val + epsilon)
                direction_bonus = 1.0 + 0.2 * math.exp(-2 * excess)
            else:
                # Below target in "higher" mode - apply penalty
                deficit = (target_val - pred_val) / (target_val + epsilon)
                direction_bonus = 1.0 - 0.5 * min(1.0, deficit)
        else:  # "lower" mode
            if pred_val < target_val:
                # Below target in "lower" mode - small bonus decreasing with distance
                deficit = (target_val - pred_val) / (target_val + epsilon)
                direction_bonus = 1.0 + 0.2 * math.exp(-2 * deficit)
            else:
                # Above target in "lower" mode - apply penalty
                excess = (pred_val - target_val) / (target_val + epsilon)
                direction_bonus = 1.0 - 0.5 * min(1.0, excess)
        
        # Combine factors
        combined_factor = proximity_factor * direction_bonus
        
        # Apply weight and ensure minimum reward
        reward = target_weight * combined_factor
        
        # Ensure reward is positive but can be very small for poor matches
        return max(0.05 * target_weight, reward)


    def calculate_diversity_reward(self, new_mofs):
        """
        Enhanced diversity reward calculation with multiple metrics to prevent mode collapse
        
        Args:
            new_mofs: List of newly generated MOF strings
            
        Returns:
            diversity_rewards: List of diversity rewards for each MOF
            avg_diversity: Average diversity score
            diversity_scores: Binary diversity scores
        """
        # Skip diversity calculation for small batches
        if len(new_mofs) <= 1:
            return [self.diversity_factor] * len(new_mofs), 1.0, [1] * len(new_mofs)
        
        # 1. Within-batch diversity (how different each MOF is from others in current batch)
        batch_diversity_scores = []
        for i, mof1 in enumerate(new_mofs):
            # Calculate average Levenshtein distance to other MOFs in batch
            total_distance = 0
            count = 0
            for j, mof2 in enumerate(new_mofs):
                if i != j:
                    # Simple approximate distance metric (faster than full Levenshtein)
                    # Could replace with Levenshtein if performance allows
                    length_diff = abs(len(mof1) - len(mof2))
                    char_diff = sum(c1 != c2 for c1, c2 in zip(mof1[:100], mof2[:100]))
                    distance = (length_diff + char_diff) / (max(len(mof1), len(mof2)) + 1e-6)
                    total_distance += distance
                    count += 1
            
            # Normalize to 0-1 range
            avg_distance = total_distance / max(1, count)
            batch_diversity_scores.append(min(1.0, avg_distance * 2))  # Scale up but cap at 1.0
        
        # 2. N-gram structural diversity (as in your original implementation)
        ngram_diversity, ngram_scores = calculate_diversity_score(new_mofs, ngram_size=4)
        
        # 3. Historical uniqueness (against previously generated MOFs)
        history_uniqueness_scores = []
        
        # First update similarity counter for existing items in history
        similarity_counter = {}
        for mof in new_mofs:
            # Create structural signature (could be enhanced with chemistry-aware features)
            signature = f"{len(mof)}_{mof[:5]}_{mof[-5:] if len(mof) > 5 else ''}"
            
            # Count similar structures
            if signature in similarity_counter:
                similarity_counter[signature] += 1
            else:
                similarity_counter[signature] = 1
        
        # Calculate history-based scores
        for mof in new_mofs:
            # Check exact matches
            exact_match = mof in self.generation_history
            
            # Check signature similarity
            signature = f"{len(mof)}_{mof[:5]}_{mof[-5:] if len(mof) > 5 else ''}"
            similarity_count = similarity_counter.get(signature, 0)
            
            # Penalize for exact matches and similar structures
            if exact_match:
                # Heavy penalty for exact duplicates
                history_score = 0.1
            else:
                # Moderate penalty for similar structures
                history_score = 1.0 / max(1, similarity_count)
                # Add to history
                self.generation_history.add(mof)
            
            history_uniqueness_scores.append(history_score)
        
        # Maintain limited history size
        if len(self.generation_history) > self.max_history_size:
            # Remove oldest entries (approximation - convert to list, remove oldest, convert back)
            history_list = list(self.generation_history)
            self.generation_history = set(history_list[-self.max_history_size:])
        
        # 4. Composition diversity (checking for variety in elements used)
        composition_scores = []
        for mof in new_mofs:
            # Extract elements/fragments (simple approximation)
            elements = set()
            i = 0
            while i < len(mof):
                if mof[i].isupper():
                    if i + 1 < len(mof) and mof[i+1].islower():
                        elements.add(mof[i:i+2])
                        i += 2
                    else:
                        elements.add(mof[i])
                        i += 1
                else:
                    i += 1
            
            # More elements = more diversity (up to a reasonable limit)
            element_diversity = min(1.0, len(elements) / 10.0)
            composition_scores.append(element_diversity)
        
        # Combine all diversity metrics with weights
        # - 30% batch diversity (difference from other MOFs in batch)
        # - 25% n-gram diversity (structural patterns)
        # - 35% history uniqueness (avoid repeating past generations - also avoid cycling between few high reward generations)
        # - 10% composition diversity (variety of elements used)
        combined_scores = []
        for i in range(len(new_mofs)):
            batch_score = batch_diversity_scores[i] if i < len(batch_diversity_scores) else 0.5
            ngram_score = ngram_scores[i] if i < len(ngram_scores) else 0.5
            history_score = history_uniqueness_scores[i] if i < len(history_uniqueness_scores) else 0.5
            comp_score = composition_scores[i] if i < len(composition_scores) else 0.5
            
            weighted_score = (0.30 * batch_score + 
                            0.25 * ngram_score + 
                            0.35 * history_score + 
                            0.10 * comp_score)
            combined_scores.append(weighted_score)
        
        # Convert to rewards
        diversity_rewards = [self.diversity_factor * score for score in combined_scores]
        
        # Calculate average
        avg_combined_diversity = sum(combined_scores) / len(combined_scores)
        
        # Print additional statistics
        print(f"Diversity breakdown: batch={sum(batch_diversity_scores)/len(batch_diversity_scores):.3f}, "
            f"ngram={ngram_diversity:.3f}, history={sum(history_uniqueness_scores)/len(history_uniqueness_scores):.3f}, "
            f"composition={sum(composition_scores)/len(composition_scores):.3f}")
        
        return diversity_rewards, avg_combined_diversity, combined_scores
    
    def calculate_rewards(self, new_mofs, metals_list, all_mof_strs, predicted_targets,
                         generation_config, training_config, topology_labels_key):
        """
        Calculate comprehensive rewards for generated MOFs
        
        Args:
            new_mofs: List of generated MOF strings
            metals_list: List of valid metal atoms
            all_mof_strs: List of existing MOFs for novelty check
            predicted_targets: List of predicted property values
            generation_config: Configuration for generation
            training_config: Configuration for training
            topology_labels_key: List of valid topology keys
            
        Returns:
            total_rewards: List of total rewards
            novelty_rewards: List of novelty rewards
            validity_rewards: List of validity rewards
            diversity_rewards: List of diversity rewards
            target_rewards_list: List of target-specific rewards
        """
        total_generated = len(new_mofs)
        
        # Check novelty (if MOF exists in dataset)
        novel_mofs, bool_novel = check_for_existence(
            curr_mof_list=new_mofs,
            all_mof_list=all_mof_strs
        )
        
        # Check validity (using RDKit verification)
        valid_mofs, bool_valid = verify_rdkit(
            curr_mofs=new_mofs,
            metal_atom_list=metals_list,
            generation_config=generation_config,
            training_config=training_config,
            topology_labels_key=topology_labels_key
        )
        
        # Ensure consistent list lengths
        if len(bool_valid) != total_generated:
            bool_valid = bool_valid + [False] * (total_generated - len(bool_valid)) if len(bool_valid) < total_generated else bool_valid[:total_generated]
        
        if len(bool_novel) != total_generated:
            bool_novel = bool_novel + [False] * (total_generated - len(bool_novel)) if len(bool_novel) < total_generated else bool_novel[:total_generated]
        
        # Calculate basic rewards with higher weight for valid and novel MOFs
        novelty_rewards = [self.novelty_factor * n for n in bool_novel]
        validity_rewards = [self.validity_factor * v for v in bool_valid]
        
        # Calculate diversity rewards (preventing mode collapse)
        diversity_rewards, avg_diversity, diversity_scores = self.calculate_diversity_reward(new_mofs)
        
        # Calculate target-specific rewards
        target_rewards_list = []
        
        # Process predicted targets if available
        if predicted_targets and len(predicted_targets) > 0:
            # Handle single target case
            if not isinstance(predicted_targets[0], (list, tuple)) and self.num_targets == 1:
                target_rewards = []
                for i, pred_val in enumerate(predicted_targets):
                    # Apply reward only for valid and novel MOFs
                    if i < len(bool_valid) and i < len(bool_novel) and bool_valid[i] and bool_novel[i]:
                        reward = self.calculate_proximity_reward(pred_val, 0)
                        target_rewards.append(reward)
                    else:
                        # Minimal reward for invalid or non-novel MOFs
                        target_rewards.append(0.01 * self.target_weights[0])
                target_rewards_list = [target_rewards]
            
            # Handle multi-target case
            elif isinstance(predicted_targets[0], (list, tuple)):
                for target_idx in range(self.num_targets):
                    target_rewards = []
                    for i, pred_values in enumerate(predicted_targets):
                        # Apply reward only for valid and novel MOFs
                        if i < len(bool_valid) and i < len(bool_novel) and bool_valid[i] and bool_novel[i] and target_idx < len(pred_values):
                            pred_val = pred_values[target_idx]
                            reward = self.calculate_proximity_reward(pred_val, target_idx)
                            target_rewards.append(reward)
                        else:
                            # Minimal reward for invalid or non-novel MOFs
                            target_rewards.append(0.01 * self.target_weights[target_idx])
                    target_rewards_list.append(target_rewards)
        else:
            # No predictions available
            target_rewards_list = [[0.01 * w for _ in range(total_generated)] for w in self.target_weights]
        
        # Apply a multiplicative boost for valid & novel MOFs to encourage both
        combined_validators = [float(n and v) for n, v in zip(bool_novel, bool_valid)]
        
        # Combine all rewards
        total_rewards = []
        
        for i in range(total_generated):
            # Get component rewards
            nov_reward = novelty_rewards[i] if i < len(novelty_rewards) else 0
            val_reward = validity_rewards[i] if i < len(validity_rewards) else 0
            div_reward = diversity_rewards[i] if i < len(diversity_rewards) else 0
            
            # Sum target rewards for this MOF
            tgt_reward = 0
            for target_idx in range(self.num_targets):
                if target_idx < len(target_rewards_list):
                    target_rewards = target_rewards_list[target_idx]
                    if i < len(target_rewards):
                        tgt_reward += target_rewards[i]
            
            # Apply validity+novelty multiplier to target reward
            combined_valid = combined_validators[i] if i < len(combined_validators) else 0
            valid_novel_boost = 1.0 + combined_valid  # 2x boost if both valid & novel
            boosted_tgt_reward = tgt_reward * valid_novel_boost
            
            # Calculate total reward (sum of all components)
            total = nov_reward + val_reward + div_reward + boosted_tgt_reward
            total_rewards.append(total)
        
        # Print statistics
        min_length = min(len(bool_valid), len(bool_novel))
        valid_and_novel = sum(1 for i in range(min_length) if bool_valid[i] and bool_novel[i])
        
        print(f"\n--- Reward Statistics ---")
        print(f"Novelty: {sum(bool_novel)}/{len(bool_novel)} MOFs are novel ({(sum(bool_novel)/max(1, len(bool_novel))*100):.1f}%)")
        print(f"Validity: {sum(bool_valid)}/{len(bool_valid)} MOFs are valid ({(sum(bool_valid)/max(1, len(bool_valid))*100):.1f}%)")
        print(f"Valid & Novel: {valid_and_novel}/{total_generated} MOFs ({(valid_and_novel/total_generated*100):.1f}%)")
        print(f"Diversity score: {avg_diversity:.4f}")
        
        # Print target statistics
        for i in range(self.num_targets):
            if i < len(target_rewards_list):
                rewards = target_rewards_list[i]
                if rewards:
                    avg_reward = sum(rewards) / len(rewards)
                    print(f"Target {i+1}: Avg reward = {avg_reward:.4f}")
                    
                    if predicted_targets and len(predicted_targets) > 0:
                        if isinstance(predicted_targets[0], (list, tuple)):
                            pred_vals = [p[i] if i < len(p) else None for p in predicted_targets]
                        else:
                            pred_vals = predicted_targets if i == 0 else []
                        
                        if pred_vals and len(pred_vals) > 0:
                            # Calculate statistics about predictions
                            valid_preds = [p for p in pred_vals if p is not None]
                            if valid_preds:
                                avg_pred = sum(valid_preds) / len(valid_preds)
                                std_pred = math.sqrt(sum((p - avg_pred)**2 for p in valid_preds) / len(valid_preds)) if len(valid_preds) > 1 else 0
                                min_pred = min(valid_preds)
                                max_pred = max(valid_preds)
                                
                                print(f"  → Target value: {self.target_values[i]:.4f}")
                                print(f"  → Prediction stats: avg={avg_pred:.4f}, std={std_pred:.4f}, range=[{min_pred:.4f}, {max_pred:.4f}]")
                                print(f"  → Samples: {[round(p, 4) for p in valid_preds[:3] if p is not None]}")
        
        avg_total = sum(total_rewards) / max(1, len(total_rewards))
        print(f"Total reward: {avg_total:.4f}")
        print(f"------------------------")
        
        return total_rewards, novelty_rewards, validity_rewards, diversity_rewards, target_rewards_list
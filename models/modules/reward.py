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
    Enhanced reward function for MOF-RL with global memory, curriculum learning,
    and progressive target guidance to prevent mode collapse and improve convergence.
    """
    
    def __init__(self, novelty_factor, validity_factor, diversity_factor,
                 num_targets=1, target_values=None, target_weights=None, 
                 optimization_modes=None, reward_tolerance=0.05):
        """
        Initialize the enhanced reward function
        
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

        # Add these new attributes
        self.target_std = reward_tolerance * 2  # Default target std is 2x tolerance
        self.prediction_history = []  # Track recent predictions
        self.history_size = 100  # Increased history size
        
        # NEW: Global Top-K memory across ALL epochs
        self.global_top_k = []  # List of (mof, prediction, target_score, reward) tuples
        self.max_global_memory = 200  # Keep top 200 MOFs globally
        
        # NEW: Progressive target guidance
        self.current_best_distance = float('inf')
        
        # NEW: Training progress tracking
        self.epoch_count = 0
        self.recent_improvements = []
        self.max_improvements_tracked = 20
        
        print(f"Initialized Enhanced Reward Function:")
        print(f"  - Novelty factor: {self.novelty_factor}")
        print(f"  - Validity factor: {self.validity_factor}")
        print(f"  - Diversity factor: {self.diversity_factor}")
        print(f"  - Global Top-K memory size: {self.max_global_memory}")
        for i in range(self.num_targets):
            print(f"  - Target {i+1}: value={self.target_values[i]}, weight={self.target_weights[i]}, mode={self.optimization_modes[i]}")


    def calculate_proximity_reward(self, pred_val, target_idx):
        """
        Use staged curriculum targets instead of final targets for reward calculation
        """
        # Get current stage-appropriate targets
        current_targets = self.get_current_targets()
        target_val = current_targets[target_idx]  # Use staged target, not final target
        target_weight = self.target_weights[target_idx]
        opt_mode = self.optimization_modes[target_idx]
        
        # Show progress toward final goal
        final_target = self.original_targets[target_idx]
        if self.epoch_count % 20 == 0:  # Print every 20 epochs
            print(f"🎯 Target {target_idx+1} Curriculum: Current={target_val:.4f} → Final={final_target:.4f}")
        
        # Standard proximity calculation with current (staged) target
        base_tolerance = max(0.1, abs(target_val) * 0.2)
        
        abs_diff = abs(pred_val - target_val)
        relative_distance = abs_diff / (abs(target_val) + 1e-6)
        
        # Reward based on staged target
        if relative_distance <= 0.05:
            base_reward = 15.0
        elif relative_distance <= 0.1:
            base_reward = 12.0
        elif relative_distance <= 0.2:
            base_reward = 8.0
        elif relative_distance <= 0.5:
            base_reward = 4.0
        else:
            base_reward = max(1.0, 4.0 * (1.0 - relative_distance))
        
        # Direction bonus
        direction_factor = 1.0
        if opt_mode == "higher":
            if pred_val >= target_val:
                direction_factor = 1.3
            elif pred_val > target_val * 0.8:
                direction_factor = 1.1
            else:
                direction_factor = 0.95
        elif opt_mode == "lower":
            if pred_val <= target_val:
                direction_factor = 1.3
            elif pred_val < target_val * 1.2:
                direction_factor = 1.1
            else:
                direction_factor = 0.95
        
        total_reward = base_reward * direction_factor * target_weight
        return max(0.1 * target_weight, total_reward)
    
    
    
    def update_global_memory(self, mofs, predictions, rewards):
        """
        Fixed global memory update that maintains consistent tuple format
        """
        # Add new candidates to global memory
        for mof, pred, reward in zip(mofs, predictions, rewards):
            # Calculate target-specific score with better handling of distant targets
            target_score = 0
            
            for i in range(self.num_targets):
                if isinstance(pred, (list, tuple)) and i < len(pred):
                    pred_val = pred[i]
                else:
                    pred_val = pred if i == 0 else 0.0
                
                target_val = self.target_values[i]
                opt_mode = self.optimization_modes[i]
                
                # IMPROVED: Progress-based scoring for distant targets
                if abs(target_val) > 0.1:  # Non-zero target
                    # Calculate relative progress
                    if opt_mode == "higher":
                        # For "higher" targets, reward movement toward and above target
                        if pred_val >= target_val:
                            progress = 1.0 + (pred_val - target_val) / abs(target_val)  # Bonus for exceeding
                        else:
                            progress = max(0.1, pred_val / target_val)  # Progress toward target, min 0.1
                    elif opt_mode == "lower":
                        # For "lower" targets, reward movement toward and below target
                        if pred_val <= target_val:
                            progress = 1.0 + (target_val - pred_val) / abs(target_val)  # Bonus for being below
                        else:
                            progress = max(0.1, target_val / pred_val if pred_val > 0 else 0)  # Progress toward target
                    else:
                        # Exact target matching
                        progress = 1.0 / (1.0 + abs(pred_val - target_val) / abs(target_val))
                else:
                    # Zero target - just use inverse distance
                    progress = 1.0 / (1.0 + abs(pred_val))
                
                target_score += progress * self.target_weights[i]
            
            # FIXED: Store with consistent 4-tuple format (mof, pred, target_score, reward)
            self.global_top_k.append((mof, pred, target_score, reward))
        
        # Sort by target score (progress toward goals)
        self.global_top_k.sort(key=lambda x: x[2], reverse=True)
        self.global_top_k = self.global_top_k[:self.max_global_memory]
        
        # Update current best distance with more sophisticated tracking
        if self.global_top_k:
            best_entry = self.global_top_k[0]
            best_pred = best_entry[1]
            best_target_score = best_entry[2]
            
            # Calculate composite distance to all targets
            total_distance = 0
            for i in range(self.num_targets):
                if isinstance(best_pred, (list, tuple)) and i < len(best_pred):
                    pred_val = best_pred[i]
                else:
                    pred_val = best_pred if i == 0 else 0.0
                
                target_val = self.target_values[i]
                
                # Weighted distance
                distance = abs(pred_val - target_val) * self.target_weights[i]
                total_distance += distance
            
            if total_distance < self.current_best_distance:
                improvement = self.current_best_distance - total_distance
                self.recent_improvements.append(improvement)
                if len(self.recent_improvements) > self.max_improvements_tracked:
                    self.recent_improvements = self.recent_improvements[-self.max_improvements_tracked:]
                self.current_best_distance = total_distance
                
                print(f"🎯 NEW GLOBAL BEST!")
                print(f"   Distance to targets: {total_distance:.6f} (improvement: {improvement:.6f})")
                print(f"   Target score: {best_target_score:.4f}")
                
                # Show per-target progress
                for i in range(self.num_targets):
                    if isinstance(best_pred, (list, tuple)) and i < len(best_pred):
                        pred_val = best_pred[i]
                    else:
                        pred_val = best_pred if i == 0 else 0.0
                    
                    target_val = self.target_values[i]
                    print(f"   Target {i+1}: {pred_val:.6f} → {target_val:.6f} (gap: {abs(pred_val-target_val):.6f})")

    def get_current_targets(self):
        """
        STAGED CURRICULUM: Gradually increase targets during training
        Instead of jumping directly to distant targets, build up progressively
        """
        if not hasattr(self, 'original_targets'):
            # Store original targets on first call
            self.original_targets = self.target_values.copy()
        
        # Calculate training progress (0 to 1)
        max_epochs = 400  # Adjust based on your training length
        progress = min(1.0, self.epoch_count / max_epochs)
        
        # Calculate current targets based on progress
        current_targets = []
        
        for i, original_target in enumerate(self.original_targets):
            # Start from a reasonable baseline (e.g., current model's capability)
            if original_target > 0.8:
                # For high targets like 0.96 or 1.41
                starting_point = 0.3  # Start from what model can currently achieve
                gap = original_target - starting_point
                
                # Staged progression with acceleration
                if progress < 0.3:
                    # Stage 1: Easy progress (30% of training)
                    stage_progress = progress / 0.3
                    current_target = starting_point + gap * 0.2 * stage_progress
                elif progress < 0.7:
                    # Stage 2: Medium progress (40% of training)
                    stage_progress = (progress - 0.3) / 0.4
                    current_target = starting_point + gap * (0.2 + 0.4 * stage_progress)
                else:
                    # Stage 3: Final push (30% of training)
                    stage_progress = (progress - 0.7) / 0.3
                    current_target = starting_point + gap * (0.6 + 0.4 * stage_progress)
            else:
                # For lower targets, less aggressive staging
                starting_point = max(0.2, original_target * 0.5)
                gap = original_target - starting_point
                current_target = starting_point + gap * progress
            
            current_targets.append(current_target)
        
        return current_targets



    def get_curriculum_examples(self, batch_size):
        """
        Fixed curriculum examples that handles the correct tuple unpacking
        """
        if not self.global_top_k:
            return [], []
        
        # Calculate how many curriculum examples to include
        if self.epoch_count < 50:
            curriculum_ratio = 0.4  # 40% curriculum examples early on
        elif self.epoch_count < 150:
            curriculum_ratio = 0.3  # 30% mid-training
        else:
            curriculum_ratio = 0.2  # 20% late training
        
        num_examples = min(batch_size // 2, int(len(self.global_top_k) * curriculum_ratio))
        if num_examples == 0:
            return [], []
        
        # FIXED: Handle both old (4-tuple) and new (5-tuple) formats
        curriculum_mofs = []
        curriculum_predictions = []
        
        # Sort global memory by target score (best first)
        sorted_memory = sorted(self.global_top_k, key=lambda x: x[2], reverse=True)
        
        # Take examples from different performance levels to create learning progression
        if len(sorted_memory) >= num_examples:
            # Strategy 1: Take top performers + some intermediate performers
            top_count = max(1, num_examples // 2)
            intermediate_count = num_examples - top_count
            
            # Best performers
            for i in range(min(top_count, len(sorted_memory))):
                entry = sorted_memory[i]
                # Handle both formats: (mof, pred, score, reward) or (mof, pred, score, reward, progress_score)
                mof = entry[0]
                pred = entry[1]
                curriculum_mofs.append(mof)
                curriculum_predictions.append(pred)
            
            # Intermediate performers (create stepping stones)
            if intermediate_count > 0 and len(sorted_memory) > top_count:
                step_size = max(1, (len(sorted_memory) - top_count) // intermediate_count)
                for i in range(intermediate_count):
                    idx = top_count + i * step_size
                    if idx < len(sorted_memory):
                        entry = sorted_memory[idx]
                        mof = entry[0]
                        pred = entry[1]
                        curriculum_mofs.append(mof)
                        curriculum_predictions.append(pred)
        else:
            # Take all available examples
            for entry in sorted_memory[:num_examples]:
                mof = entry[0]
                pred = entry[1]
                curriculum_mofs.append(mof)
                curriculum_predictions.append(pred)
        
        if curriculum_mofs:
            print(f"🎓 Curriculum Learning Active:")
            print(f"  - Adding {len(curriculum_mofs)} examples from global memory")
            
            # Show progression info
            if curriculum_predictions:
                if isinstance(curriculum_predictions[0], (list, tuple)):
                    for target_idx in range(self.num_targets):
                        target_val = self.target_values[target_idx]
                        preds = [p[target_idx] if target_idx < len(p) else 0 for p in curriculum_predictions]
                        if preds:
                            best_pred = max(preds) if self.optimization_modes[target_idx] == "higher" else min(preds)
                            avg_pred = sum(preds) / len(preds)
                            print(f"  - Target {target_idx+1}: Best={best_pred:.4f}, Avg={avg_pred:.4f}, Goal={target_val:.4f}")
                else:
                    target_val = self.target_values[0]
                    best_pred = max(curriculum_predictions) if self.optimization_modes[0] == "higher" else min(curriculum_predictions)
                    avg_pred = sum(curriculum_predictions) / len(curriculum_predictions)
                    print(f"  - Target: Best={best_pred:.4f}, Avg={avg_pred:.4f}, Goal={target_val:.4f}")
        
        return curriculum_mofs, curriculum_predictions


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
        
        return diversity_rewards, avg_combined_diversity, combined_scores

    def calculate_rewards(self, new_mofs, metals_list, all_mof_strs, predicted_targets,
                     generation_config, training_config, topology_labels_key):
        """
        BALANCED: Stable reward calculation for all target distances
        """
        self.epoch_count += 1
        
        # Track predictions for distribution analysis
        if predicted_targets and len(predicted_targets) > 0:
            flat_preds = []
            if isinstance(predicted_targets[0], (list, tuple)):
                for p in predicted_targets:
                    if p and len(p) > 0:
                        flat_preds.extend(p)
            else:
                flat_preds = list(predicted_targets)
            
            self.prediction_history.extend(flat_preds)
            if len(self.prediction_history) > self.history_size:
                self.prediction_history = self.prediction_history[-self.history_size:]

        total_generated = len(new_mofs)
        
        # Check novelty and validity (for bonus rewards only)
        novel_mofs, bool_novel = check_for_existence(
            curr_mof_list=new_mofs,
            all_mof_list=all_mof_strs
        )
        
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
        
        # Calculate basic rewards (small bonuses)
        novelty_rewards = [self.novelty_factor * n * 0.1 for n in bool_novel]
        validity_rewards = [self.validity_factor * v * 0.1 for v in bool_valid]
        
        # Calculate diversity rewards (small bonus)
        diversity_rewards, avg_diversity, diversity_scores = self.calculate_diversity_reward(new_mofs)
        diversity_rewards = [r * 0.1 for r in diversity_rewards]
        
        # Calculate target rewards for ALL MOFs
        target_rewards_list = []
        
        if predicted_targets and len(predicted_targets) > 0:
            # Handle single target case
            if not isinstance(predicted_targets[0], (list, tuple)) and self.num_targets == 1:
                target_rewards = []
                for i, pred_val in enumerate(predicted_targets):
                    reward = self.calculate_proximity_reward(pred_val, 0)
                    target_rewards.append(reward)
                target_rewards_list = [target_rewards]
            
            # Handle multi-target case  
            elif isinstance(predicted_targets[0], (list, tuple)):
                for target_idx in range(self.num_targets):
                    target_rewards = []
                    for i, pred_values in enumerate(predicted_targets):
                        if target_idx < len(pred_values):
                            pred_val = pred_values[target_idx]
                            reward = self.calculate_proximity_reward(pred_val, target_idx)
                            target_rewards.append(reward)
                        else:
                            target_rewards.append(0.1 * self.target_weights[target_idx])
                    target_rewards_list.append(target_rewards)
        else:
            # No predictions available
            target_rewards_list = [[0.1 * w for _ in range(total_generated)] for w in self.target_weights]
        
        # Calculate individual target scores
        individual_target_scores = []
        for i in range(total_generated):
            target_score = 0
            for target_idx in range(self.num_targets):
                if target_idx < len(target_rewards_list) and i < len(target_rewards_list[target_idx]):
                    target_score += target_rewards_list[target_idx][i]
            individual_target_scores.append((i, target_score))
        
        # Sort by target reward
        individual_target_scores.sort(key=lambda x: x[1], reverse=True)
        
        # STABLE: Consistent Top-K selection regardless of target distance
        if self.epoch_count < 100:
            top_k_ratio = 0.5  # Top 50%
        elif self.epoch_count < 200:
            top_k_ratio = 0.4  # Top 40%
        else:
            top_k_ratio = 0.3  # Top 30%
        
        top_k = max(3, int(len(new_mofs) * top_k_ratio))
        top_indices = [idx for idx, _ in individual_target_scores[:top_k]]
        
        # Update global memory
        all_scores = [score for _, score in individual_target_scores]
        self.update_global_memory(new_mofs, predicted_targets or [0.0]*len(new_mofs), all_scores)
        
        # CRITICAL: STABLE reward scaling (no distance-based multiplier!)
        base_multiplier = 3.0  # Fixed, reasonable multiplier
        
        total_rewards = []
        for i in range(total_generated):
            if i in top_indices:
                # Top performers: Target reward + small bonuses
                base_target = individual_target_scores[i][1]
                
                # Small validity/novelty multipliers
                validity_multiplier = 1.1 if bool_valid[i] else 1.0
                novelty_multiplier = 1.1 if bool_novel[i] else 1.0
                
                # Stable primary reward
                primary_reward = base_target * base_multiplier * validity_multiplier * novelty_multiplier
                
                # Small bonuses
                bonus = (novelty_rewards[i] + validity_rewards[i] + diversity_rewards[i])
                
                total_reward = primary_reward + bonus
                total_rewards.append(total_reward)
                
            else:
                # Poor performers: Modest target signal
                target_signal = individual_target_scores[i][1] * 0.3
                total_rewards.append(max(0.1, target_signal))
        
        # Print enhanced statistics
        self.print_enhanced_stats(
            bool_valid, bool_novel, predicted_targets, target_rewards_list, 
            top_indices, individual_target_scores, total_rewards, avg_diversity
        )

        # IMPROVED: Less aggressive normalization that preserves more signal
        if total_rewards:
            reward_mean = sum(total_rewards) / len(total_rewards)
            reward_std = (sum((r - reward_mean) ** 2 for r in total_rewards) / len(total_rewards)) ** 0.5
            
            # Only normalize if rewards are extremely large (>100) or have huge variance
            if reward_mean > 100 or reward_std > 50:
                # Prevent division by zero
                if reward_std < 1e-6:
                    reward_std = 1.0
                
                # More conservative normalization: mean=20, std=10
                target_mean = 20.0
                target_std = 10.0
                
                normalized_rewards = []
                for reward in total_rewards:
                    z_score = (reward - reward_mean) / reward_std
                    normalized_reward = target_mean + z_score * target_std
                    normalized_rewards.append(max(0.1, normalized_reward))
                
                print(f"🔧 Reward Normalization Applied:")
                print(f"   Original: mean={reward_mean:.2f}, std={reward_std:.2f}")
                print(f"   Normalized: mean={sum(normalized_rewards)/len(normalized_rewards):.2f}")
                
                total_rewards = normalized_rewards
            else:
                print(f"📊 Rewards stable: mean={reward_mean:.2f}, std={reward_std:.2f} (no normalization needed)")
        
        return total_rewards, novelty_rewards, validity_rewards, diversity_rewards, target_rewards_list
    
    
    def print_enhanced_stats(self, bool_valid, bool_novel, predicted_targets, target_rewards_list, 
                           top_indices, individual_target_scores, total_rewards, avg_diversity):
        """Print enhanced statistics including global memory info"""
        
        print(f"\n=== ENHANCED REWARD STATISTICS (Epoch {self.epoch_count}) ===")
        
        # Basic stats
        total_generated = len(bool_valid)
        valid_count = sum(bool_valid)
        novel_count = sum(bool_novel)
        valid_and_novel = sum(1 for i in range(len(bool_valid)) if bool_valid[i] and bool_novel[i])
        
        print(f"Validity: {valid_count}/{total_generated} ({valid_count/max(1,total_generated)*100:.1f}%)")
        print(f"Novelty: {novel_count}/{total_generated} ({novel_count/max(1,total_generated)*100:.1f}%)")
        print(f"Valid & Novel: {valid_and_novel}/{total_generated} ({valid_and_novel/max(1,total_generated)*100:.1f}%)")
        print(f"Diversity score: {avg_diversity:.4f}")
        
        # Target achievement stats
        if predicted_targets:
            for i in range(self.num_targets):
                target_val = self.target_values[i]
                
                if isinstance(predicted_targets[0], (list, tuple)):
                    preds = [p[i] if i < len(p) else None for p in predicted_targets]
                else:
                    preds = predicted_targets if i == 0 else []
                
                if preds:
                    valid_preds = [p for p in preds if p is not None]
                    if valid_preds:
                        avg_pred = sum(valid_preds) / len(valid_preds)
                        distance = abs(avg_pred - target_val)
                        
                        # Top-K predictions
                        top_preds = [valid_preds[idx] for idx in top_indices if idx < len(valid_preds)]
                        if top_preds:
                            top_avg = sum(top_preds) / len(top_preds)
                            top_distance = abs(top_avg - target_val)
                            
                            print(f"Target {i+1}: {target_val:.6f}")
                            print(f"  All MOFs    - Avg: {avg_pred:.6f}, Distance: {distance:.6f}")
                            print(f"  Top-K MOFs  - Avg: {top_avg:.6f}, Distance: {top_distance:.6f}")
                            if distance > 0:
                                improvement_pct = ((distance - top_distance) / distance * 100)
                                print(f"  Improvement: {improvement_pct:.1f}%")
        
        # Global memory stats
        if self.global_top_k:
            global_best = self.global_top_k[0]
            print(f"Global Best - Score: {global_best[2]:.6f}, Current Distance: {self.current_best_distance:.6f}")
            print(f"Global Memory: {len(self.global_top_k)}/{self.max_global_memory} MOFs")
        
        # Recent improvements
        if self.recent_improvements:
            recent_avg = sum(self.recent_improvements[-10:]) / len(self.recent_improvements[-10:])
            print(f"Recent Improvement Rate: {recent_avg:.6f}")
        
        # Reward distribution debug
        top_rewards = [total_rewards[i] for i in top_indices[:3]]
        bottom_rewards = [total_rewards[i] for i in range(len(total_rewards)) if i not in top_indices][:3]
        if top_rewards and bottom_rewards:
            print(f"Reward range: Top-K={[f'{r:.2f}' for r in top_rewards]}, Others={[f'{r:.2f}' for r in bottom_rewards]}")
            reward_ratio = max(top_rewards)/max(0.001, max(bottom_rewards) if bottom_rewards else 0.001)
            print(f"Reward ratio: {reward_ratio:.1f}x")
        
        print("="*50)

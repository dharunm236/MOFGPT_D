# rl_modules/generation.py

"""
Generation module for MOF-RL.

This module handles generation of novel MOFs using trained language models.
"""

import torch
from tqdm import tqdm
from .model_utils import clear_memory, log_gpu_memory
from .data_utils import process_sequence_to_str


def generate(model,
             tokenizer,
             generation_config,
             training_config,
             device,
             num_return_sequences,
             energy_predictor):
    """
    Non-batched function to generate MOF sequences with multi-target property prediction.
    
    Args:
        model: Base language model
        tokenizer: Tokenizer for MOF strings
        generation_config: Configuration for generation
        training_config: Training configuration
        device: Device to use for computation
        num_return_sequences: Number of sequences to generate
        energy_predictor: Model for predicting target properties
        
    Returns:
        generated_sequences: List of generated token sequences
        generated_scores: List of token scores during generation
        generated_targets: List of predicted target values for each sequence
    """
    model.eval()
    
    # Clear memory before generation
    clear_memory()
    log_gpu_memory("Before non-batched generation")
    
    # Start with BOS token
    input_smiles = tokenizer.bos_token
    token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize_smiles(input_smiles))
    # Pin memory for faster CPU->GPU transfers
    input_tensor = torch.tensor([token_ids], device=device)
    
    # Storage for generated items
    generated_sequences = []
    generated_scores = []
    generated_targets = []

    with torch.no_grad():
        # Adjust for different numbers of sequences
        if num_return_sequences > 1:
            # Generate sequences one by one to avoid shape issues
            print(f"Generating {num_return_sequences} sequences one by one...")
            
            for i in tqdm(range(num_return_sequences), desc="Generating MOFs"):
                # Generate a single sequence with model
                with torch.amp.autocast('cuda', enabled=training_config.get('fp16', False)):
                    generated_sequence = model.generate(
                        inputs=input_tensor,
                        max_length=generation_config["max_seq_len"],
                        do_sample=generation_config["do_sample"],
                        early_stopping=False,
                        temperature=generation_config["temperature"],
                        top_k=generation_config["top_k"],
                        top_p=generation_config.get("top_p", 0.95),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        use_cache=generation_config["use_cache"],
                        return_dict_in_generate=generation_config["return_dict"],
                        output_scores=generation_config["output_scores"]
                    )
                
                # Store the generated sequence
                sequence = generated_sequence['sequences']
                scores = torch.stack(generated_sequence['scores'])
                
                generated_sequences.append(sequence)
                generated_scores.append(scores)
                
                # Create mask for target prediction
                mask_tokens = torch.ones_like(sequence).to(device)
                
                # Predict target values if predictor is available
                if training_config["do_energy_reward"] and energy_predictor is not None:
                    with torch.amp.autocast('cuda', enabled=training_config.get('fp16', False)):
                        energy_predicted = energy_predictor(
                            token_ids=sequence,
                            mask_ids=mask_tokens
                        )
                        
                        # Handle multi-target predictions
                        if hasattr(energy_predicted, 'shape') and len(energy_predicted.shape) > 0:
                            # Convert tensor to list for easier handling
                            if energy_predicted.dim() > 1:
                                energy_list = energy_predicted.squeeze().tolist()
                                # If it's a single prediction with multiple targets
                                if not isinstance(energy_list, list):
                                    energy_list = [energy_list]
                            else:
                                energy_list = [energy_predicted.item()]
                                
                            generated_targets.append(energy_list)
                        else:
                            generated_targets.append([energy_predicted.item()])
                        
                        # Free memory
                        del energy_predicted
                else:
                    # Default to zero if no predictor
                    if training_config.get("num_targets", 1) > 1:
                        generated_targets.append([0.0] * training_config.get("num_targets", 1))
                    else:
                        generated_targets.append([0.0])
        else:
            # For just one sequence, use the original approach
            input_tensor = torch.tensor([token_ids]).to(device)
            
            with torch.amp.autocast('cuda', enabled=training_config.get('fp16', False)):
                generated_sequence = model.generate(
                    inputs=input_tensor,
                    max_length=generation_config["max_seq_len"],
                    do_sample=generation_config["do_sample"],
                    early_stopping=False,
                    temperature=generation_config["temperature"],
                    top_k=generation_config["top_k"],
                    top_p=generation_config.get("top_p", 0.95),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    use_cache=generation_config["use_cache"],
                    return_dict_in_generate=generation_config["return_dict"],
                    output_scores=generation_config["output_scores"]
                )
            
            sequence = generated_sequence['sequences']
            scores = torch.stack(generated_sequence['scores'])
            
            generated_sequences.append(sequence)
            generated_scores.append(scores)
            
            # Create mask for target prediction
            mask_tokens = torch.ones_like(sequence).to(device)
            
            # Predict target
            if training_config["do_energy_reward"] and energy_predictor is not None:
                with torch.amp.autocast('cuda', enabled=training_config.get('fp16', False)):
                    energy_predicted = energy_predictor(
                        token_ids=sequence,
                        mask_ids=mask_tokens
                    )
                    
                    # Process the prediction
                    if hasattr(energy_predicted, 'shape') and len(energy_predicted.shape) > 0:
                        if energy_predicted.dim() > 1:
                            energy_list = energy_predicted.squeeze().tolist()
                            if not isinstance(energy_list, list):
                                energy_list = [energy_list]
                        else:
                            energy_list = [energy_predicted.item()]
                            
                        generated_targets.append(energy_list)
                    else:
                        generated_targets.append([energy_predicted.item()])
            else:
                # Default to zero
                if training_config.get("num_targets", 1) > 1:
                    generated_targets.append([0.0] * training_config.get("num_targets", 1))
                else:
                    generated_targets.append([0.0])
    
    # Clear memory after generation
    clear_memory()
    log_gpu_memory("After non-batched generation")
    
    return generated_sequences, generated_scores, generated_targets


def generate_with_diversity(model,
                           tokenizer,
                           generation_config,
                           training_config,
                           device,
                           num_return_sequences,
                           energy_predictor,
                           diversity_buffer=None):
    """
    Generate MOFs with enhanced diversity to prevent mode collapse
    
    Args:
        model: Base language model
        tokenizer: Tokenizer for MOF strings
        generation_config: Configuration for generation
        training_config: Training configuration
        device: Device to use for computation
        num_return_sequences: Number of sequences to generate
        energy_predictor: Model for predicting target properties
        diversity_buffer: Optional diversity buffer to track n-grams
        
    Returns:
        generated_sequences: List of generated token sequences
        generated_scores: List of token scores during generation
        generated_targets: List of predicted target values for each sequence
        diversity_scores: List of diversity scores for each sequence
    """
    # Start with standard generation
    generated_sequences, generated_scores, generated_targets = generate(
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        training_config=training_config,
        device=device,
        num_return_sequences=num_return_sequences,
        energy_predictor=energy_predictor
    )
    
    # Calculate diversity scores
    diversity_scores = []
    
    # Convert sequences to MOF strings for diversity analysis
    mof_strings = []
    for sequence in generated_sequences:
        mof_str = process_sequence_to_str(
            sequence=sequence,
            tokenizer=tokenizer,
            training_config=training_config,
            generation_config=generation_config
        )
        mof_strings.append(mof_str)
    
    # Use diversity buffer if provided
    if diversity_buffer is not None:
        # Calculate diversity for each MOF
        for mof_str in mof_strings:
            diversity_score = diversity_buffer.calculate_diversity_score(mof_str)
            diversity_scores.append(diversity_score)
            
            # Add to buffer after scoring
            diversity_buffer.add_mof(mof_str)
    else:
        # Simple n-gram-based diversity scoring
        import re
        from collections import Counter
        
        # Extract n-grams from all MOFs
        ngram_size = 4
        all_ngrams = []
        
        for mof in mof_strings:
            # Create character n-grams
            ngrams = [mof[i:i+ngram_size] for i in range(len(mof)-ngram_size+1)]
            all_ngrams.extend(ngrams)
        
        # Count n-gram frequencies
        ngram_counts = Counter(all_ngrams)
        
        # Calculate diversity score for each MOF
        for mof in mof_strings:
            if len(mof) < ngram_size:
                # If MOF is shorter than n-gram size, give minimal diversity score
                diversity_scores.append(0.5)
                continue
                
            # Calculate n-grams for this MOF
            mof_ngrams = [mof[i:i+ngram_size] for i in range(len(mof)-ngram_size+1)]
            
            # Calculate average frequency of this MOF's n-grams in the entire set
            avg_freq = sum(ngram_counts[ng] for ng in mof_ngrams) / max(1, len(mof_ngrams))
            
            # Convert to diversity score (lower frequency = higher diversity)
            # Normalize by total number of MOFs
            diversity = 1.0 / max(1.0, (avg_freq / len(mof_strings)))
            
            # Scale to 0-1 range with a soft cap
            diversity = min(1.0, diversity)
            
            diversity_scores.append(diversity)
    
    return generated_sequences, generated_scores, generated_targets, diversity_scores


def batch_generate(model,
                 tokenizer,
                 generation_config,
                 training_config,
                 device,
                 num_return_sequences,
                 energy_predictor,
                 batch_size=4):
    """
    Batched generation for more efficient MOF generation
    
    Args:
        model: Base language model
        tokenizer: Tokenizer for MOF strings
        generation_config: Configuration for generation
        training_config: Training configuration
        device: Device to use for computation
        num_return_sequences: Total number of sequences to generate
        energy_predictor: Model for predicting target properties
        batch_size: Number of sequences to generate in each batch
        
    Returns:
        all_sequences: List of all generated token sequences
        all_scores: List of all token scores during generation
        all_targets: List of all predicted target values
    """
    model.eval()
    
    # Storage for all generated items
    all_sequences = []
    all_scores = []
    all_targets = []
    
    # Calculate number of batches
    num_batches = (num_return_sequences + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        # Calculate actual batch size for this iteration
        current_batch_size = min(batch_size, num_return_sequences - batch_idx * batch_size)
        
        # Clear memory before each batch
        clear_memory()
        
        # Create batch of input tensors
        input_smiles = tokenizer.bos_token
        token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize_smiles(input_smiles))
        input_tensor = torch.tensor([token_ids] * current_batch_size, device=device)
        
        with torch.no_grad():
            # Generate sequences for this batch
            with torch.amp.autocast('cuda', enabled=training_config.get('fp16', False)):
                generated_sequence = model.generate(
                    inputs=input_tensor,
                    max_length=generation_config["max_seq_len"],
                    do_sample=generation_config["do_sample"],
                    early_stopping=False,
                    temperature=generation_config["temperature"],
                    top_k=generation_config["top_k"],
                    top_p=generation_config.get("top_p", 0.95),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    use_cache=generation_config["use_cache"],
                    return_dict_in_generate=generation_config["return_dict"],
                    output_scores=generation_config["output_scores"]
                )
            
            # Process the batch results
            sequences = generated_sequence['sequences']
            scores = torch.stack(generated_sequence['scores'])
            
            # Create mask for target prediction
            mask_tokens = torch.ones_like(sequences).to(device)
            
            # Predict target values for the batch
            if training_config["do_energy_reward"] and energy_predictor is not None:
                with torch.amp.autocast('cuda', enabled=training_config.get('fp16', False)):
                    energy_predicted = energy_predictor(
                        token_ids=sequences,
                        mask_ids=mask_tokens
                    )
                    
                    # Convert to list of lists for each sample in batch
                    batch_targets = []
                    
                    # Handle different predictor output formats
                    if energy_predicted.dim() > 1:
                        # Multi-target predictions
                        for i in range(current_batch_size):
                            sample_targets = energy_predicted[i].tolist()
                            if not isinstance(sample_targets, list):
                                sample_targets = [sample_targets]
                            batch_targets.append(sample_targets)
                    else:
                        # Single-target predictions
                        for i in range(current_batch_size):
                            batch_targets.append([energy_predicted[i].item()])
            else:
                # Default to zeros if no predictor
                num_targets = training_config.get("num_targets", 1)
                batch_targets = [[0.0] * num_targets for _ in range(current_batch_size)]
            
            # Split batch results into individual samples
            for i in range(current_batch_size):
                all_sequences.append(sequences[i:i+1])  # Keep dimensions consistent with non-batched version
                all_scores.append(scores)  # Note: scores are shared across batch
                all_targets.append(batch_targets[i])
            
            # Free memory for next batch
            del sequences, scores, mask_tokens, generated_sequence
            if training_config["do_energy_reward"] and energy_predictor is not None:
                del energy_predicted
            clear_memory()
    
    return all_sequences, all_scores, all_targets
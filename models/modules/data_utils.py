# rl_modules/data_utils.py

"""
Data utilities for MOF-RL

This module provides utility functions for data processing and analysis.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

# Disable RDKit logging
RDLogger.DisableLog('rdApp.*')


def load_all_mofs(data_config, training_config):
    """
    Load all MOFs from train, validation, and test sets
    
    Args:
        data_config: Data configuration dictionary
        training_config: Training configuration dictionary
        
    Returns:
        all_mof_strs_train: List of training MOFs
        all_mof_strs_val: List of validation MOFs
        all_mof_strs_test: List of test MOFs
        all_mof_strs: Combined list of all MOFs
    """
    print("Loading MOF datasets...")
    
    all_mof_strs = []
    all_mof_strs_train = []
    all_mof_strs_val = []
    all_mof_strs_test = []
    
    # Load datasets
    train_data_pd = pd.read_csv(data_config["train_csv_filename"], header=None)
    val_data_pd = pd.read_csv(data_config["val_csv_filename"], header=None)
    test_data_pd = pd.read_csv(data_config["test_csv_filename"], header=None)
    
    train_data_list = train_data_pd.to_numpy().tolist()
    val_data_list = val_data_pd.to_numpy().tolist()
    test_data_list = test_data_pd.to_numpy().tolist()

    # Process datasets
    if training_config["topology_verification"]:
        for row in train_data_list:
            all_mof_strs.append(row[0])
            all_mof_strs_train.append(row[0])
        for row in val_data_list:
            all_mof_strs.append(row[0])
            all_mof_strs_val.append(row[0])
        for row in test_data_list:
            all_mof_strs.append(row[0])
            all_mof_strs_test.append(row[0])
    else:
        for row in train_data_list:
            all_mof_strs.append(row[0].split("&&")[0])
            all_mof_strs_train.append(row[0].split("&&")[0])
        for row in val_data_list:
            all_mof_strs.append(row[0].split("&&")[0])
            all_mof_strs_val.append(row[0].split("&&")[0])
        for row in test_data_list:
            all_mof_strs.append(row[0].split("&&")[0])
            all_mof_strs_test.append(row[0].split("&&")[0])
    
    # Clean up pandas objects to free memory
    del train_data_pd, val_data_pd, test_data_pd
    del train_data_list, val_data_list, test_data_list
    
    print(f"Number of MOFs: {len(all_mof_strs)}")
    print(f"Number of MOFs in train set: {len(all_mof_strs_train)}")
    print(f"Number of MOFs in val set: {len(all_mof_strs_val)}")
    print(f"Number of MOFs in test set: {len(all_mof_strs_test)}")
    
    return all_mof_strs_train, all_mof_strs_val, all_mof_strs_test, all_mof_strs


def check_for_existence(curr_mof_list, all_mof_list):
    """
    Check if MOFs already exist in the dataset (novelty check)
    
    Args:
        curr_mof_list: List of MOFs to check
        all_mof_list: List of all known MOFs
        
    Returns:
        novel_mof_list: List of novel MOFs
        bool_val: Boolean list indicating novelty for each MOF
    """
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
    """
    Verify if a MOF has a valid topology token
    
    Args:
        curr_mof_with_topology: MOF string with topology
        topology_labels_key: List of valid topology keys
        sep_token: Separator token between MOF and topology
        
    Returns:
        valid_mof: Boolean indicating if topology is valid
    """
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
    """
    Verify MOFs using RDKit and check for valid structures
    
    Args:
        curr_mofs: List of MOFs to verify
        metal_atom_list: List of valid metal atoms
        generation_config: Generation configuration
        training_config: Training configuration
        topology_labels_key: List of valid topology keys
        
    Returns:
        valid_mofs: List of valid MOFs
        bool_val: Boolean list indicating validity for each MOF
    """
    valid_mofs = []
    bool_val = []
    
    for curr_mof in tqdm(curr_mofs, desc="Verifying MOFs", disable=len(curr_mofs) < 100):
        valid_mof = True
        valid_organic_compound = 0
        valid_inorganic_compound = 0

        if training_config["topology_verification"]: 
            valid_topology = verify_topology(
                curr_mof_with_topology=curr_mof,
                topology_labels_key=topology_labels_key,
                sep_token=generation_config["sep_token"]
            )
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
        else:
            if len(bool_val) < len(curr_mofs):
                bool_val.append(int(0))
            
    # Ensure bool_val has the same length as curr_mofs
    if len(bool_val) < len(curr_mofs):
        bool_val.extend([int(0)] * (len(curr_mofs) - len(bool_val)))
            
    return valid_mofs, bool_val


def process_sequence_to_str(sequence, tokenizer, training_config, generation_config):
    """
    Helper function to convert token sequence to MOF string
    
    Args:
        sequence: Token sequence tensor
        tokenizer: Tokenizer for MOF strings
        training_config: Training configuration
        generation_config: Generation configuration
        
    Returns:
        sequence_str: MOF string
    """
    # Convert to CPU for string processing
    seq_cpu = sequence.cpu().numpy().reshape(-1)
    sequence_list = tokenizer.convert_ids_to_tokens(list(seq_cpu))
    
    if training_config["topology_verification"] and generation_config["sep_token"] not in sequence_list:
        sequence_str = ''.join(sequence_list).replace("[PAD]", "").replace("[BOS]", "").replace("[MASK]", "").replace("[UNK]", "").replace("[EOS]", "").strip()
    else:
        separator_index = sequence_list.index(generation_config["sep_token"]) if generation_config["sep_token"] in sequence_list else -1
        
        if separator_index >= 0:
            num_topology_tokens = len(sequence_list) - 1 - separator_index - 1
            modified_sequence_list = sequence_list[:separator_index + 1]
            
            for j in range(num_topology_tokens):
                modified_sequence_list.append(sequence_list[separator_index + 1 + j])
                if j >= num_topology_tokens - 1:
                    continue
                elif j == num_topology_tokens - 2:
                    modified_sequence_list.append(".")
                else:
                    modified_sequence_list.append(",")
                    
            modified_sequence_list.append(sequence_list[-1])
            sequence_str = ''.join(modified_sequence_list).replace("[PAD]", "").replace("[BOS]", "").replace("[MASK]", "").replace("[UNK]", "").replace("[EOS]", "").strip()
        else:
            sequence_str = ''.join(sequence_list).replace("[PAD]", "").replace("[BOS]", "").replace("[MASK]", "").replace("[UNK]", "").replace("[EOS]", "").strip()
    
    # Clean up
    del seq_cpu, sequence_list
    return sequence_str


def analyze_mof_properties(mof_strings, property_values=None):
    """
    Analyze structural properties of MOFs
    
    Args:
        mof_strings: List of MOF strings
        property_values: Optional list of property values
        
    Returns:
        stats: Dictionary with MOF statistics
    """
    if not mof_strings:
        return {"count": 0}
    
    # Basic counts
    num_mofs = len(mof_strings)
    
    # Analyze MOF lengths
    mof_lengths = [len(mof) for mof in mof_strings]
    avg_length = sum(mof_lengths) / num_mofs
    min_length = min(mof_lengths)
    max_length = max(mof_lengths)
    
    # Analyze frequency of metal atoms
    metal_counts = {}
    for mof in mof_strings:
        components = mof.split(".")
        for comp in components:
            # Extract metal atoms (assuming they start with uppercase and may have lowercase)
            import re
            atoms = re.findall(r'[A-Z][a-z]?', comp)
            for atom in atoms:
                metal_counts[atom] = metal_counts.get(atom, 0) + 1
    
    # Calculate most common metals
    sorted_metals = sorted(metal_counts.items(), key=lambda x: x[1], reverse=True)
    most_common_metals = sorted_metals[:5] if sorted_metals else []
    
    # Calculate property statistics if provided
    property_stats = {}
    if property_values is not None and len(property_values) == num_mofs:
        property_stats = {
            "mean": np.mean(property_values),
            "std": np.std(property_values),
            "min": np.min(property_values),
            "max": np.max(property_values),
            "median": np.median(property_values)
        }
    
    # Compile all statistics
    stats = {
        "count": num_mofs,
        "avg_length": avg_length,
        "min_length": min_length,
        "max_length": max_length,
        "most_common_metals": most_common_metals,
        "property_stats": property_stats
    }
    
    return stats


def calculate_fingerprints(mof_strings):
    """
    Calculate molecular fingerprints for MOFs
    
    Args:
        mof_strings: List of MOF strings
        
    Returns:
        fingerprints: List of fingerprint vectors
    """
    from rdkit.Chem import AllChem
    
    fingerprints = []
    
    for mof in mof_strings:
        # Process each compound separately
        compounds = mof.split(".")
        mof_fp = None
        
        for compound in compounds:
            # Skip metal-containing compounds
            if any(atom in compound for atom in ["Fe", "Cu", "Zn", "Mg", "Ca", "Na", "K", "Li"]):
                continue
                
            # Calculate fingerprint for organic part
            mol = Chem.MolFromSmiles(compound)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                if mof_fp is None:
                    mof_fp = np.array(fp)
                else:
                    mof_fp = np.maximum(mof_fp, np.array(fp))
        
        if mof_fp is not None:
            fingerprints.append(mof_fp)
        else:
            # Use empty fingerprint if no valid compounds
            fingerprints.append(np.zeros(1024))
    
    return fingerprints


def calculate_diversity(mof_strings, sample_size=1000):
    """
    Calculate diversity of a set of MOFs
    
    Args:
        mof_strings: List of MOF strings
        sample_size: Maximum number of MOFs to sample for efficiency
        
    Returns:
        diversity_score: Overall diversity score
        pairwise_similarities: List of pairwise similarities
    """
    from rdkit import DataStructs
    import random
    
    # If too many MOFs, sample a subset
    if len(mof_strings) > sample_size:
        sampled_mofs = random.sample(mof_strings, sample_size)
    else:
        sampled_mofs = mof_strings
    
    # Calculate fingerprints
    fingerprints = calculate_fingerprints(sampled_mofs)
    
    # Calculate pairwise similarities
    pairwise_similarities = []
    for i in range(len(fingerprints)):
        for j in range(i+1, len(fingerprints)):
            similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            pairwise_similarities.append(similarity)
    
    # Calculate diversity score (1 - average similarity)
    if pairwise_similarities:
        avg_similarity = sum(pairwise_similarities) / len(pairwise_similarities)
        diversity_score = 1.0 - avg_similarity
    else:
        diversity_score = 1.0  # Maximum diversity if only one MOF
    
    return diversity_score, pairwise_similarities


def validate_generated_file(csv_path, model_type, property_name):
    """Validate generated CSV file"""
    try:
        df = pd.read_csv(csv_path)
        if df.empty or 'target_1' not in df.columns:
            print(f"Invalid output from {model_type} model")
            return False
            
        print(f"\n{model_type.upper()} Model {property_name} Stats:")
        print(f"Mean: {df['target_1'].mean():.4f}")
        print(f"Range: {df['target_1'].min():.4f} - {df['target_1'].max():.4f}")
        return True
    except Exception as e:
        print(f"Error validating {csv_path}: {str(e)}")
        return False


def analyze_data(data_file, property_col, output_dir, property_name):
    """Analyze property distribution in training data"""
    data = pd.read_csv(data_file, header=None)
    prop_values = data.iloc[:, property_col]
    
    stats = {
        'mean': prop_values.mean(),
        'std': prop_values.std(),
        'min': prop_values.min(),
        'max': prop_values.max(),
    }
    
    # Calculate target values
    stats.update({
        'mean_plus_1std': stats['mean'] + stats['std'],
        'mean_plus_2std': stats['mean'] + 2*stats['std'],
        'mean_plus_3std': stats['mean'] + 3*stats['std'],
        'mean_minus_1std': max(0, stats['mean'] - stats['std']),
        'mean_minus_2std': max(0, stats['mean'] - 2*stats['std']),
        'mean_minus_3std': max(0, stats['mean'] - 3*stats['std']),
    })
    
    # Save stats
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Plot distribution
    plt.figure(figsize=(10,6))
    sns.histplot(prop_values, kde=True)
    plt.title(f"{property_name} Distribution")
    plt.savefig(f"{output_dir}/data_distribution.png")
    plt.close()
    
    return stats
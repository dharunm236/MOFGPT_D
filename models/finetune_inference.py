import os
import sys
import torch
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import traceback
from collections import defaultdict

sys.path.append("../")
from modules.model_utils import get_model, clear_memory
from tokenizer.mof_tokenizer_gpt import MOFTokenizerGPT
from modules.models import LLMFineTuneModel
from modules.data_utils import check_for_existence
from modules.model_utils import set_seeds

def calculate_diversity(mof_strings):
    """Calculate diversity metrics for generated MOFs"""
    if len(mof_strings) <= 1:
        return {
            'unique_count': len(mof_strings),
            'diversity_ratio': 0.0,
            'unique_smiles': []
        }
    
    unique_smiles = list(set(mof_strings))
    return {
        'unique_count': len(unique_smiles),
        'diversity_ratio': len(unique_smiles) / len(mof_strings),
        'unique_smiles': unique_smiles
    }

def verify_rdkit(mof_strings, metals_list):
    """Verify MOF validity using RDKit"""
    valid_mofs = []
    valid_flags = []
    
    try:
        from rdkit import Chem
        for mof in mof_strings:
            try:
                # Get SMILES part (before topology if present)
                smiles = mof.split("&&")[0] if "&&" in mof else mof
                
                # Replace metals with carbons for RDKit
                temp_smiles = smiles
                for metal in metals_list:
                    temp_smiles = temp_smiles.replace(metal, "C")
                
                mol = Chem.MolFromSmiles(temp_smiles)
                if mol is not None:
                    valid_mofs.append(mof)
                    valid_flags.append(True)
                else:
                    valid_flags.append(False)
            except Exception:
                valid_flags.append(False)
    except ImportError:
        print("RDKit not available - skipping validity checks")
        valid_mofs = mof_strings
        valid_flags = [True] * len(mof_strings)
    
    return valid_mofs, valid_flags

def save_generation_stats(stats_dict, output_dir):
    """Save generation statistics to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "generation_stats.csv")
    pd.DataFrame([stats_dict]).to_csv(stats_path, index=False)
    print(f"Saved generation statistics to {stats_path}")
    return stats_path


def generate_mofs(model, tokenizer, generation_config, training_config,
                 device, metals_list, all_mof_strs, topology_labels_key,
                 energy_predictor, output_filename, num_generations=100,
                 batch_size=20, property_name="Property", **kwargs):
    """
    Generate MOFs while tracking comprehensive statistics
    Returns DataFrame with only MOFs and their property predictions
    """
    print(f"Starting generation of {num_generations} MOFs...")
    
    model.eval()
    results = []
    stats = defaultdict(int)
    
    # Generate in batches
    for batch_idx in tqdm(range(0, num_generations, batch_size), desc="Generating MOFs"):
        current_batch_size = min(batch_size, num_generations - len(results))
        
        # Generate sequences
        input_tokens = tokenizer.tokenize_smiles("[BOS]")
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        input_tensor = torch.tensor([input_ids]).to(device)
        
        try:
            generated = model.llm_network.generate(
                inputs=input_tensor,
                max_length=generation_config.get("max_length", tokenizer.max_len),
                do_sample=True,
                temperature=generation_config.get("temperature", 0.8),
                top_k=generation_config.get("top_k", 50),
                num_return_sequences=current_batch_size,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id
            )
        except Exception as e:
            print(f"Generation error: {str(e)}")
            continue
        
        # Process generated sequences
        mof_strings = []
        for seq in generated:
            tokens = tokenizer.convert_ids_to_tokens(seq.cpu().numpy())
            mof_str = ''.join(tokens).replace("[BOS]", "").replace("[EOS]", "").replace("[PAD]", "").strip()
            if mof_str:  # Only keep non-empty strings
                mof_strings.append(mof_str)
        
        # Verify validity
        valid_mofs, valid_flags = verify_rdkit(mof_strings, metals_list)
        stats['total_generated'] += len(mof_strings)
        stats['valid_count'] += sum(valid_flags)
        
        # Check novelty against training data
        novel_flags = []
        for mof, is_valid in zip(mof_strings, valid_flags):
            is_novel = is_valid and not check_for_existence(mof, all_mof_strs)[1][0]
            novel_flags.append(is_novel)
            if is_novel:
                stats['novel_count'] += 1
        
        # Predict properties for MOFs
        predictions = []
        for mof in mof_strings:
            try:
                # Convert to tokens then ids for prediction
                pred_tokens = tokenizer.tokenize_smiles(mof)
                pred_token_ids = tokenizer.convert_tokens_to_ids(pred_tokens)
                pred_token_tensor = torch.tensor([pred_token_ids]).to(device)
                pred_mask = torch.ones_like(pred_token_tensor).to(device)
                
                # Get prediction from model
                with torch.no_grad():
                    output = model(token_ids=pred_token_tensor, mask_ids=pred_mask)
                    
                # Extract prediction value
                if isinstance(output, torch.Tensor):
                    if output.numel() == 1:
                        predictions.append(output.item())
                    else:
                        predictions.append(output[0].item())
                else:
                    predictions.append(float(output))
                    
            except Exception as e:
                print(f"Prediction error for MOF: {str(e)}")
                predictions.append(0.0)  # Default on error
        
        # Save results with predictions
        for idx, (mof, pred) in enumerate(zip(mof_strings, predictions)):
            results.append({
                'id': len(results) + 1,
                'mof': mof,
                'target_1': pred
            })
    
    # Calculate final statistics
    stats['validity_rate'] = stats['valid_count'] / max(1, stats['total_generated'])
    stats['novelty_rate'] = stats['novel_count'] / max(1, stats['valid_count'])
    
    # Calculate diversity
    all_mof_strings = [r['mof'] for r in results]
    diversity = calculate_diversity(all_mof_strings)
    stats.update({
        'unique_count': diversity['unique_count'],
        'diversity_ratio': diversity['diversity_ratio'],
        'efficiency': stats['novel_count'] / max(1, stats['total_generated'])
    })
    
    # Save statistics
    save_generation_stats(stats, os.path.dirname(output_filename))
    
    # Save only MOF and target to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_filename, index=False)
    
    print("\n=== Generation Statistics ===")
    print(f"Total generated: {stats['total_generated']}")
    print(f"Valid: {stats['valid_count']} ({stats['validity_rate']:.1%})")
    print(f"Novel: {stats['novel_count']} ({stats['novelty_rate']:.1%})")
    print(f"Unique: {stats['unique_count']} (diversity: {stats['diversity_ratio']:.2f})")
    print(f"Overall efficiency: {stats['efficiency']:.1%}")
    
    return df

def save_generation_stats(stats_dict, output_dir, filename="generation_stats.csv"):
    """Save generation statistics to a CSV file in the original simple format"""
    stats_df = pd.DataFrame([stats_dict])
    stats_path = os.path.join(output_dir, filename)
    stats_df.to_csv(stats_path, index=False)
    print(f"Generation statistics saved to: {stats_path}")
    return stats_path



def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate MOFs using fine-tuned model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--config", type=str, default="../config/config_finetune.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--generation-config", type=str, default="../config/rl/config_generation.yaml", 
                       help="Path to generation configuration file")
    parser.add_argument("--output-dir", type=str, default="./pipeline_results/finetune_inference", 
                       help="Directory to save results")
    parser.add_argument("--output-filename", type=str, default=None, 
                       help="Explicit filename for output CSV") 
    parser.add_argument("--num-generations", type=int, default=100, 
                       help="Number of MOFs to generate")
    parser.add_argument("--batch-size", type=int, default=20, 
                       help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=None, 
                       help="Sampling temperature (overrides config)")
    parser.add_argument("--cuda-device", type=int, default=0, 
                       help="CUDA device ID to use")
    parser.add_argument("--property-name", type=str, default="Property", 
                       help="Name of the property for plotting")
    parser.add_argument("--disable-topology", action="store_true",
                       help="Disable topology verification")
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    set_seeds()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("\n" + "="*70)
    print("MOF GENERATION WITH FINE-TUNED MODEL".center(70))
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Generations: {args.num_generations}")
    print(f"Batch size: {args.batch_size}")
    print(f"Property: {args.property_name}")
    print("="*70 + "\n")
    
    # Set device
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # try:
    # Load configs
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    with open(config["model"]["model_config_filename"], "r") as f:
        model_config = yaml.safe_load(f)
    with open(args.generation_config, "r") as f:
        generation_config = yaml.safe_load(f)

    # Get configs
    training_config = config["training"]
    data_config = config["data"]
    base_model_config = model_config["base_model"]
    fine_tune_config = model_config.get("fine_tune", {})
    
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
        use_topology=data_config["tokenizer"].get("use_topology", False),
    )
    print(f"Tokenizer initialized with use_topology={tokenizer.use_topology}")
    
    # Update model configuration
    base_model_config.update({
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "ignore_index": data_config.get("ignore_index", -100),
        "max_position_embeddings": data_config["tokenizer"]["max_seq_len"]
    })
    
    # Initialize model
    print("Initializing model...")
    clear_memory()
    base_model = get_model(config=base_model_config, device=device)
    model = LLMFineTuneModel(
        llm_network=base_model,
        llm_config=base_model_config,
        fine_tune_config=fine_tune_config,
        is_fp16=training_config.get('fp16', False)
    ).to(device)
    
    # Load model weights
    print(f"Loading weights from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Try different possible state dict keys
    loaded = False
    possible_keys = ["model_state_dict", "state_dict", "llm_network.state_dict"]
    for key in possible_keys:
        if key in checkpoint:
            try:
                model.load_state_dict(checkpoint[key])
                loaded = True
                print(f"Model loaded using key: {key}")
                break
            except Exception as e:
                print(f"Failed with key {key}: {str(e)}")
    
    if not loaded:
        raise RuntimeError("Could not load model weights with any known key")
        
    # Override temperature if specified
    if args.temperature is not None:
        generation_config["temperature"] = args.temperature
        print(f"Temperature set to {args.temperature}")
        
    # Load metals list for RDKit validation
    with open(generation_config["metals_list_file"], "r") as f:
        metals_list = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(metals_list)} metals for validation")
    
    # Load existing MOFs for novelty check
    print("\nLoading existing MOFs for novelty check...")
    all_mof_strs = []
    try:
        train_df = pd.read_csv(data_config["train_csv_filename"])
        all_mof_strs.extend(train_df.iloc[:, 0].tolist())
        print(f"Loaded {len(train_df)} training MOFs")
        
        if "val_csv_filename" in data_config:
            val_df = pd.read_csv(data_config["val_csv_filename"])
            all_mof_strs.extend(val_df.iloc[:, 0].tolist())
            print(f"Loaded {len(val_df)} validation MOFs")
        
        if "test_csv_filename" in data_config:
            test_df = pd.read_csv(data_config["test_csv_filename"])
            all_mof_strs.extend(test_df.iloc[:, 0].tolist())
            print(f"Loaded {len(test_df)} test MOFs")
    except Exception as e:
        print(f"Warning: Error loading existing MOFs: {str(e)}")
    
    print(f"Total {len(all_mof_strs)} existing MOFs loaded for novelty checking")
    
    # Determine output filename
    if args.output_filename:
        output_csv = args.output_filename
    else:
        model_name = Path(args.model).stem
        output_csv = os.path.join(args.output_dir, f"{model_name}_generated_{args.num_generations}.csv")
    
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
    
    
    
    # Generate MOFs
    results_df = generate_mofs(
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        device=device,
        metals_list=metals_list,
        all_mof_strs=all_mof_strs,
        output_filename=output_csv,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        property_name=args.property_name,
        training_config=training_config, 
        topology_labels_key=None,      
        energy_predictor=None  
    )
    
    print("\nGeneration complete!")
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    sys.exit(main())

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
    Fixed version with proper token validation to prevent CUDA errors
    """
    print(f"Starting generation of {num_generations} MOFs...")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Max token ID should be: {tokenizer.vocab_size - 1}")
    
    model.eval()
    results = []
    stats = defaultdict(int)
    
    # Generate in batches
    total_batches = (num_generations + batch_size - 1) // batch_size
    print(f"Will process {total_batches} batches")
    
    for batch_idx in tqdm(range(total_batches), desc="Generating MOFs"):
        # Calculate actual batch size for this iteration
        remaining = num_generations - len(results)
        current_batch_size = min(batch_size, remaining)
        
        if current_batch_size <= 0:
            break
            
        print(f"\nBatch {batch_idx + 1}/{total_batches}: generating {current_batch_size} MOFs")
        
        # Generate sequences
        input_tokens = tokenizer.tokenize_smiles("[BOS]")
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        
        # CRITICAL: Validate input token IDs
        if any(id >= tokenizer.vocab_size or id < 0 for id in input_ids):
            print(f"ERROR: Invalid input token IDs: {input_ids}")
            print(f"Max allowed: {tokenizer.vocab_size - 1}")
            continue
            
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
        
        # Process generated sequences with validation
        mof_strings = []
        for i, seq in enumerate(generated):
            # CRITICAL: Validate all generated token IDs
            token_ids = seq.cpu().numpy()
            invalid_tokens = [id for id in token_ids if id >= tokenizer.vocab_size or id < 0]
            
            if invalid_tokens:
                print(f"Sequence {i}: Invalid token IDs found: {invalid_tokens[:5]}...")
                print(f"Max token ID in sequence: {max(token_ids)}")
                print(f"Vocab size: {tokenizer.vocab_size}")
                continue  # Skip this sequence
            
            try:
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                
                # More careful token processing
                clean_tokens = []
                for token in tokens:
                    if token not in ["[BOS]", "[EOS]", "[PAD]"]:
                        clean_tokens.append(token)
                    elif token == "[EOS]":
                        break  # Stop at EOS
                
                mof_str = ''.join(clean_tokens).strip()
                
                # Basic validation
                if mof_str and len(mof_str) > 5:
                    mof_strings.append(mof_str)
                    
            except Exception as e:
                print(f"Token conversion error for sequence {i}: {e}")
                continue
        
        print(f"Generated {len(mof_strings)} valid MOF strings")
        
        if not mof_strings:
            print("No valid MOFs in this batch, continuing...")
            continue
        
        # Verify validity with RDKit
        try:
            valid_mofs, valid_flags = verify_rdkit(mof_strings, metals_list)
            stats['total_generated'] += len(mof_strings)
            stats['valid_count'] += sum(valid_flags)
        except Exception as e:
            print(f"RDKit validation error: {e}")
            valid_flags = [False] * len(mof_strings)
        
        # Check novelty against training data
        novel_flags = []
        for mof, is_valid in zip(mof_strings, valid_flags):
            if not is_valid:
                novel_flags.append(False)
            else:
                try:
                    is_novel = not check_for_existence(mof, all_mof_strs)[1][0]
                    novel_flags.append(is_novel)
                    if is_novel:
                        stats['novel_count'] += 1
                except Exception as e:
                    print(f"Novelty check error: {e}")
                    novel_flags.append(False)
        
        # FIXED: Predict properties with proper token validation
        predictions = []
        for i, mof in enumerate(mof_strings):
            try:
                # Tokenize the MOF string
                pred_tokens = tokenizer.tokenize_smiles(mof)
                pred_token_ids = tokenizer.convert_tokens_to_ids(pred_tokens)
                
                # CRITICAL: Validate prediction token IDs
                invalid_pred_tokens = [id for id in pred_token_ids if id >= tokenizer.vocab_size or id < 0]
                
                if invalid_pred_tokens:
                    print(f"MOF {i}: Invalid prediction token IDs: {invalid_pred_tokens[:3]}...")
                    print(f"MOF string: {mof[:50]}...")
                    predictions.append(0.0)  # Default value
                    continue
                
                # Ensure proper tensor dimensions
                if len(pred_token_ids) == 0:
                    print(f"MOF {i}: Empty token sequence")
                    predictions.append(0.0)
                    continue
                
                # Truncate if too long
                max_len = tokenizer.max_len
                if len(pred_token_ids) > max_len:
                    pred_token_ids = pred_token_ids[:max_len]
                
                pred_token_tensor = torch.tensor([pred_token_ids]).to(device)
                pred_mask = torch.ones_like(pred_token_tensor).to(device)
                
                # Validate tensor dimensions
                if pred_token_tensor.size(1) == 0:
                    print(f"MOF {i}: Empty tensor")
                    predictions.append(0.0)
                    continue
                
                # Get prediction from model
                with torch.no_grad():
                    output = model(token_ids=pred_token_tensor, mask_ids=pred_mask)
                    
                # Extract prediction value
                if isinstance(output, torch.Tensor):
                    if output.numel() == 1:
                        pred_value = output.item()
                    else:
                        pred_value = output[0].item() if len(output) > 0 else 0.0
                else:
                    pred_value = float(output) if output is not None else 0.0
                
                # Validate prediction value
                if torch.isnan(torch.tensor(pred_value)) or torch.isinf(torch.tensor(pred_value)):
                    pred_value = 0.0
                
                predictions.append(pred_value)
                
            except Exception as e:
                print(f"Prediction error for MOF {i}: {str(e)}")
                print(f"MOF: {mof[:50]}...")
                predictions.append(0.0)  # Default on error
        
        # Save results
        for idx, (mof, pred) in enumerate(zip(mof_strings, predictions)):
            results.append({
                'id': len(results) + 1,
                'mof': mof,
                'target_1': pred
            })
        
        print(f"Total results so far: {len(results)}")
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    
    # Calculate final statistics
    stats['validity_rate'] = stats['valid_count'] / max(1, stats['total_generated']) if stats['total_generated'] > 0 else 0
    stats['novelty_rate'] = stats['novel_count'] / max(1, stats['valid_count']) if stats['valid_count'] > 0 else 0
    
    # Calculate diversity
    all_mof_strings = [r['mof'] for r in results]
    diversity = calculate_diversity(all_mof_strings)
    stats.update({
        'unique_count': diversity['unique_count'],
        'diversity_ratio': diversity['diversity_ratio'],
        'efficiency': stats['novel_count'] / max(1, stats['total_generated']) if stats['total_generated'] > 0 else 0
    })
    
    # Save statistics
    try:
        save_generation_stats(stats, os.path.dirname(output_filename))
    except Exception as e:
        print(f"Error saving stats: {e}")
    
    # Save final results
    df = pd.DataFrame(results)
    df.to_csv(output_filename, index=False)
    
    print("\n=== Generation Statistics ===")
    print(f"Total generated: {stats['total_generated']}")
    print(f"Valid: {stats['valid_count']} ({stats['validity_rate']:.1%})")
    print(f"Novel: {stats['novel_count']} ({stats['novelty_rate']:.1%})")
    print(f"Unique: {stats['unique_count']} (diversity: {stats['diversity_ratio']:.2f})")
    print(f"Overall efficiency: {stats['efficiency']:.1%}")
    
    return df


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
    """Main execution function with proper error handling"""
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
    
    try:
        # Load configs
        print("Loading configurations...")
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
        print("Initializing tokenizer...")
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
        print(f"Tokenizer initialized with vocab_size={tokenizer.vocab_size}")
        print(f"Special tokens: BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, PAD={tokenizer.pad_token_id}")
        
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
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found: {args.model}")
            
        checkpoint = torch.load(args.model, map_location=device)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
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
            # Try loading the checkpoint directly
            try:
                model.load_state_dict(checkpoint)
                loaded = True
                print("Model loaded directly from checkpoint")
            except Exception as e:
                print(f"Failed to load directly: {str(e)}")
        
        if not loaded:
            raise RuntimeError("Could not load model weights with any known key")
            
        # Override temperature if specified
        if args.temperature is not None:
            generation_config["temperature"] = args.temperature
            print(f"Temperature set to {args.temperature}")
        
        # Load metals list for RDKit validation
        print("Loading metals list...")
        if not os.path.exists(generation_config["metals_list_file"]):
            print(f"Warning: Metals file not found: {generation_config['metals_list_file']}")
            metals_list = ["Zn", "Cu", "Fe", "Co", "Ni", "Mn", "Cr", "Cd", "Pb"]  # Default metals
        else:
            with open(generation_config["metals_list_file"], "r") as f:
                metals_list = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(metals_list)} metals for validation: {metals_list[:5]}...")
        
        # Load existing MOFs for novelty check
        print("\nLoading existing MOFs for novelty check...")
        all_mof_strs = []
        try:
            if os.path.exists(data_config["train_csv_filename"]):
                train_df = pd.read_csv(data_config["train_csv_filename"])
                all_mof_strs.extend(train_df.iloc[:, 0].tolist())
                print(f"Loaded {len(train_df)} training MOFs")
            
            if "val_csv_filename" in data_config and os.path.exists(data_config["val_csv_filename"]):
                val_df = pd.read_csv(data_config["val_csv_filename"])
                all_mof_strs.extend(val_df.iloc[:, 0].tolist())
                print(f"Loaded {len(val_df)} validation MOFs")
            
            if "test_csv_filename" in data_config and os.path.exists(data_config["test_csv_filename"]):
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
            output_csv = os.path.join(args.output_dir, f"{model_name}_generated.csv") # _{args.num_generations} in filename?
        
        print(f"Output will be saved to: {output_csv}")
        
        # Generate MOFs using debug version
        print("\nStarting MOF generation...")
        results_df = generate_mofs(  # Use debug version
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
        
        # Verify the output file
        if os.path.exists(output_csv):
            df_check = pd.read_csv(output_csv)
            print(f"Output verification: {len(df_check)} rows in CSV")
            print(f"Sample entries:")
            print(df_check.head())
        else:
            print("ERROR: Output file was not created!")

    except Exception as e:
        print(f"\nERROR: {str(e)}")


if __name__ == "__main__":
    sys.exit(main())

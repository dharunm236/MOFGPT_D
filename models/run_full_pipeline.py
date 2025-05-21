import os
import sys
import argparse
import subprocess
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import random
from modules.pipeline_metrics_logger import PipelineMetricsLogger
from modules.model_utils import set_seeds
from modules.data_utils import validate_generated_file, analyze_data


def determine_property_name(dataset_path):
    """Consistently determine property name based on dataset name."""
    dataset_name = os.path.basename(os.path.dirname(dataset_path))
    print(f"Dataset name for property detection: {dataset_name}")
    
    if "QMOF" in dataset_name:
        property_name = "Band Gap (eV)"
        print(f"Detected QMOF dataset - using property name: {property_name}")
        return property_name
    elif "hMOF" in dataset_name:
        # Extract gas type and pressure
        gas_type = "gas"
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
        
        print(f"Detected hMOF dataset - using property name: {property_name}")
        return property_name
    
    # Default if no match
    return "Property"

# def run_comparison(train_data_path, finetune_csv, rl_csvs, stats, output_dir, property_name, property_col):
#     """Run the comparison script to generate normalized plots"""
#     # Build command to run the comparison script
#     rl_files = []
#     rl_names = []
#     for name, csv_path in rl_csvs.items():
#         rl_files.append(csv_path)
#         rl_names.append(name)
    
#     # Create a filtered stats dictionary with only the targets that are used in RL models
#     filtered_stats = {}
#     for target_name in rl_names:
#         if target_name in stats:
#             filtered_stats[target_name] = stats[target_name]
    
#     # Save the filtered stats to a temporary JSON file
#     filtered_stats_path = os.path.join(output_dir, "filtered_stats.json")
#     os.makedirs(os.path.dirname(filtered_stats_path), exist_ok=True)
#     with open(filtered_stats_path, 'w') as f:
#         json.dump(filtered_stats, f)
    
#     # Extract more intelligent property name if possible
#     dataset_name = os.path.basename(os.path.dirname(train_data_path))
#     intelligent_property_name = property_name
    
#     # Try to parse dataset name for more context
#     # Example: hMOF_CH4_0.5_small_mofid_finetune -> CH4 adsorption at 0.5 bar
#     parts = dataset_name.split('_')
#     if len(parts) >= 3:
#         if "CH4" in parts or "CO2" in parts or "N2" in parts:
#             gas = next((p for p in parts if p in ["CH4", "CO2", "N2"]), None)
#             # Look for a potential pressure value
#             pressure_val = None
#             pressure_unit = "bar"
#             for part in parts:
#                 try:
#                     val = float(part)
#                     pressure_val = val
#                     # Adjust unit based on value
#                     if val < 0.1:
#                         pressure_unit = "kPa"
#                     elif val < 1.0:
#                         pressure_unit = "bar"
#                     else:
#                         pressure_unit = "bar"
#                     break
#                 except ValueError:
#                     continue
            
#             if gas and pressure_val is not None:
#                 intelligent_property_name = f"{gas} adsorption at {pressure_val} {pressure_unit}"
#             elif gas:
#                 intelligent_property_name = f"{gas} adsorption"
    
#     cmd = [
#         "python", "compare_mof_results.py",
#         "--original-data", train_data_path,
#         "--finetune-data", finetune_csv,
#         "--property-col", str(property_col),
#         "--property-name", intelligent_property_name,
#         "--output-dir", output_dir,
#         "--stats-json", filtered_stats_path,  # Use the filtered stats
#         "--include-no-finetune"  # Add this flag to always generate both sets of plots
#     ]
    
#     # Add RL files if available
#     if rl_files:
#         cmd.extend(["--rl-data"] + rl_files)
#         cmd.extend(["--rl-names"] + rl_names)
    
#     # Run comparison script
#     if not run_command(
#         cmd,
#         f"Comparison plots generated successfully in {output_dir}",
#         "Failed to generate comparison plots"
#     ):
#         print("Note: Comparison plots could not be generated, but pipeline can continue")
#         return False

#     # Now collect and compare statistics
#     finetune_stats_path = os.path.join(os.path.dirname(finetune_csv), "generation_stats.csv")
#     stats_data = {}
    
#     if os.path.exists(finetune_stats_path):
#         try:
#             finetune_stats = pd.read_csv(finetune_stats_path)
#             stats_data["Fine-tune"] = finetune_stats.iloc[0].to_dict()
#         except Exception as e:
#             print(f"Error reading finetune stats: {str(e)}")
    
#     # Collect RL stats
#     for name, csv_path in rl_csvs.items():
#         rl_stats_path = os.path.join(os.path.dirname(csv_path), "generation_stats.csv")
#         if os.path.exists(rl_stats_path):
#             try:
#                 rl_stats = pd.read_csv(rl_stats_path)
#                 stats_data[f"RL-{name}"] = rl_stats.iloc[0].to_dict()
#             except Exception as e:
#                 print(f"Error reading RL stats for {name}: {str(e)}")
    
#     # If we have statistics, create a comparison table
#     if stats_data:
#         stats_df = pd.DataFrame.from_dict(stats_data, orient='index')
#         comparison_path = os.path.join(output_dir, "generation_stats_comparison.csv")
#         stats_df.to_csv(comparison_path)
#         print(f"Generation statistics comparison saved to: {comparison_path}")
        
#         # Print the comparison
#         print("\nGeneration Statistics Comparison:")
#         print(stats_df)
    
#     return True


def run_comparison(train_data_path, finetune_csv, rl_csvs, stats, output_dir, property_name, property_col):
    """Run the comparison script to generate normalized plots"""
    # Build command to run the comparison script
    rl_files = []
    rl_names = []
    for name, csv_path in rl_csvs.items():
        rl_files.append(csv_path)
        rl_names.append(name)
    
    # Create a filtered stats dictionary with only the targets that are used in RL models
    filtered_stats = {}
    for target_name in rl_names:
        if target_name in stats:
            filtered_stats[target_name] = stats[target_name]
    
    # Save the filtered stats to a temporary JSON file
    filtered_stats_path = os.path.join(output_dir, "filtered_stats.json")
    os.makedirs(os.path.dirname(filtered_stats_path), exist_ok=True)
    with open(filtered_stats_path, 'w') as f:
        json.dump(filtered_stats, f)
    
    # Extract more intelligent property name if possible
    dataset_name = os.path.basename(os.path.dirname(train_data_path))
    intelligent_property_name = property_name
    
    # Try to parse dataset name for more context
    # Example: hMOF_CH4_0.5_small_mofid_finetune -> CH4 adsorption at 0.5 bar
    parts = dataset_name.split('_')
    if len(parts) >= 3:
        if "CH4" in parts or "CO2" in parts:
            gas = next((p for p in parts if p in ["CH4", "CO2"]), None)
            # Look for a potential pressure value
            pressure_val = None
            pressure_unit = "bar"
            for part in parts:
                try:
                    val = float(part)
                    pressure_val = val
                    # Adjust unit based on value
                    # if val < 0.1:
                    #     pressure_unit = "kPa"
                    # elif val < 1.0:
                    #     pressure_unit = "bar"
                    # else:
                    #     pressure_unit = "bar"
                    break
                except ValueError:
                    continue
            
            if gas and pressure_val is not None:
                intelligent_property_name = f"{gas} adsorption at {pressure_val} {pressure_unit}"
            elif gas:
                intelligent_property_name = f"{gas} adsorption"
    
    # Construct the command line arguments
    cmd = [
        "python", "compare_mof_results.py",
        "--original-data", train_data_path,
        "--finetune-data", finetune_csv,
        "--property-col", str(property_col),
        "--property-name", intelligent_property_name,
        "--output-dir", output_dir,
        "--stats-json", filtered_stats_path,  # Use the filtered stats
        "--include-no-finetune"  # Add this flag to always generate both sets of plots
    ]
    
    # Add RL files if available
    if rl_files:
        cmd.extend(["--rl-data"] + rl_files)
        cmd.extend(["--rl-names"] + rl_names)
    
    # Run comparison script
    if not run_command(
        cmd,
        f"Comparison plots generated successfully in {output_dir}",
        "Failed to generate comparison plots"
    ):
        print("Note: Comparison plots could not be generated, but pipeline can continue")
        return False

    # Generate additional statistics comparison - this is still valuable to keep
    # Collect and compare statistics from generation_stats.csv files
    finetune_stats_path = os.path.join(os.path.dirname(finetune_csv), "generation_stats.csv")
    stats_data = {}
    
    if os.path.exists(finetune_stats_path):
        try:
            finetune_stats = pd.read_csv(finetune_stats_path)
            stats_data["Fine-tune"] = finetune_stats.iloc[0].to_dict()
        except Exception as e:
            print(f"Error reading finetune stats: {str(e)}")
    
    # Collect RL stats
    for name, csv_path in rl_csvs.items():
        rl_stats_path = os.path.join(os.path.dirname(csv_path), "generation_stats.csv")
        if os.path.exists(rl_stats_path):
            try:
                rl_stats = pd.read_csv(rl_stats_path)
                stats_data[f"RL-{name}"] = rl_stats.iloc[0].to_dict()
            except Exception as e:
                print(f"Error reading RL stats for {name}: {str(e)}")
    
    # If we have statistics, create a comparison table
    if stats_data:
        stats_df = pd.DataFrame.from_dict(stats_data, orient='index')
        comparison_path = os.path.join(output_dir, "generation_stats_comparison.csv")
        stats_df.to_csv(comparison_path)
        print(f"Generation statistics comparison saved to: {comparison_path}")
        
        # Print the comparison
        print("\nGeneration Statistics Comparison:")
        print(stats_df)
    
    return True


def run_command(cmd, success_msg, error_msg):
    """Helper to run shell commands"""
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(error_msg)
        return False
    print(success_msg)
    return True

def run_finetune(config_path, output_dir, epochs=None, project_name=None):
    """Run fine-tuning and return path to best model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and update config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Update project name and save dir
    if project_name:
        config['project_name'] = project_name
    config['training']['save_dir'] = output_dir
    
    if epochs:
        config['training']['epochs'] = epochs
    
    temp_config = f"{output_dir}/finetune_config.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run training
    if not run_command(
        ["python", "train_finetune.py", "--config_filename", temp_config],
        "Fine-tuning completed successfully",
        "Fine-tuning failed"
    ):
        return None

    # Find best model
    model_files = list(Path(output_dir).glob("*_best.pt"))
    return str(model_files[0]) if model_files else None


def run_inference(model_path, config_path, output_dir, num_generations, model_type="finetune", temperature=None):
    """Run inference with either fine-tuned or RL model"""
    os.makedirs(output_dir, exist_ok=True)
    model_name = Path(model_path).stem
    output_csv = f"{output_dir}/{model_name}_generations.csv"
    
    ## ADDED: Initialize target argument as empty list
    target_arg = []
    ## ADDED: Check if this is an RL model to pass target information
    if model_type == "rl":
        ## ADDED: Find the target_info.json file in the model directory
        model_dir = os.path.dirname(model_path)
        ## ADDED: Create path to target info file
        target_file = os.path.join(model_dir, "target_info.json")
        ## ADDED: Check if the file exists
        if os.path.exists(target_file):
            ## ADDED: Load the target info
            with open(target_file, 'r') as f:
                target_info = json.load(f)
                ## ADDED: Print found target info
                print(f"Found target info for RL model: {target_info}")
                ## ADDED: Add target value as a command line argument
                target_arg = ["--target-value", str(target_info.get("target_value", 0.0))]
                print(f"Using target value: {target_info.get('target_value', 0.0)} for inference")
        ## ADDED: Print warning if target info not found
        else:
            print(f"Warning: No target info found for RL model at {target_file}")
    
    cmd = [
        "python", "finetune_inference.py" if model_type == "finetune" else "rl_inference.py",
        "--model", model_path,
        "--config", config_path,
        "--output-dir", output_dir,
        "--output-filename", output_csv,
        "--num-generations", str(num_generations),
    ]
    
    ## ADDED: Add target arguments if available
    if target_arg:
        cmd.extend(target_arg)
    
    if temperature:
        cmd.extend(["--temperature", str(temperature)])
    if model_type == "rl":
        cmd.extend(["--model-type", "rl"])
    
    if not run_command(
        cmd,
        f"Inference completed - results saved to {output_csv}",
        "Inference failed"
    ):
        return None
    
    return output_csv if os.path.exists(output_csv) else None


def run_rl_training(finetune_model_path, rl_config_path, output_dir, target_value, optimization_mode, target_name, epochs=None, project_name=None):
    """Run RL training for a single target"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create target-specific config
    with open(rl_config_path) as f:
        config = yaml.safe_load(f)
    
    # Update project name if provided
    if project_name:
        config['project_name'] = project_name
    
    # Set target values and optimization mode
    config['training']['target_values'] = [float(target_value)]
    config['training']['optimization_modes'] = [optimization_mode]
    
    ## ADDED: Save the target information to a file for later use during inference
    target_info = {
        "target_name": target_name,
        "target_value": float(target_value),
        "optimization_mode": optimization_mode
    }
    ## ADDED: Create path and save target info
    target_file = os.path.join(output_dir, "target_info.json")
    ## ADDED: Write target info to JSON file
    with open(target_file, 'w') as f:
        json.dump(target_info, f, indent=4)
    ## ADDED: Print confirmation message
    print(f"Saved target info to {target_file} for later use during inference")
    
    # Add the target name as a command-line argument for the wandb_run_name
    temp_config = f"{output_dir}/rl_config.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Create the command with the wandb run name based on target
    cmd = [
        "python", "train_rl.py", 
        "--config", temp_config, 
        "--output-dir", output_dir,
        "--wandb-run-name", f"RL_{target_name}"
    ]
    
    if epochs:
        cmd.extend(["--epochs", str(epochs)])
    
    # Run RL training
    if not run_command(
        cmd,
        "RL training completed successfully",
        "RL training failed"
    ):
        return None

    # Find best model
    model_files = list(Path(output_dir).glob("*_best.pt"))
    return str(model_files[0]) if model_files else None

def generate_comparison_plots(train_data_path, finetune_csv, rl_csvs, stats, output_dir, property_name):
    """Generate comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    original_data = pd.read_csv(train_data_path, header=None)
    finetune_data = pd.read_csv(finetune_csv)
    rl_data = {name: pd.read_csv(path) for name, path in rl_csvs.items()}
    
    # Plot distributions
    plt.figure(figsize=(12,6))
    sns.kdeplot(original_data.iloc[:,1], label='Original Data', fill=True)
    sns.kdeplot(finetune_data['target_1'], label='Fine-tuned Model', fill=True)
    
    colors = ['red', 'green', 'purple', 'orange']
    for i, (name, df) in enumerate(rl_data.items()):
        sns.kdeplot(df['target_1'], label=f'RL - {name}', color=colors[i%len(colors)], fill=True)
    
    plt.title(f"{property_name} Distributions Comparison")
    plt.legend()
    plt.savefig(f"{output_dir}/distributions_comparison.png")
    plt.close()

def run_pipeline_for_dataset(dataset_folder, args):
    """Run the complete pipeline for a single dataset"""
    # Extract project name from folder name
    project_name = f"mofgpt_{os.path.basename(dataset_folder)}"
    print(f"\nProcessing dataset: {project_name}")
    
    property_name = determine_property_name(dataset_folder)
    print(f"Using property name: {property_name}")
    
    # Override args.property_name for this dataset run
    original_property_name = args.property_name
    args.property_name = property_name

    # Create output directory with project name
    output_dir = f"{project_name}_pipeline_results"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize metrics logger
    metrics_dir = os.path.join(output_dir, "metrics")
    metrics_logger = PipelineMetricsLogger(metrics_dir, project_name)
    
    # Update RL config to point to the correct train/val/test files
    with open(args.rl_config) as f:
        rl_config = yaml.safe_load(f)
    
    # Update data paths in RL config to point to this dataset
    rl_config['data']['train_csv_filename'] = os.path.join(dataset_folder, "train.csv")
    rl_config['data']['val_csv_filename'] = os.path.join(dataset_folder, "val.csv")
    rl_config['data']['test_csv_filename'] = os.path.join(dataset_folder, "test.csv")
    
    # Write the updated RL config
    temp_rl_config = f"{output_dir}/temp_rl_config.yaml"
    with open(temp_rl_config, 'w') as f:
        yaml.dump(rl_config, f)
    
    # Also update finetune config to point to this dataset
    with open(args.finetune_config) as f:
        finetune_config = yaml.safe_load(f)
    
    finetune_config['data']['train_csv_filename'] = os.path.join(dataset_folder, "train.csv")
    finetune_config['data']['val_csv_filename'] = os.path.join(dataset_folder, "val.csv")
    finetune_config['data']['test_csv_filename'] = os.path.join(dataset_folder, "test.csv")
    
    temp_finetune_config = f"{output_dir}/temp_finetune_config.yaml"
    with open(temp_finetune_config, 'w') as f:
        yaml.dump(finetune_config, f)
    
    # Step 1: Data Analysis
    train_data_path = rl_config['data']['train_csv_filename']
    print("Analyzing training data...")
    stats = analyze_data(
        train_data_path, 
        args.property_col, 
        f"{output_dir}/analysis", 
        args.property_name
    )
    
    # Step 2: Fine-tuning
    print("\nRunning fine-tuning...")
    with open(temp_finetune_config) as f:
        config = yaml.safe_load(f)
        actual_finetune_epochs = config['training'].get('epochs', 0)
    metrics_logger.start_finetune(actual_finetune_epochs)
    finetune_model = run_finetune(
        temp_finetune_config,
        f"{output_dir}/finetune",
        args.finetune_epochs,
        project_name
    )
    metrics_logger.end_finetune()
    
    if not finetune_model:
        if args.continue_on_failure:
            print("Failed to fine-tune model. Skipping remaining steps for this dataset.")
            return False
        else:
            print("Failed to fine-tune model. Exiting.")
            sys.exit(1)
    
    # After fine-tuning is complete, update model paths in configs
    finetune_model_path = finetune_model
    
    # Update the RL config with path to fine-tuned model
    rl_config["training"]["resume_model_path"] = finetune_model_path
    
    # Also update the model.yaml config
    with open(rl_config["model"]["model_config_filename"]) as f:
        rl_model_config = yaml.safe_load(f)
    
    rl_model_config["fine_tune"]["pretrained_model_path"] = finetune_model_path
    
    # Write the updated configs back
    with open(temp_rl_config, 'w') as f:
        yaml.dump(rl_config, f)
    
    temp_model_config = f"{output_dir}/temp_model_config.yaml"
    with open(temp_model_config, 'w') as f:
        yaml.dump(rl_model_config, f)
    
    # Update the path in RL config to point to our temp model config
    rl_config["model"]["model_config_filename"] = temp_model_config
    with open(temp_rl_config, 'w') as f:
        yaml.dump(rl_config, f)
    
    # Step 3: Fine-tuned Model Inference
    print("\nRunning fine-tuned model inference...")
    metrics_logger.start_finetune_inference(args.num_generations)
    finetune_csv = run_inference(
        finetune_model,
        temp_finetune_config,
        f"{output_dir}/finetune_inference",
        args.num_generations,
        "finetune",
        args.temperature
    )
    metrics_logger.end_finetune_inference()
    
    finetune_inference_success = finetune_csv is not None and validate_generated_file(finetune_csv, "fine-tuned", args.property_name)
    
    if not finetune_inference_success:
        if args.continue_on_failure:
            print("Fine-tuned inference failed. Proceeding to RL training anyway.")
            # Create a placeholder CSV if needed for later stages
            if not finetune_csv:
                print("Creating placeholder inference results from training data...")
                try:
                    train_df = pd.read_csv(train_data_path, header=None)
                    placeholder_df = pd.DataFrame({
                        'id': range(min(len(train_df), args.num_generations)),
                        'mof': train_df.iloc[:args.num_generations, 0].values,
                        'target_1': train_df.iloc[:args.num_generations, args.property_col].values
                    })
                    finetune_csv = f"{output_dir}/finetune_inference/placeholder_generations.csv"
                    os.makedirs(os.path.dirname(finetune_csv), exist_ok=True)
                    placeholder_df.to_csv(finetune_csv, index=False)
                    print(f"Created placeholder at {finetune_csv}")
                except Exception as e:
                    print(f"Error creating placeholder: {str(e)}")
                    if not args.continue_on_failure:
                        return False
        else:
            print("Fine-tuned inference failed. Exiting.")
            sys.exit(1)
    
    # Step 4: RL Training and Inference
    rl_results = {}
    for target in args.targets:
        print(f"\nProcessing target: {target}")
        
        # Get target value
        if target == "custom":
            if not args.custom_target:
                print("No custom target value provided. Skipping.")
                continue
            target_value = args.custom_target
        else:
            target_value = stats[target]

        print(f"DEBUG: Target value used for RL training: {target_value:.6f}")
        
        # Skip invalid targets for minimization
        if args.optimization_mode == "lower" and target_value <= 0:
            print(f"Skipping {target} (invalid for minimization)")
            continue
        
        # RL Training
        print(f"RL Training for target: {target_value:.4f}")
        with open(temp_rl_config) as f:
            config = yaml.safe_load(f)
            actual_rl_epochs = config['training'].get('epochs', 0)
        metrics_logger.start_rl_training(target, actual_rl_epochs)
        rl_model = run_rl_training(
            finetune_model,
            temp_rl_config,
            f"{output_dir}/rl_{target}",
            target_value,
            args.optimization_mode,
            target,  # Pass the target name for WandB run name
            actual_rl_epochs,
            project_name
        )
        metrics_logger.end_rl_training(target)

        if not rl_model:
            print(f"RL training failed for {target}")
            continue
        
        # RL Inference
        print(f"RL Inference for {target}")
        metrics_logger.start_rl_inference(target, args.num_generations)
        rl_csv = run_inference(
            rl_model,
            temp_rl_config,
            f"{output_dir}/rl_{target}_inference",
            args.num_generations,
            "rl",
            args.temperature
        )
        # Find the total_attempts from the generation_stats.csv if it exists
        try:
            stats_path = os.path.join(f"{output_dir}/rl_{target}_inference", "generation_stats.csv")
            if os.path.exists(stats_path):
                stats_df = pd.read_csv(stats_path)
                total_attempts = stats_df.iloc[0].get('total_attempts', None)
            else:
                total_attempts = None
        except Exception as e:
            print(f"Error reading stats: {e}")
            total_attempts = None
            
        # End timing RL inference
        metrics_logger.end_rl_inference(target, total_attempts)

        if rl_csv and validate_generated_file(rl_csv, "RL", args.property_name):
            rl_results[target] = rl_csv
    
    # Step 5: Run the separate comparison script instead of generating plots directly
    if finetune_csv and rl_results and not args.skip_comparison:
        print("\nRunning comparison analysis...")
        run_comparison(
            train_data_path,
            finetune_csv,
            rl_results,
            stats,
            f"{output_dir}/comparison",
            args.property_name,
            args.property_col
        )
        
    metrics_logger.generate_summary_table()
    
    print("\n" + "="*50)
    print(f"PIPELINE COMPLETED FOR {project_name}".center(50))
    print("="*50)
    print(f"Results saved to: {output_dir}")

    args.property_name = original_property_name
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run full MOF pipeline on multiple datasets")
    parser.add_argument("--dataset-folders", nargs="+", required=True,
                       help="List of dataset folders containing train.csv, val.csv, test.csv")
    parser.add_argument("--finetune-config", default="../config/config_finetune.yaml", help="Fine-tuning config")
    parser.add_argument("--rl-config", default="../config/rl/config.yaml", help="RL config")
    parser.add_argument("--generation-config", default="../config/rl/config_generation.yaml", help="Generation config")
    parser.add_argument("--property-col", type=int, default=1, help="Property column index")
    parser.add_argument("--property-name", default="CO2 adsorption", help="Property display name")
    parser.add_argument("--num-generations", type=int, default=10, help="Number of MOFs to generate")
    parser.add_argument("--finetune-epochs", type=int, help="Fine-tuning epochs")
    parser.add_argument("--rl-epochs", type=int, help="RL training epochs")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--targets", nargs="+", default=["mean", "mean_plus_1std"], 
                           choices=["mean", "mean_plus_1std", "mean_plus_2std", "mean_plus_3std",
                                   "mean_minus_1std", "mean_minus_2std", "mean_minus_3std", "custom"],
                           help="Target types to run")
    parser.add_argument("--custom-target", type=float, help="Custom target value")
    parser.add_argument("--optimization-mode", choices=["higher", "lower"], default="higher", 
                           help="Optimization direction")
    parser.add_argument("--continue-on-failure", action="store_true", 
                       help="Continue pipeline even if some steps fail")
    parser.add_argument("--skip-comparison", action="store_true",
                   help="Skip running the comparison analysis")
    
    args = parser.parse_args()
    set_seeds()
    
    print("\n" + "="*50)
    print("STARTING MULTI-DATASET MOF PIPELINE".center(50))
    print("="*50)
    print(f"Processing {len(args.dataset_folders)} datasets")
    
    # Process each dataset
    successful = 0
    failed = 0
    
    for dataset_folder in args.dataset_folders:
        print("\n" + "="*70)
        print(f"PROCESSING DATASET: {dataset_folder}".center(70))
        print("="*70)
        
        if run_pipeline_for_dataset(dataset_folder, args):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "="*50)
    print("MULTI-DATASET PIPELINE SUMMARY".center(50))
    print("="*50)
    print(f"Total datasets: {len(args.dataset_folders)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())



# python run_full_pipeline.py \
#     --dataset-folders ../benchmark_datasets/finetune/hMOF_CH4_0.5_small_mofid_finetune ../benchmark_datasets/finetune/hMOF_CH4_0.05_small_mofid_finetune \
#     --finetune-config ../config/config_finetune.yaml \
#     --rl-config ../config/rl/config.yaml \
#     --targets mean mean_plus_1std mean_plus_2std \
#     --num-generations 100 \
#     --continue-on-failure


# python run_full_pipeline.py \
#     --dataset-folders ../benchmark_datasets/finetune/hMOF_CH4_0.5_small_mofid_finetune ../benchmark_datasets/finetune/hMOF_CH4_0.05_small_mofid_finetune ../benchmark_datasets/finetune/hMOF_CH4_0.9_small_mofid_finetune ../benchmark_datasets/finetune/hMOF_CH4_2.5_small_mofid_finetune ../benchmark_datasets/finetune/hMOF_CH4_4.5_small_mofid_finetune\
#     --finetune-config ../config/config_finetune.yaml \
#     --rl-config ../config/rl/config.yaml \
#     --targets mean mean_plus_1std mean_plus_2std \
#     --num-generations 30 \
#     --continue-on-failure


# python run_full_pipeline.py \
#     --dataset-folders ../benchmark_datasets/QMOF_finetune\
#     --finetune-config ../config/config_finetune.yaml \
#     --rl-config ../config/rl/config.yaml \
#     --targets mean mean_plus_1std mean_plus_2std \
#     --num-generations 30 \
#     --continue-on-failure


# python run_full_pipeline.py \
#     --dataset-folders ../benchmark_datasets/finetune/hMOF_CH4_0.5_small_mofid_finetune ../benchmark_datasets/finetune/hMOF_CH4_0.05_small_mofid_finetune ../benchmark_datasets/finetune/hMOF_CH4_0.9_small_mofid_finetune ../benchmark_datasets/finetune/hMOF_CH4_2.5_small_mofid_finetune ../benchmark_datasets/finetune/hMOF_CH4_4.5_small_mofid_finetune\
#     --finetune-config ../config/config_finetune.yaml \
#     --rl-config ../config/rl/config.yaml \
#     --targets mean mean_plus_1std mean_plus_2std \
#     --num-generations 2 \
#     --continue-on-failure


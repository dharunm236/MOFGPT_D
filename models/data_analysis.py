#!/usr/bin/env python
# data_analysis.py
"""
Data Analysis Module for MOF-RL

This module analyzes MOF datasets to calculate statistics needed for
setting target values in the reinforcement learning process.
Supports analyzing combined datasets (train, validation, test).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import json
import argparse
from pathlib import Path
from yaml import FullLoader


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze MOF dataset distributions and set RL targets")
    parser.add_argument("--config", type=str, default="../config/rl/config.yaml", 
                        help="Path to main configuration file")
    parser.add_argument("--generation-config", type=str, default="../config/rl/config_generation.yaml", 
                        help="Path to generation configuration file")
    parser.add_argument("--property-col", type=int, default=1, 
                        help="Column index for the property to analyze (0-based)")
    parser.add_argument("--property-name", type=str, default="Property", 
                        help="Display name for the property")
    parser.add_argument("--output-dir", type=str, default="./analysis", 
                        help="Directory to save output plots and configs")
    parser.add_argument("--csv-file", type=str, default=None, 
                        help="Path to CSV file with generated MOFs (for comparison)")
    parser.add_argument("--plot-only", action="store_true", 
                        help="Only plot existing results without analyzing data")
    parser.add_argument("--target-types", nargs="+", 
                        choices=["mean", "mean_plus_1std", "mean_plus_2std", "mean_plus_3std", 
                                "mean_minus_1std", "mean_minus_2std", "mean_minus_3std", "custom"],
                        default=["mean", "mean_plus_1std", "mean_plus_2std"], 
                        help="Types of targets to analyze")
    parser.add_argument("--custom-target", type=float, default=None,
                        help="Custom target value (only used if 'custom' in target-types)")
    parser.add_argument("--optimization-mode", type=str, choices=["higher", "lower"],
                        default="higher", help="Optimization mode (higher or lower)")
    parser.add_argument("--combined", action="store_true", default=True,
                        help="Analyze combined statistics from all datasets")
    parser.add_argument("--datasets", nargs="+", default=["train", "val", "test"],
                        help="Datasets to include in combined analysis")
    
    return parser.parse_args()


def load_dataset(file_path, property_col):
    """
    Load a dataset from CSV file
    
    Args:
        file_path: Path to CSV file
        property_col: Column index for the property
        
    Returns:
        Series with property values
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return pd.Series()
    
    # Load data without headers
    data = pd.read_csv(file_path, header=None)
    
    # Extract property values
    if property_col < len(data.columns):
        return data.iloc[:, property_col]
    else:
        print(f"Warning: Property column {property_col} not found in {file_path}")
        return pd.Series()


def calculate_statistics(property_values):
    """
    Calculate statistics for property values
    
    Args:
        property_values: Series with property values
        
    Returns:
        Dictionary with statistics
    """
    # Calculate statistics
    stats = {
        "count": len(property_values),
        "mean": property_values.mean(),
        "median": property_values.median(),
        "std": property_values.std(),
        "min": property_values.min(),
        "max": property_values.max(),
    }
    
    # Calculate potential target values
    stats["mean_plus_1std"] = stats["mean"] + stats["std"]
    stats["mean_plus_2std"] = stats["mean"] + 2 * stats["std"]
    stats["mean_plus_3std"] = stats["mean"] + 3 * stats["std"]
    stats["mean_minus_1std"] = max(0, stats["mean"] - stats["std"])  # Ensure >= 0
    stats["mean_minus_2std"] = max(0, stats["mean"] - 2 * stats["std"])  # Ensure >= 0
    stats["mean_minus_3std"] = max(0, stats["mean"] - 3 * stats["std"])  # Ensure >= 0
    
    return stats


def analyze_dataset(data_file, property_col, dataset_name):
    """
    Analyze a single dataset
    
    Args:
        data_file: Path to data file
        property_col: Column index for the property
        dataset_name: Name of the dataset (train, val, test)
        
    Returns:
        Tuple of (property_values, stats)
    """
    print(f"Analyzing {dataset_name} data from {data_file}, using column {property_col} for property...")
    
    # Load data and extract property values
    property_values = load_dataset(data_file, property_col)
    
    if property_values.empty:
        print(f"No valid data found for {dataset_name}")
        return property_values, {}
    
    # Calculate statistics
    stats = calculate_statistics(property_values)
    
    # Print statistics
    print(f"\n----- {dataset_name.upper()} Data Statistics -----")
    print(f"Count: {stats['count']}")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"Standard Deviation: {stats['std']:.4f}")
    print(f"Range: {stats['min']:.4f} to {stats['max']:.4f}")
    
    return property_values, stats


def analyze_combined_data(dataset_files, property_col, output_dir, property_name):
    """
    Analyze and combine multiple datasets
    
    Args:
        dataset_files: Dictionary mapping dataset names to file paths
        property_col: Column index for the property
        output_dir: Directory to save analysis results
        property_name: Display name for the property
        
    Returns:
        Tuple of (combined_values, dataset_values, combined_stats, dataset_stats)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analysis results
    dataset_values = {}
    dataset_stats = {}
    all_values = []
    
    # Analyze each dataset
    for dataset_name, file_path in dataset_files.items():
        if os.path.exists(file_path):
            values, stats = analyze_dataset(file_path, property_col, dataset_name)
            
            if not values.empty:
                dataset_values[dataset_name] = values
                dataset_stats[dataset_name] = stats
                all_values.append(values)
    
    # Combine all values
    if all_values:
        combined_values = pd.concat(all_values)
        combined_stats = calculate_statistics(combined_values)
        
        # Print combined statistics
        print("\n----- COMBINED Data Statistics -----")
        print(f"Total Count: {combined_stats['count']}")
        print(f"Combined Mean: {combined_stats['mean']:.4f}")
        print(f"Combined Median: {combined_stats['median']:.4f}")
        print(f"Combined Standard Deviation: {combined_stats['std']:.4f}")
        print(f"Combined Range: {combined_stats['min']:.4f} to {combined_stats['max']:.4f}")
        
        print("\nPotential Target Values (Combined):")
        print(f"Mean: {combined_stats['mean']:.4f}")
        print(f"Mean + 1σ: {combined_stats['mean_plus_1std']:.4f}")
        print(f"Mean + 2σ: {combined_stats['mean_plus_2std']:.4f}")
        print(f"Mean + 3σ: {combined_stats['mean_plus_3std']:.4f}")
        print(f"Mean - 1σ: {combined_stats['mean_minus_1std']:.4f}")
        print(f"Mean - 2σ: {combined_stats['mean_minus_2std']:.4f}")
        print(f"Mean - 3σ: {combined_stats['mean_minus_3std']:.4f}")
        
        # Save combined statistics to JSON
        stats_file = os.path.join(output_dir, f"{property_name.lower()}_combined_stats.json")
        with open(stats_file, "w") as f:
            json.dump(combined_stats, f, indent=2)
            
        # Also save individual dataset statistics
        for dataset_name, stats in dataset_stats.items():
            dataset_stats_file = os.path.join(output_dir, f"{property_name.lower()}_{dataset_name}_stats.json")
            with open(dataset_stats_file, "w") as f:
                json.dump(stats, f, indent=2)
        
        return combined_values, dataset_values, combined_stats, dataset_stats
    
    else:
        print("No valid data found in any dataset")
        return pd.Series(), {}, {}, {}


def plot_dataset_comparison(dataset_values, dataset_stats, property_name, output_dir):
    """
    Plot comparison of different datasets
    
    Args:
        dataset_values: Dictionary mapping dataset names to property values
        dataset_stats: Dictionary mapping dataset names to statistics
        property_name: Display name for the property
        output_dir: Directory to save the plot
    """
    if not dataset_values:
        print("No datasets to compare")
        return
    
    print("\nCreating dataset comparison plot...")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot KDE for each dataset
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for i, (dataset_name, values) in enumerate(dataset_values.items()):
        color_idx = i % len(colors)
        sns.kdeplot(values, label=f'{dataset_name.capitalize()} (n={len(values)})', 
                   color=colors[color_idx])
        
        # Add mean line
        plt.axvline(dataset_stats[dataset_name]["mean"], color=colors[color_idx], 
                   linestyle='--', linewidth=1.5, 
                   label=f'{dataset_name.capitalize()} Mean: {dataset_stats[dataset_name]["mean"]:.4f}')
    
    # Set labels and title
    plt.xlabel(property_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Comparison of {property_name} Distributions Across Datasets', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{property_name.lower()}_dataset_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Dataset comparison plot saved to: {output_path}")
    
    plt.close()
    
    # Create statistics comparison plot
    plt.figure(figsize=(10, 6))
    
    # Extract key statistics for comparison
    dataset_names = list(dataset_stats.keys())
    means = [stats["mean"] for stats in dataset_stats.values()]
    medians = [stats["median"] for stats in dataset_stats.values()]
    stds = [stats["std"] for stats in dataset_stats.values()]
    
    # Set up bar positions
    x = np.arange(len(dataset_names))
    width = 0.25
    
    # Create grouped bar chart
    plt.bar(x - width, means, width, label='Mean', color='blue')
    plt.bar(x, medians, width, label='Median', color='green')
    plt.bar(x + width, stds, width, label='Std Dev', color='red')
    
    # Add labels and title
    plt.xlabel('Dataset')
    plt.ylabel('Value')
    plt.title(f'{property_name} - Key Statistics Comparison')
    plt.xticks(x, [name.capitalize() for name in dataset_names])
    plt.legend()
    
    plt.tight_layout()
    
    # Save statistics comparison plot
    output_path = os.path.join(output_dir, f"{property_name.lower()}_statistics_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Statistics comparison plot saved to: {output_path}")
    
    plt.close()


def plot_distribution(property_values, stats, target_types, property_name, output_dir, dataset_name="combined"):
    """
    Plot property distribution with statistics
    
    Args:
        property_values: Series with property values
        stats: Dictionary with statistics
        target_types: List of target types to highlight
        property_name: Display name for the property
        output_dir: Directory to save the plot
        dataset_name: Name of the dataset being plotted
    """
    print(f"\nCreating distribution plot for {dataset_name} data...")
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with KDE
    sns.histplot(property_values, kde=True, color='skyblue', edgecolor='black')
    
    # Add vertical lines for key statistics
    plt.axvline(stats["mean"], color='red', linestyle='dashed', linewidth=2, label=f'Mean: {stats["mean"]:.4f}')
    plt.axvline(stats["median"], color='green', linestyle='dashed', linewidth=2, label=f'Median: {stats["median"]:.4f}')
    
    # Add target value lines
    colors = ['purple', 'brown', 'orange', 'magenta', 'cyan']
    for i, target_type in enumerate(target_types):
        if target_type in stats:
            color_idx = i % len(colors)
            plt.axvline(stats[target_type], color=colors[color_idx], linestyle=':', linewidth=2, 
                       label=f'{target_type}: {stats[target_type]:.4f}')
    
    # Set labels and title
    plt.xlabel(property_name, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Distribution of {property_name} Values ({dataset_name.capitalize()})', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    
    # Add text box with statistics
    stats_text = '\n'.join([
        f"Count: {len(property_values)}",
        f"Mean: {stats['mean']:.4f}",
        f"Std Dev: {stats['std']:.4f}",
        f"Min: {stats['min']:.4f}",
        f"Max: {stats['max']:.4f}"
    ])
    plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                 ha='left', va='top', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f"{property_name.lower()}_{dataset_name}_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.close()


def compare_distributions(original_values, generated_values_dict, stats, property_name, output_dir):
    """
    Compare original and generated distributions
    
    Args:
        original_values: Series with original property values
        generated_values_dict: Dictionary mapping target types to generated values
        stats: Dictionary with statistics
        property_name: Display name for the property
        output_dir: Directory to save the plot
    """
    if not generated_values_dict:
        print("No generated distributions to compare")
        return
    
    print("\nComparing distributions...")
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot original distribution
    sns.histplot(original_values, kde=True, alpha=0.6, 
                color='black', edgecolor='black', label='Original Data')
    
    # Plot each generated distribution
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    target_means = {}
    
    for i, (target_type, generated_values) in enumerate(generated_values_dict.items()):
        color_idx = i % len(colors)
        
        # Store mean for later
        target_means[target_type] = generated_values.mean()
        
        # Plot histogram
        sns.histplot(generated_values, kde=True, alpha=0.5, 
                  color=colors[color_idx], edgecolor='black', 
                  label=f'{target_type} (Target: {stats[target_type]:.4f})')
    
    # Add vertical line for original mean
    plt.axvline(stats["mean"], color='black', linestyle='dashed', linewidth=2, 
               label=f'Original Mean: {stats["mean"]:.4f}')
    
    # Add target value lines
    for i, (target_type, _) in enumerate(generated_values_dict.items()):
        color_idx = i % len(colors)
        plt.axvline(stats[target_type], color=colors[color_idx], linestyle=':', linewidth=2)
        
        # Add mean line for generated distribution
        if target_type in target_means:
            plt.axvline(target_means[target_type], color=colors[color_idx], linestyle='--', linewidth=1.5, 
                       label=f'Generated Mean ({target_type}): {target_means[target_type]:.4f}')
    
    # Set labels and title
    plt.xlabel(property_name, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Comparison of All Distributions', fontsize=14)
    plt.legend(loc='best', fontsize=9)
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "all_distributions_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    plt.close()
    
    # Create target achievement summary plot
    plt.figure(figsize=(10, 6))
    
    target_types = []
    target_values = []
    achieved_values = []
    
    for target_type in generated_values_dict.keys():
        if target_type in target_means and target_type in stats:
            target_types.append(target_type)
            target_values.append(stats[target_type])
            achieved_values.append(target_means[target_type])
    
    x = np.arange(len(target_types))
    width = 0.35
    
    plt.bar(x - width/2, target_values, width, label='Target')
    plt.bar(x + width/2, achieved_values, width, label='Achieved')
    
    plt.xlabel('Target Type')
    plt.ylabel(property_name)
    plt.title(f'{property_name} - Target vs. Achieved Values')
    plt.xticks(x, target_types)
    plt.legend()
    
    # Add percentage labels
    for i, (target, achieved) in enumerate(zip(target_values, achieved_values)):
        if target > 0:
            achievement_pct = (achieved / target) * 100
            plt.annotate(f"{achievement_pct:.1f}%", 
                         xy=(i, max(target, achieved) + 0.05 * max(target_values)),
                         ha='center')
    
    plt.tight_layout()
    
    # Save achievement summary plot
    output_path = os.path.join(output_dir, "target_achievement_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Target achievement summary saved to: {output_path}")
    
    plt.close()


def main():
    """Main function for data analysis"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print banner
    print("\n" + "="*70)
    print("MOF-RL DATA ANALYSIS".center(70))
    print("="*70)
    print(f"Configuration file: {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Property column: {args.property_col}")
    print(f"Property name: {args.property_name}")
    print(f"Target types: {args.target_types}")
    print(f"Analyzing combined data: {args.combined}")
    print(f"Datasets to include: {args.datasets}")
    print("="*70 + "\n")
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=FullLoader)
    
    # Get dataset file paths
    dataset_files = {}
    
    # Get training data file (always required)
    train_data_file = config["data"]["train_csv_filename"]
    dataset_files["train"] = train_data_file
    
    # Try to get validation and test data files
    if "val_csv_filename" in config["data"]:
        dataset_files["val"] = config["data"]["val_csv_filename"]
    
    if "test_csv_filename" in config["data"]:
        dataset_files["test"] = config["data"]["test_csv_filename"]
    
    # Filter datasets based on user selection
    dataset_files = {k: v for k, v in dataset_files.items() if k in args.datasets}
    
    # Analyze individual and combined datasets
    combined_values, dataset_values, combined_stats, dataset_stats = analyze_combined_data(
        dataset_files, args.property_col, args.output_dir, args.property_name)
    
    # Plot distributions
    if not combined_values.empty:
        # Plot combined distribution
        plot_distribution(combined_values, combined_stats, args.target_types, 
                         args.property_name, args.output_dir, "combined")
        
        # Plot individual dataset distributions
        for dataset_name, values in dataset_values.items():
            plot_distribution(values, dataset_stats[dataset_name], args.target_types,
                             args.property_name, args.output_dir, dataset_name)
        
        # Plot comparison between datasets
        plot_dataset_comparison(dataset_values, dataset_stats, args.property_name, args.output_dir)
    
    # If CSV file with generated MOFs is provided, compare distributions
    if args.csv_file and os.path.exists(args.csv_file):
        # If single CSV file provided
        if os.path.isfile(args.csv_file):
            generated_data = pd.read_csv(args.csv_file)
            if "target_1" in generated_data.columns:
                # Compare distribution with specified target types
                target_type = next((t for t in args.target_types if t in args.csv_file), "unknown")
                generated_values_dict = {target_type: generated_data["target_1"].dropna()}
                compare_distributions(combined_values, generated_values_dict, combined_stats, 
                                      args.property_name, args.output_dir)
        
        # If directory with multiple CSV files provided
        elif os.path.isdir(args.csv_file):
            # Look for CSV files in each target subdirectory
            generated_values_dict = {}
            
            for target_type in args.target_types:
                # Try to find CSV in target-specific subdirectory
                target_csv = os.path.join(args.csv_file, target_type, "inference", "generated_mofs.csv")
                if os.path.exists(target_csv):
                    generated_data = pd.read_csv(target_csv)
                    if "target_1" in generated_data.columns:
                        generated_values_dict[target_type] = generated_data["target_1"].dropna()
            
            if generated_values_dict:
                compare_distributions(combined_values, generated_values_dict, combined_stats, 
                                      args.property_name, args.output_dir)
            else:
                print("No target-specific CSV files found for comparison")
    
    print("\nData analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    main()
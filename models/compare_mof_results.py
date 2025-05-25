#!/usr/bin/env python
# compare_mof_results.py

"""
Script to compare MOF generation results from different models.
This script normalizes density plots for fair comparison regardless of sample sizes.
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde


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
            property_name = f"{gas_type} adsorption at {pressure} bar (mol/kg)"
        else:
            property_name = f"{gas_type} adsorption (mol/kg)"
        
        print(f"Detected hMOF dataset - using property name: {property_name}")
        return property_name
    
    # Default if no match
    return "Property"

def load_data(filepath, column=None, header=None):
    """
    Load data from CSV file, handling both with and without headers
    
    Args:
        filepath: Path to CSV file
        column: Column index for data without header
        header: Header specification (None for no header)
        
    Returns:
        Series with the property values
    """
    try:
        df = pd.read_csv(filepath, header=header)
        
        # If no header, use column index
        if header is None and column is not None:
            return df.iloc[:, column]
        
        # For files with header, use 'target_1' column
        if 'target_1' in df.columns:
            return df['target_1']
        
        # Return first property column as fallback
        if column is not None and column < len(df.columns):
            return df.iloc[:, column]
            
        print(f"Warning: Could not find property column in {filepath}")
        return None
    except Exception as e:
        print(f"Error loading {filepath}: {str(e)}")
        return None

def generate_stats_table(data_dict):
    """
    Generate a statistics table for all datasets
    
    Args:
        data_dict: Dictionary of {name: data_series}
        
    Returns:
        DataFrame with statistics
    """
    stats = []
    
    for name, series in data_dict.items():
        if series is not None and len(series) > 0:
            stats.append({
                'Dataset': name,
                'Count': len(series),
                'Mean': series.mean(),
                'Std Dev': series.std(),
                'Min': series.min(),
                'Max': series.max(),
                'Median': series.median(),
                'Q1': series.quantile(0.25),
                'Q3': series.quantile(0.75)
            })
    
    return pd.DataFrame(stats)


def plot_normalized_distributions(data_dict, target_values=None, output_path=None, 
                                 property_name="Property", figsize=(12, 6)):
    """
    Create normalized density plots with target markers on x-axis.
    
    Args:
        data_dict: Dictionary of {name: data_series}
        target_values: Dictionary of target values to mark
        output_path: Path to save the plot
        property_name: Name of the property for labels
        figsize: Figure dimensions
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Define color palette - ensure distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    name_to_color = {}
    
    # Get global min/max to ensure we capture full distribution range
    all_values = []
    for name, series in data_dict.items():
        if series is not None and len(series) > 0:
            all_values.extend(series.values)
    
    global_min = min(all_values) if all_values else 0
    global_max = max(all_values) if all_values else 1
    
    # Add padding to ensure we don't cut off distributions
    # padding = (global_max - global_min) * 0.2
    # plot_min = global_min - padding
    # plot_max = global_max + padding

    q_low = np.percentile(all_values, 1)
    q_high = np.percentile(all_values, 99)
    range_padding = (q_high - q_low) * 0.1

    plot_min = q_low - range_padding
    plot_max = q_high + range_padding

    
    # Group data by type
    original_data = None
    finetune_data = None
    rl_data = {}
    
    # First pass: compute KDEs to get curve data with extended range
    kde_data = {}
    for i, (name, series) in enumerate(data_dict.items()):
        if series is not None and len(series) > 0:
            name_to_color[name] = colors[i]
            
            if name.lower() == "original data":
                original_data = series
            elif name.lower() == "fine-tuned model":
                finetune_data = series
            else:
                rl_data[name] = series
            
            # Use a wider range for KDE to prevent cutoffs
            try:
                # Compute KDE with adaptive bandwidth based on data size
                if len(series) < 10:
                    # For small samples, use a wider bandwidth
                    bw = 0.5  # Larger bandwidth for smoother curve
                    kde = gaussian_kde(series, bw_method=bw)
                else:
                    # Default bandwidth for larger samples
                    kde = gaussian_kde(series)
                
                # Generate more points for smoother curves
                x = np.linspace(plot_min, plot_max, 2000)
                y = kde(x)
                kde_data[name] = (x, y)
            except Exception as e:
                print(f"Warning: KDE failed for {name}: {e}")
                # Fallback for very small datasets - create simple histogram
                if len(series) > 0:
                    hist, bin_edges = np.histogram(series, bins=20, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    kde_data[name] = (bin_centers, hist)

    # Separate legends for distributions and targets
    distribution_handles = []
    distribution_labels = []
    target_handles = []
    target_labels = []

    # Second pass: plot distributions
    for name, series in data_dict.items():
        if series is not None and len(series) > 0 and name in kde_data:
            color = name_to_color[name]
            x, y = kde_data[name]
            
            # Plot filled KDE
            ax.fill_between(x, y, alpha=0.2, color=color)
            
            # Create label with sample count
            if name.lower() == "original data":
                label = f'Original Data (n={len(series)})'
            elif name.lower() == "fine-tuned model":
                label = f'Fine-tuned Model (n={len(series)})'
            else:
                # For RL models - create clear names based on target type
                simple_name = name
                # Check if this is a target-specific RL model
                target_mapping = {
                    "mean_plus_1std": "Mean + 1σ Target", 
                    "mean_plus_2std": "Mean + 2σ Target",
                    "mean_minus_1std": "Mean - 1σ Target",
                    "mean_minus_2std": "Mean - 2σ Target",
                    "mean_plus_3std": "Mean + 3σ Target",
                    "mean_minus_3std": "Mean - 3σ Target",
                    "mean": "Mean Target",
                    "custom": "Custom Target"
                }

                # Use exact match for the name instead of substring matching
                if name in target_mapping:
                    simple_name = target_mapping[name]
                
                label = f'{simple_name} (n={len(series)})'
            
            # Plot the line and store for legend
            line = ax.plot(x, y, color=color, linewidth=2)[0]
            distribution_handles.append(line)
            distribution_labels.append(label)

    # Third pass: add target markers on x-axis - only for targets that have RL data
    if target_values:
        # Find the y-range for placing the markers
        y_min, y_max = ax.get_ylim()
        marker_y = -0.03 * y_max  # Below the x-axis
        
        # Ensure axis extends to show markers
        ax.set_ylim(marker_y, y_max * 1.05)
        
        # Only add markers for targets that actually have RL data
        for name, value in target_values.items():
            # Only add markers for targets that are used in RL models
            if name in rl_data:
                # Use exact color match with the RL curve
                target_color = name_to_color.get(name, 'black')
                
                # Add triangle marker on x-axis
                triangle = ax.plot([value], [marker_y], 
                        marker='^', markersize=12, 
                        markeredgecolor='black',
                        markeredgewidth=1.5,
                        color=target_color, zorder=5)[0]
                
                # Add value label below the marker
                ax.annotate(f'{value:.4f}',
                          xy=(value, marker_y),
                          xytext=(0, -20),
                          textcoords='offset points',
                          ha='center', fontsize=11,
                          color=target_color,
                          fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", ec=target_color,
                                  alpha=0.9, lw=1))
                
                # Create user-friendly name for the target legend
                target_mapping = {
                    "mean_plus_1std": "Mean + 1σ",
                    "mean_plus_2std": "Mean + 2σ",
                    "mean_minus_1std": "Mean - 1σ",
                    "mean_minus_2std": "Mean - 2σ",
                    "mean_plus_3std": "Mean + 3σ",
                    "mean_minus_3std": "Mean - 3σ",
                    "mean": "Mean",
                    "custom": "Custom"
                }
                
                friendly_name = target_mapping.get(name, name)
                
                # Add to target legend
                target_handles.append(triangle)
                target_labels.append(f"{friendly_name}: {value:.4f}")
    
    # Drawing the complete box - ensure all spines are visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
    
    # Styling
    ax.set_xlabel(property_name, fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel("Density (normalized)", fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    
    # Ensure x-axis range captures all data plus padding
    ax.set_xlim(plot_min, plot_max)
    
    # Create main distribution legend
    if distribution_handles:
        first_legend = ax.legend(
            handles=distribution_handles,
            labels=distribution_labels,
            fontsize=14, 
            frameon=True, 
            framealpha=0.9,
            edgecolor='gray', 
            loc='upper right'
        )
        ax.add_artist(first_legend)
    
    # Create target values legend
    if target_handles:
        # Create header for targets section
        header_handle = plt.Line2D([0], [0], color='none')
        header_label = 'Target Values:'
        
        # Create target legend with header
        second_legend = ax.legend(
            handles=[header_handle] + target_handles,
            labels=[header_label] + target_labels,
            fontsize=14, 
            frameon=True, 
            framealpha=0.9,
            edgecolor='gray', 
            loc='upper center',
            # bbox_to_anchor=(0.2, 0.98)
        )
        ax.add_artist(second_legend)
    
    plt.tight_layout()

    # Output
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close()



def plot_boxplot_comparison(data_dict, target_values=None, output_path=None, 
                           property_name="Property", figsize=(12, 6)):
    """
    Create boxplot comparison
    
    Args:
        data_dict: Dictionary of {name: data_series}
        target_values: Dictionary of target values to mark with horizontal lines
        output_path: Path to save the plot (if None, displays instead)
        property_name: Name of the property for title and labels
        figsize: Size of the figure as (width, height)
    """
    # Convert to format suitable for boxplot
    df_list = []
    for name, series in data_dict.items():
        if series is not None and len(series) > 0:
            df = pd.DataFrame({property_name: series, 'Dataset': name})
            df_list.append(df)
    
    if not df_list:
        print("No valid data for boxplot")
        return
    
    combined_df = pd.concat(df_list)
    
    plt.figure(figsize=figsize)
    
    # Create boxplot
    ax = sns.boxplot(data=combined_df, x='Dataset', y=property_name, hue='Dataset', palette='tab10', legend=False)
    
    # Add swarmplot with small points for individual data points
    # sns.swarmplot(data=combined_df, x='Dataset', y=property_name, color='black', size=2, alpha=0.5)
    
    # Add target value lines if provided
    if target_values:
        for name, value in target_values.items():
            plt.axhline(value, color='red', linestyle='--', linewidth=1.5, 
                       label=f'{name}: {value:.4f}')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45 if len(data_dict) > 4 else 0, ha='right' if len(data_dict) > 4 else 'center')
    
    # Add labels and title
    plt.xlabel('', fontsize=14)  # Empty label since dataset names are on x-axis
    plt.ylabel(property_name, fontsize=14)
    plt.title(f"Boxplot Comparison - {property_name}", fontsize=14)
    
    # Add legend only if target values exist
    if target_values:
        plt.legend(fontsize=10)
    
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved boxplot to {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_violin_comparison(data_dict, target_values=None, output_path=None, 
                          property_name="Property", figsize=(12, 6)):
    """
    Create violin plot comparison
    
    Args:
        data_dict: Dictionary of {name: data_series}
        target_values: Dictionary of target values to mark with horizontal lines
        output_path: Path to save the plot (if None, displays instead)
        property_name: Name of the property for title and labels
        figsize: Size of the figure as (width, height)
    """
    # Convert to format suitable for violin plot
    df_list = []
    for name, series in data_dict.items():
        if series is not None and len(series) > 0:
            df = pd.DataFrame({property_name: series, 'Dataset': name})
            df_list.append(df)
    
    if not df_list:
        print("No valid data for violin plot")
        return
    
    combined_df = pd.concat(df_list)
    
    plt.figure(figsize=figsize)
    
    # Create violin plot
    ax = sns.violinplot(data=combined_df, x='Dataset', y=property_name, hue='Dataset', palette='tab10', inner='box', legend=False)
    
    # Add target value lines if provided
    if target_values:
        for name, value in target_values.items():
            plt.axhline(value, color='red', linestyle='--', linewidth=1.5, 
                       label=f'{name}: {value:.4f}')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45 if len(data_dict) > 4 else 0, ha='right' if len(data_dict) > 4 else 'center')
    
    # Add labels and title
    plt.xlabel('', fontsize=12)  # Empty label since dataset names are on x-axis
    plt.ylabel(property_name, fontsize=12)
    plt.title(f"Violin Plot Comparison - {property_name}", fontsize=14)
    
    # Add legend only if target values exist
    if target_values:
        plt.legend(fontsize=10)
    
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved violin plot to {output_path}")
    else:
        plt.show()
    
    plt.close()

def plot_ecdf_comparison(data_dict, target_values=None, output_path=None, 
                        property_name="Property", figsize=(12, 6)):
    """
    Create empirical cumulative distribution function (ECDF) plot
    
    Args:
        data_dict: Dictionary of {name: data_series}
        target_values: Dictionary of target values to mark with vertical lines
        output_path: Path to save the plot (if None, displays instead)
        property_name: Name of the property for title and labels
        figsize: Size of the figure as (width, height)
    """
    plt.figure(figsize=figsize)
    
    # Define color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    
    # Plot ECDF for each dataset
    for i, (name, series) in enumerate(data_dict.items()):
        if series is not None and len(series) > 0:
            color = colors[i]
            # Calculate ECDF
            x = np.sort(series)
            y = np.arange(1, len(x) + 1) / len(x)
            plt.plot(x, y, marker='.', linestyle='-', markersize=3, 
                     color=color, label=f'{name} (n={len(series)})', alpha=0.7)
    
    # Add target value lines if provided
    if target_values:
        for name, value in target_values.items():
            plt.axvline(value, color='black', linestyle='--', linewidth=1.5, 
                        label=f'{name}: {value:.4f}')
    
    # Add labels and legend
    plt.xlabel(property_name, fontsize=12)
    plt.ylabel("Cumulative Probability", fontsize=12)
    plt.title(f"ECDF Comparison - {property_name}", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    # Add tight layout
    plt.tight_layout()
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved ECDF plot to {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare MOF generation results from different models")
    parser.add_argument("--original-data", required=True, help="Original training data CSV")
    parser.add_argument("--finetune-data", required=True, help="Fine-tuned model output CSV")
    parser.add_argument("--rl-data", nargs="+", help="RL model output CSVs (space-separated)")
    parser.add_argument("--rl-names", nargs="+", help="Names for RL outputs (space-separated)")
    parser.add_argument("--property-col", type=int, default=1, help="Property column index (for data without header)")
    parser.add_argument("--property-name", default="Property", help="Property display name")
    parser.add_argument("--stats-json", help="Path to stats.json with target values")
    parser.add_argument("--output-dir", default="comparison", help="Output directory")
    parser.add_argument("--include-no-finetune", action="store_true", help="Also generate plots without fine-tune data")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load original data (no header)
    original_data = load_data(args.original_data, args.property_col, header=None)
    
    # Load fine-tuned model data (with header)
    finetune_data = load_data(args.finetune_data, args.property_col, header=0)

   

    if args.property_name == "Property":
        args.property_name = determine_property_name(args.original_data)
        print(f"Using detected property name: {args.property_name}")

    
    
    # Load RL model data (with header)
    rl_data = {}
    if args.rl_data:
        if args.rl_names and len(args.rl_names) == len(args.rl_data):
            for name, file in zip(args.rl_names, args.rl_data):
                rl_data[name] = load_data(file, args.property_col, header=0)
        else:
            # Use default names
            for i, file in enumerate(args.rl_data):
                name = f"RL Model {i+1}"
                rl_data[name] = load_data(file, args.property_col, header=0)
    
    # Combine all data for plotting
    all_data = {"Original Data": original_data, "Fine-tuned Model": finetune_data}
    all_data.update(rl_data)
    
    # Load target values if specified
    target_values = {}
    if args.stats_json and os.path.exists(args.stats_json):
        try:
            with open(args.stats_json, 'r') as f:
                stats = json.load(f)

            # Use raw keys from stats.json that match any RL names
            if args.rl_names:
                for key in args.rl_names:
                    if key in stats:
                        target_values[key] = stats[key]
            else:
                # fallback: include all keys
                target_values = stats


        except Exception as e:
            print(f"Error loading stats.json: {str(e)}")
    
    # Generate statistics table for all data
    stats_df = generate_stats_table(all_data)
    stats_output = os.path.join(args.output_dir, "statistics_comparison.csv")
    stats_df.to_csv(stats_output, index=False)
    print(f"Saved statistics to {stats_output}")
    print("\nStatistics Summary:")
    print(stats_df.to_string())
    
    # Create plots with all data
    print("\nGenerating plots with all data...")
    output_path = os.path.join(args.output_dir, "normalized_distribution_comparison.png")
    plot_normalized_distributions(all_data, target_values, output_path, args.property_name)
     
    violin_path = os.path.join(args.output_dir, "violin_comparison.png")
    plot_violin_comparison(all_data, target_values, violin_path, args.property_name)
    
    ecdf_path = os.path.join(args.output_dir, "ecdf_comparison.png")
    plot_ecdf_comparison(all_data, target_values, ecdf_path, args.property_name)
    
    boxplot_path = os.path.join(args.output_dir, "boxplot_comparison.png")
    plot_boxplot_comparison(all_data, target_values, boxplot_path, args.property_name)
    
    # Create plots without fine-tune data if requested
    if args.include_no_finetune:
        print("\nGenerating plots without fine-tune data...")
        
        # Create a subdirectory for plots without fine-tune
        no_finetune_dir = os.path.join(args.output_dir, "no_finetune")
        os.makedirs(no_finetune_dir, exist_ok=True)
        
        # Data dictionary without fine-tune model
        no_finetune_data = {"Original Data": original_data}
        no_finetune_data.update(rl_data)
        
        # Generate statistics table without fine-tune
        no_finetune_stats = generate_stats_table(no_finetune_data)
        no_finetune_stats_output = os.path.join(no_finetune_dir, "statistics_comparison.csv")
        no_finetune_stats.to_csv(no_finetune_stats_output, index=False)
        print(f"Saved no-finetune statistics to {no_finetune_stats_output}")
        
        # Create plots without fine-tune data
        output_path = os.path.join(no_finetune_dir, "normalized_distribution_comparison.png")
        plot_normalized_distributions(no_finetune_data, target_values, output_path, args.property_name)
        
        violin_path = os.path.join(no_finetune_dir, "violin_comparison.png")
        plot_violin_comparison(no_finetune_data, target_values, violin_path, args.property_name)
        
        ecdf_path = os.path.join(no_finetune_dir, "ecdf_comparison.png")
        plot_ecdf_comparison(no_finetune_data, target_values, ecdf_path, args.property_name)
        
        boxplot_path = os.path.join(no_finetune_dir, "boxplot_comparison.png")
        plot_boxplot_comparison(no_finetune_data, target_values, boxplot_path, args.property_name)
        
        print(f"No-finetune plots saved to {no_finetune_dir}")
    
    print(f"\nAll comparison plots saved to {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

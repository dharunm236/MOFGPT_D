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


# def plot_normalized_distributions(data_dict, target_values=None, output_path=None, 
#                                  property_name="Property", figsize=(12, 6)):
#     """
#     Create normalized density plots for comparison with matching-colored target markers.

#     Args:
#         data_dict: Dictionary of {name: data_series}
#         target_values: Dictionary of target values to mark (color-matched to curves)
#         output_path: Path to save the plot (if None, displays instead)
#         property_name: Name of the property for title and labels
#         figsize: Size of the figure as (width, height)
#     """
#     plt.figure(figsize=figsize)

#     # Define color palette
#     colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
#     name_to_color = {}

#     # Plot each distribution with normalization
#     for i, (name, series) in enumerate(data_dict.items()):
#         if series is not None and len(series) > 0:
#             color = colors[i]
#             name_to_color[name] = color
#             sns.kdeplot(
#                 series,
#                 label=f'{name} (n={len(series)})',
#                 color=color,
#                 fill=True,
#                 alpha=0.3,
#                 common_norm=False,
#                 bw_adjust=1.0
#             )

#     # Add target markers if provided
#     if target_values:
#         for name, value in target_values.items():
#             color = name_to_color.get(name, 'red')
#             # Add triangle marker at a low y-value (just above zero)
#             plt.plot([value], [0.001], marker='v', markersize=10, color=color, zorder=5)
#             # Annotated label below the marker
#             plt.annotate(f'{name}: {value:.2f}',
#                          xy=(value, 0.001),
#                          xytext=(0, -25),
#                          textcoords='offset points',
#                          ha='center', fontsize=10,
#                          color=color,
#                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8))

#     # Aesthetic improvements
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlabel(property_name, fontsize=16, fontweight='bold')
#     plt.ylabel("Density (normalized)", fontsize=16, fontweight='bold')
#     plt.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='gray')
#     plt.tight_layout()

#     # Save or show
#     if output_path:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         plt.savefig(output_path, dpi=1200, bbox_inches='tight')
#         print(f"Saved plot to {output_path}")
#     else:
#         plt.show()

#     plt.close()



# def plot_normalized_distributions(data_dict, target_values=None, output_path=None, 
#                                  property_name="Property", figsize=(12, 6)):
#     """
#     Create normalized density plots with target markers on x-axis.
    
#     Args:
#         data_dict: Dictionary of {name: data_series}
#         target_values: Dictionary of target values to mark
#         output_path: Path to save the plot
#         property_name: Name of the property for labels
#         figsize: Figure dimensions
#     """
#     plt.figure(figsize=figsize)
#     ax = plt.gca()

#     # Define color palette
#     colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
#     name_to_color = {}
    
#     # Group data by type
#     original_data = None
#     finetune_data = None
#     rl_data = {}
    
#     for name, series in data_dict.items():
#         if name.lower() == "original data" and series is not None:
#             original_data = series
#         elif name.lower() == "fine-tuned model" and series is not None:
#             finetune_data = series
#         elif series is not None and len(series) > 0:
#             rl_data[name] = series

#     # First pass: compute KDEs to get curve data
#     kde_data = {}
#     for name, series in data_dict.items():
#         if series is not None and len(series) > 0:
#             # Compute KDE manually
#             kde = gaussian_kde(series)
#             x = np.linspace(series.min(), series.max(), 1000)
#             y = kde(x)
#             kde_data[name] = (x, y)

#     # Set up custom legend handlers
#     legend_elements = []
    
#     # Second pass: plot distributions with grouping
#     for i, (name, series) in enumerate(data_dict.items()):
#         if series is not None and len(series) > 0:
#             color = colors[i]
#             name_to_color[name] = color
#             x, y = kde_data[name]
            
#             # Plot filled KDE
#             ax.fill_between(x, y, alpha=0.2, color=color)
            
#             if name.lower() == "original data":
#                 label = f'Original Data (n={len(series)})'
#             elif name.lower() == "fine-tuned model":
#                 label = f'Fine-tuned Model (n={len(series)})'
#             else:
#                 # For RL models
#                 is_target_in_name = any(target in name for target in ["mean", "std"])
#                 if is_target_in_name:
#                     label = f'{name} (n={len(series)})'
#                 else:
#                     label = f'{name} (n={len(series)})'
            
#             ax.plot(x, y, color=color, linewidth=2, label=label)

#     # Add target markers on x-axis
#     if target_values:
#         target_handles = []
#         target_labels = []
        
#         # Find the y-range for placing the markers
#         y_min, y_max = ax.get_ylim()
#         marker_y = y_min - (y_max - y_min) * 0.05  # Slightly below the x-axis
        
#         for name, value in target_values.items():
#             # Skip non-descriptive names like keys from stats.json
#             if name.lower() in ["mean", "mean plus 1std", "mean plus 2std", "mean minus 1std"]:
#                 # Use a color matching the RL model if possible, otherwise use a distinct color
#                 for rl_name, rl_series in rl_data.items():
#                     if name.lower().replace(" ", "_") in rl_name.lower():
#                         color = name_to_color.get(rl_name, 'black')
#                         break
#                 else:
#                     color = 'black'
                
#                 # Add a marker on the x-axis
#                 triangle = ax.plot([value], [marker_y], 
#                         marker='^', markersize=10, 
#                         markeredgecolor='black',
#                         markeredgewidth=1,
#                         color=color, zorder=5)[0]
                
#                 # Add target value label below the marker
#                 ax.annotate(f'{value:.4f}',
#                           xy=(value, marker_y),
#                           xytext=(0, -15),
#                           textcoords='offset points',
#                           ha='center', fontsize=10,
#                           color=color,
#                           bbox=dict(boxstyle="round,pad=0.2",
#                                   fc="white", ec=color,
#                                   alpha=0.8, lw=1))
                
#                 # Add to separate legend grouping
#                 target_handles.append(triangle)
#                 target_labels.append(f"{name}: {value:.4f}")
        
#         # If we have target markers, add a separate legend group
#         if target_handles:
#             # Add a separator line to the main legend
#             legend_elements.append(plt.Line2D([0], [0], color='none', label=' '))
#             # Add "Target Values:" header 
#             legend_elements.append(plt.Line2D([0], [0], color='none', label='Target Values:'))
#             # Add target elements to legend
#             for handle, label in zip(target_handles, target_labels):
#                 marker_style = dict(marker='^', markersize=8, fillstyle='full', 
#                                    markeredgecolor='black', markeredgewidth=1)
#                 legend_elements.append(plt.Line2D([0], [0], color=handle.get_color(), 
#                                                **marker_style, label=label))

#     # Drawing the complete box
#     ax.spines['top'].set_visible(True)
#     ax.spines['right'].set_visible(True)
#     ax.spines['bottom'].set_visible(True)
#     ax.spines['left'].set_visible(True)
    
#     # Styling
#     ax.set_xlabel(property_name, fontsize=16, fontweight='bold')
#     ax.set_ylabel("Density (normalized)", fontsize=16, fontweight='bold')
#     ax.tick_params(axis='both', labelsize=14)
    
#     # Create a legend with all elements
#     if legend_elements:
#         first_legend = ax.legend(fontsize=12, frameon=True, framealpha=0.9, 
#                                 edgecolor='gray', loc='upper right')
#         ax.add_artist(first_legend)
#         ax.legend(handles=legend_elements, fontsize=12, frameon=True, 
#                  framealpha=0.9, edgecolor='gray', loc='upper left')
#     else:
#         ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='gray')
    
#     # Ensure axes extend slightly beyond data
#     x_min, x_max = ax.get_xlim()
#     x_range = x_max - x_min
#     ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    
#     y_min, y_max = ax.get_ylim()
#     ax.set_ylim(y_min, y_max * 1.05)  # Add some headroom
    
#     plt.tight_layout()

#     # Output
#     if output_path:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"Saved plot to {output_path}")
#     else:
#         plt.show()
#     plt.close()







# def plot_normalized_distributions(data_dict, target_values=None, output_path=None, 
#                                  property_name="Property", figsize=(12, 6)):
#     """
#     Create normalized density plots with target markers on x-axis.
    
#     Args:
#         data_dict: Dictionary of {name: data_series}
#         target_values: Dictionary of target values to mark
#         output_path: Path to save the plot
#         property_name: Name of the property for labels
#         figsize: Figure dimensions
#     """
#     plt.figure(figsize=figsize)
#     ax = plt.gca()

#     # Define color palette - ensure distinct colors
#     colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
#     name_to_color = {}
    
#     # Get global min/max to ensure we capture full distribution range
#     all_values = []
#     for name, series in data_dict.items():
#         if series is not None and len(series) > 0:
#             all_values.extend(series.values)
    
#     global_min = min(all_values) if all_values else 0
#     global_max = max(all_values) if all_values else 1
    
#     # Add padding to ensure we don't cut off distributions
#     padding = (global_max - global_min) * 0.2
#     plot_min = global_min - padding
#     plot_max = global_max + padding
    
#     # Group data by type
#     original_data = None
#     finetune_data = None
#     rl_data = {}
    
#     # First pass: compute KDEs to get curve data with extended range
#     kde_data = {}
#     for i, (name, series) in enumerate(data_dict.items()):
#         if series is not None and len(series) > 0:
#             name_to_color[name] = colors[i]
            
#             if name.lower() == "original data":
#                 original_data = series
#             elif name.lower() == "fine-tuned model":
#                 finetune_data = series
#             elif "mean" in name.lower() or "std" in name.lower():
#                 rl_data[name] = series
            
#             # Use a wider range for KDE to prevent cutoffs
#             try:
#                 # Compute KDE with adaptive bandwidth based on data size
#                 if len(series) < 10:
#                     # For small samples, use a wider bandwidth
#                     bw = 0.5  # Larger bandwidth for smoother curve
#                     kde = gaussian_kde(series, bw_method=bw)
#                 else:
#                     # Default bandwidth for larger samples
#                     kde = gaussian_kde(series)
                
#                 # Generate more points for smoother curves
#                 x = np.linspace(plot_min, plot_max, 2000)
#                 y = kde(x)
#                 kde_data[name] = (x, y)
#             except Exception as e:
#                 print(f"Warning: KDE failed for {name}: {e}")
#                 # Fallback for very small datasets - create simple histogram
#                 if len(series) > 0:
#                     hist, bin_edges = np.histogram(series, bins=20, density=True)
#                     bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#                     kde_data[name] = (bin_centers, hist)

#     # Separate legends for distributions and targets
#     distribution_handles = []
#     distribution_labels = []
#     target_handles = []
#     target_labels = []

#     # Second pass: plot distributions
#     for name, series in data_dict.items():
#         if series is not None and len(series) > 0 and name in kde_data:
#             color = name_to_color[name]
#             x, y = kde_data[name]
            
#             # Plot filled KDE
#             ax.fill_between(x, y, alpha=0.2, color=color)
            
#             # Create label with sample count
#             if name.lower() == "original data":
#                 label = f'Original Data (n={len(series)})'
#             elif name.lower() == "fine-tuned model":
#                 label = f'Fine-tuned Model (n={len(series)})'
#             else:
#                 # # For RL models - simplify names if they contain targets
#                 # simple_name = name
#                 # for target_text in ["mean", "mean_plus_1std", "mean_plus_2std", "mean_minus_1std"]:
#                 #     if target_text.lower().replace("_", " ") in name.lower():
#                 #         simple_name = target_text.replace("_", " ").title()
#                 #         break
#                 # label = f'{simple_name} (n={len(series)})'


#                 # For RL models - create clear names based on target type
#                 simple_name = name
#                 # Check if this is a target-specific RL model
#                 target_mapping = {
#                     "mean_plus_1std": "Mean + 1σ Target", 
#                     "mean_plus_2std": "Mean + 2σ Target",
#                     "mean_minus_1std": "Mean - 1σ Target",
#                     "mean_minus_2std": "Mean - 2σ Target",
#                     "mean_plus_3std": "Mean + 3σ Target",
#                     "mean_minus_3std": "Mean - 3σ Target",
#                     "mean": "Mean Target",
#                     "custom": "Custom Target"
#                 }

#                 # Try to extract the target type from the name
#                 for target_key, target_label in target_mapping.items():
#                     if target_key in name.lower():
#                         simple_name = target_label
#                         break

#                 label = f'{simple_name} (n={len(series)})'
            
#             # Plot the line and store for legend
#             line = ax.plot(x, y, color=color, linewidth=2)[0]
#             distribution_handles.append(line)
#             distribution_labels.append(label)

#     # Third pass: add target markers on x-axis
#     if target_values:
#         # Find the y-range for placing the markers
#         y_min, y_max = ax.get_ylim()
#         marker_y = -0.03 * y_max  # Below the x-axis
        
#         # Ensure axis extends to show markers
#         ax.set_ylim(marker_y, y_max * 1.05)
        
#         # Add a label for target values section
#         for name, value in target_values.items():
#             # Determine target color - try to match with corresponding RL model
#             target_color = 'black'  # Default
#             target_name_normalized = name.lower().replace(" ", "_")
            
#             # Find matching RL dataset if possible
#             for rl_name in rl_data:
#                 if target_name_normalized in rl_name.lower().replace(" ", "_"):
#                     target_color = name_to_color.get(rl_name, 'black')
#                     break
            
#             # Add triangle marker on x-axis
#             triangle = ax.plot([value], [marker_y], 
#                     marker='^', markersize=12, 
#                     markeredgecolor='black',
#                     markeredgewidth=1.5,
#                     color=target_color, zorder=5)[0]
            
#             # Add value label below the marker
#             ax.annotate(f'{value:.4f}',
#                       xy=(value, marker_y),
#                       xytext=(0, -20),
#                       textcoords='offset points',
#                       ha='center', fontsize=11,
#                       color=target_color,
#                       fontweight='bold',
#                       bbox=dict(boxstyle="round,pad=0.3",
#                               fc="white", ec=target_color,
#                               alpha=0.9, lw=1))
            
#             # Add to target legend
#             target_handles.append(triangle)
#             target_labels.append(f"{name}: {value:.4f}")
    
#     # Drawing the complete box - ensure all spines are visible
#     for spine in ax.spines.values():
#         spine.set_visible(True)
#         spine.set_linewidth(1.2)
    
#     # Styling
#     ax.set_xlabel(property_name, fontsize=16, fontweight='bold')
#     ax.set_ylabel("Density (normalized)", fontsize=16, fontweight='bold')
#     ax.tick_params(axis='both', labelsize=14)
    
#     # Ensure x-axis range captures all data plus padding
#     ax.set_xlim(plot_min, plot_max)
    
#     # Create main distribution legend
#     if distribution_handles:
#         first_legend = ax.legend(
#             handles=distribution_handles,
#             labels=distribution_labels,
#             fontsize=12, 
#             frameon=True, 
#             framealpha=0.9,
#             edgecolor='gray', 
#             loc='upper right'
#         )
#         ax.add_artist(first_legend)
    
#     # Create target values legend
#     if target_handles:
#         # Create header for targets section
#         header_handle = plt.Line2D([0], [0], color='none')
#         header_label = 'Target Values:'
        
#         # Create target legend with header
#         second_legend = ax.legend(
#             handles=[header_handle] + target_handles,
#             labels=[header_label] + target_labels,
#             fontsize=12, 
#             frameon=True, 
#             framealpha=0.9,
#             edgecolor='gray', 
#             loc='upper left'
#         )
#         ax.add_artist(second_legend)
    
#     plt.tight_layout()

#     # Output
#     if output_path:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         print(f"Saved plot to {output_path}")
#     else:
#         plt.show()
#     plt.close()

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
    padding = (global_max - global_min) * 0.2
    plot_min = global_min - padding
    plot_max = global_max + padding
    
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
    ax.set_xlabel(property_name, fontsize=16, fontweight='bold')
    ax.set_ylabel("Density (normalized)", fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    
    # Ensure x-axis range captures all data plus padding
    ax.set_xlim(plot_min, plot_max)
    
    # Create main distribution legend
    if distribution_handles:
        first_legend = ax.legend(
            handles=distribution_handles,
            labels=distribution_labels,
            fontsize=12, 
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
            fontsize=12, 
            frameon=True, 
            framealpha=0.9,
            edgecolor='gray', 
            loc='upper left'
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
    plt.xlabel('', fontsize=12)  # Empty label since dataset names are on x-axis
    plt.ylabel(property_name, fontsize=12)
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
            
            # # Add key target values
            # target_keys = ["mean", "mean_plus_1std", "mean_plus_2std", "mean_minus_1std"]
            # for key in target_keys:
            #     if key in stats:
            #         target_values[key.replace("_", " ").title()] = stats[key]

            # # Match target value keys to those expected by the plot function
            # target_mapping = {
            #     "mean": "Mean",
            #     "mean_plus_1std": "Mean + 1σ Target",
            #     "mean_plus_2std": "Mean + 2σ Target",
            #     "mean_minus_1std": "Mean - 1σ Target",
            #     "mean_minus_2std": "Mean - 2σ Target"
            # }

            # for key, label in target_mapping.items():
            #     if key in stats:
            #         target_values[label] = stats[key]

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



# python compare_mof_results.py \
#     --original-data ../benchmark_datasets/finetune/hMOF_CH4_0.5_small_mofid_finetune/train.csv \
#     --finetune-data mofgpt_hMOF_CH4_0.5_small_mofid_finetune_pipeline_results/finetune_inference/placeholder_generations.csv \
#     --rl-data mofgpt_hMOF_CH4_0.5_small_mofid_finetune_pipeline_results/rl_mean_inference/mof_gpt_pretrain_best_RL_best_generations.csv \
#     --output-dir comparison


# python compare_mof_results.py \
#     --original-data /path/to/dataset/train.csv \
#     --finetune-data /path/to/finetune_inference/generations.csv \
#     --rl-data /path/to/rl_mean_inference/generations.csv /path/to/rl_mean_plus_1std_inference/generations.csv \
#     --rl-names "mean" "mean_plus_1std" \
#     --property-col 1 \
#     --property-name "CH4 adsorption at 0.5 bar" \
#     --stats-json /path/to/analysis/stats.json \
#     --output-dir comparison_results \
#     --include-no-finetune
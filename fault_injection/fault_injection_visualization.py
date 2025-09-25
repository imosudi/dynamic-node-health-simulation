import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import yaml
from typing import List, Optional, Dict, Any, Tuple,    Union 
from enum import Enum



def create_visualisation(results: List[Dict], metric_names: List[str], baseline_values: Dict[str, float]):
    """Create comprehensive visualisation with proper handling of multiple fault occurrences."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    steps = [r['step'] for r in results]
    
    colors = ['blue', 'red', 'green']
    fault_colors = ['red', 'orange', 'purple', 'brown', 'pink']  # Different colors for different fault types
    
    # Extract fault information
    fault_periods = []
    current_fault = None
    fault_start = None
    
    for i, result in enumerate(results):
        if result['any_fault_active'] and current_fault is None:
            # Fault starting
            current_fault = result['active_faults'][0]['fault_name'] if result['active_faults'] else "Unknown"
            fault_start = i
        elif not result['any_fault_active'] and current_fault is not None:
            # Fault ending
            fault_periods.append((fault_start, i-1, current_fault))
            current_fault = None
            fault_start = None
    
    # Handle case where fault is still active at end
    if current_fault is not None:
        fault_periods.append((fault_start, len(results)-1, current_fault))
    
    # Create a mapping of fault types to colors
    unique_faults = list(set([fault_type for _, _, fault_type in fault_periods]))
    fault_color_map = {fault: fault_colors[i % len(fault_colors)] for i, fault in enumerate(unique_faults)}
    
    for i, metric in enumerate(metric_names):
        values = [r[metric] for r in results]
        
        # Plot metric values
        axes[i].plot(steps, values, color=colors[i], linewidth=1.5, label=f'{metric.upper()}')
        
        # Add baseline
        baseline = baseline_values[metric]
        axes[i].axhline(y=baseline, color='black', linestyle='--', alpha=0.7, label='Baseline')
        
        # Highlight fault periods with different colors for different fault types
        for start, end, fault_type in fault_periods:
            color = fault_color_map[fault_type]
            # Only add label for first occurrence of each fault type in the first subplot
            label = fault_type if (i == 0 and fault_periods.index((start, end, fault_type)) == 
                                  [f[2] for f in fault_periods].index(fault_type)) else ""
            axes[i].axvspan(start, end, alpha=0.3, color=color, label=label)
        
        axes[i].set_ylabel(f'{metric.upper()}')
        axes[i].grid(True, alpha=0.3)
        
        # Only show legend on first subplot to avoid repetition
        if i == 0:
            axes[i].legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    axes[-1].set_xlabel('Time Step')
    plt.suptitle('Fault Injection Simulation Results with Multiple Fault Types')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('fault_injection_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'fault_injection_analysis.png'")
    plt.close()


# Alternative version if you want to show all active faults at each step
def create_detailed_visualisation(results: List[Dict], metric_names: List[str], baseline_values: Dict[str, float]):
    """Create visualisation showing all active faults at each time step."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    steps = [r['step'] for r in results]
    
    colors = ['blue', 'red', 'green']
    fault_colors = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'yellow', 'gray']
    
    # Create a list of all unique fault names
    all_faults = []
    for result in results:
        if result['active_faults']:
            for fault in result['active_faults']:
                if fault['fault_name'] not in all_faults:
                    all_faults.append(fault['fault_name'])
    
    # Create color mapping
    fault_color_map = {fault: fault_colors[i % len(fault_colors)] for i, fault in enumerate(all_faults)}
    
    for i, metric in enumerate(metric_names):
        values = [r[metric] for r in results]
        
        # Plot metric values
        axes[i].plot(steps, values, color=colors[i], linewidth=1.5, label=f'{metric.upper()}')
        
        # Add baseline
        baseline = baseline_values[metric]
        axes[i].axhline(y=baseline, color='black', linestyle='--', alpha=0.7, label='Baseline')
        
        # Highlight fault periods
        for step_idx, result in enumerate(results):
            if result['active_faults']:
                # Get the primary fault for coloring (use the first one)
                primary_fault = result['active_faults'][0]['fault_name']
                color = fault_color_map[primary_fault]
                
                # Draw a vertical line at this step
                axes[i].axvline(x=step_idx, color=color, alpha=0.2, linewidth=2)
        
        axes[i].set_ylabel(f'{metric.upper()}')
        axes[i].grid(True, alpha=0.3)
    
    # Create a custom legend for fault types
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=fault_color_map[fault], alpha=0.7, label=fault) 
                      for fault in all_faults]
    axes[0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    axes[-1].set_xlabel('Time Step')
    plt.suptitle('Fault Injection Simulation - All Active Faults Marked')
    plt.tight_layout()
    
    # Save plot
    plt.savefig('fault_injection_detailed_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nDetailed plot saved as 'fault_injection_detailed_analysis.png'")
    plt.close()
    
#!/usr/bin/env python3
##fault_injection/fault_injection_sim.py 
"""
Complete Fault Injection Simulation with YAML-based Configuration
Enhanced with comprehensive fault templates from YAML
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import yaml
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

class FaultType(Enum):
    ADDITIVE = "additive"
    PERSISTENT = "persistent"

class FaultSubtype(Enum):
    STUCK_AT = "stuck_at"
    RAMP = "ramp"
    ADDITIVE_PERSISTENT = "additive_persistent"

class OccurrenceModel(Enum):
    BERNOULLI = "bernoulli"

class DurationDistribution(Enum):
    GEOMETRIC = "geometric"
    DETERMINISTIC = "deterministic"


class MetricGenerator:
    def __init__(self, metric_names: List[str], baselines: List[float], 
                 noise_levels: List[float], seed: Optional[int], 
                 signs: Optional[List[float]] = None):
        self.metric_names = metric_names
        self.baselines = np.array(baselines)
        self.noise_levels = np.array(noise_levels)
        self.signs = np.array(signs) if signs is not None else np.ones_like(self.baselines)
        self.rng = np.random.default_rng(seed)

    def step(self) -> np.ndarray:
        noise = self.rng.normal(0, self.noise_levels)
        return self.baselines + noise


def create_valid_covariance_matrix(variances: List[float], correlations: List[List[float]]) -> np.ndarray:
    """Create a valid covariance matrix from variances and correlations."""
    n = len(variances)
    cov_matrix = np.zeros((n, n))
    
    # Set diagonal (variances)
    np.fill_diagonal(cov_matrix, variances)
    
    # Set off-diagonal elements (covariances = correlation * sqrt(var1 * var2))
    for i in range(n):
        for j in range(i + 1, n):
            covariance = correlations[i][j] * np.sqrt(variances[i] * variances[j])
            cov_matrix[i, j] = covariance
            cov_matrix[j, i] = covariance
    
    return cov_matrix


class YAMLFaultTemplate:
    def __init__(self, template_config: Dict[str, Any], metric_names: List[str], 
                 baseline_values: Dict[str, float], max_values: Dict[str, float]):
        #print(template_config); time.sleep(200)
        self.id = template_config['id']
        self.name = template_config['name']
        self.type = FaultType(template_config['type'])
        self.subtype = FaultSubtype(template_config.get('subtype', '')) if 'subtype' in template_config else None
        self.affected_metrics = template_config['affected_metrics']
        self.metric_names = metric_names
        self.baseline_values = baseline_values
        self.max_values = max_values
        
        # Parse severity information
        if 'severity_vector' in template_config:
            self.severity_vector = np.array(template_config['severity_vector'])
        else:
            self.severity_vector = None
            
        # Parse covariance information
        self.cov_matrix = None
        if 'covariance' in template_config:
            cov_config = template_config['covariance']
            if cov_config['mode'] == 'pairwise_rho':
                n_metrics = len(self.affected_metrics)
                # Create correlation matrix with given rho
                corr_matrix = np.ones((n_metrics, n_metrics)) * cov_config['rho']
                np.fill_diagonal(corr_matrix, 1.0)
                
                # Create variances based on severity vector
                variances = [self.severity_vector[i]**2 for i in range(n_metrics)]
                self.cov_matrix = create_valid_covariance_matrix(variances, corr_matrix.tolist())
        
        # Parse occurrence model
        self.occurrence_model = OccurrenceModel(template_config['occurrence_model']['type'])
        self.occurrence_p = template_config['occurrence_model']['p']
        
        # Parse duration distribution
        self.duration_dist = DurationDistribution(template_config['duration_dist']['type'])
        if self.duration_dist == DurationDistribution.GEOMETRIC:
            self.duration_p = template_config['duration_dist']['p']
        elif self.duration_dist == DurationDistribution.DETERMINISTIC:
            self.duration_steps = template_config['duration_dist']['n_steps']
        
        # Parse additional properties
        self.stuck_value = template_config.get('stuck_value', None)
        self.ramp_slope = template_config.get('ramp_slope', 0.0)
        self.max_deviation = template_config.get('max_deviation', float('inf'))
        self.repeatable = template_config.get('repeatable', False)
        self.note = template_config.get('note', '')
        
        # For ramp faults
        self.current_drift = None

    def sample_occurrence(self, rng: np.random.Generator) -> bool:
        """Sample whether this fault occurs in the current step."""
        if self.occurrence_model == OccurrenceModel.BERNOULLI:
            return rng.random() < self.occurrence_p
        return False

    def sample_duration(self, rng: np.random.Generator) -> int:
        """Sample the duration of this fault."""
        if self.duration_dist == DurationDistribution.GEOMETRIC:
            return rng.geometric(self.duration_p)
        elif self.duration_dist == DurationDistribution.DETERMINISTIC:
            return self.duration_steps
        return 1

    def get_initial_shift(self, rng: np.random.Generator) -> np.ndarray:
        """Get the initial shift for this fault."""
        if self.type == FaultType.ADDITIVE:
            if self.cov_matrix is not None:
                # Sample from multivariate normal distribution
                shift = rng.multivariate_normal(self.severity_vector, self.cov_matrix)
            else:
                # Simple additive shift
                shift = self.severity_vector.copy()
            return shift
            
        elif self.type == FaultType.PERSISTENT:
            if self.subtype == FaultSubtype.STUCK_AT:
                # Determine stuck value
                if self.stuck_value == 'max_value':
                    # Get max values for affected metrics
                    shift = np.array([self.max_values[self.metric_names[i]] for i in self.affected_metrics])
                elif self.stuck_value == 'baseline':
                    # Get baseline values for affected metrics
                    shift = np.array([self.baseline_values[self.metric_names[i]] for i in self.affected_metrics])
                else:
                    # Numeric stuck value
                    shift = np.array([self.stuck_value] * len(self.affected_metrics))
                return shift
                
            elif self.subtype == FaultSubtype.RAMP:
                # Start with no shift, will ramp over time
                self.current_drift = np.zeros(len(self.affected_metrics))
                return self.current_drift.copy()
                
            elif self.subtype == FaultSubtype.ADDITIVE_PERSISTENT:
                # Persistent additive shift
                return self.severity_vector.copy()
                
        return np.zeros(len(self.affected_metrics))

    def update_drift(self, current_shift: np.ndarray, step: int) -> np.ndarray:
        """Update drift for ramp-type faults."""
        if (self.type == FaultType.PERSISTENT and 
            self.subtype == FaultSubtype.RAMP and
            self.current_drift is not None):
            
            # Apply ramp slope
            new_drift = self.current_drift + np.array([self.ramp_slope] * len(self.affected_metrics))
            
            # Apply max deviation constraint
            for i in range(len(new_drift)):
                if abs(new_drift[i]) > self.max_deviation:
                    new_drift[i] = np.sign(new_drift[i]) * self.max_deviation
            
            self.current_drift = new_drift
            return new_drift.copy()
            
        return current_shift


class YAMLFaultInjector:
    def __init__(self, yaml_config_path: str, metric_names: List[str], 
                 baseline_values: Dict[str, float], max_values: Dict[str, float],
                 seed: Optional[int] ):
        # Load YAML configuration
        with open(yaml_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.templates = []
        for template_config in config['fault_templates']:
            self.templates.append(YAMLFaultTemplate(
                template_config, metric_names, baseline_values, max_values
            ))
        
        self.rng = np.random.default_rng(seed)
        self.active_faults = {}  # fault_id -> template, remaining_steps, current_delta
        self.fault_history = []
        self.step_count = 0
        self.metric_names = metric_names

    def maybe_inject(self, metrics: np.ndarray) -> np.ndarray:
        """Apply fault injection to metrics based on YAML configuration."""
        result_metrics = metrics.copy()
        active_faults_this_step = []
        
        # Check for new fault occurrences
        for template in self.templates:
            if template.id not in self.active_faults and template.sample_occurrence(self.rng):
                # Start new fault
                duration = template.sample_duration(self.rng)
                initial_shift = template.get_initial_shift(self.rng)
                
                self.active_faults[template.id] = {
                    'template': template,
                    'remaining_steps': duration,
                    'current_delta': initial_shift,
                    'start_step': self.step_count
                }
                
                self.fault_history.append({
                    'fault_id': template.id,
                    'fault_name': template.name,
                    'start_step': self.step_count,
                    'initial_delta': dict(zip(
                        [self.metric_names[i] for i in template.affected_metrics], 
                        initial_shift
                    )),
                    'duration': duration
                })
                
                print(f"Starting fault: {template.name} for {duration} steps")
        
        # Apply active faults and update them
        faults_to_remove = []
        for fault_id, fault_data in self.active_faults.items():
            template = fault_data['template']
            current_delta = fault_data['current_delta']
            
            # Apply fault to affected metrics
            for i, metric_idx in enumerate(template.affected_metrics):
                result_metrics[metric_idx] += current_delta[i]
            
            # Update drift for ramp faults
            updated_delta = template.update_drift(current_delta, self.step_count)
            fault_data['current_delta'] = updated_delta
            
            # Decrement remaining steps
            fault_data['remaining_steps'] -= 1
            if fault_data['remaining_steps'] <= 0:
                faults_to_remove.append(fault_id)
                print(f"Ending fault: {template.name}")
            
            active_faults_this_step.append(template.name)
        
        # Remove finished faults
        for fault_id in faults_to_remove:
            del self.active_faults[fault_id]
        
        self.step_count += 1
        return result_metrics

    def get_fault_status(self) -> Dict[str, Any]:
        """Get current fault status."""
        active_faults = []
        for fault_id, fault_data in self.active_faults.items():
            active_faults.append({
                'fault_id': fault_id,
                'fault_name': fault_data['template'].name,
                'remaining_steps': fault_data['remaining_steps'],
                'current_delta': dict(zip(
                    [self.metric_names[i] for i in fault_data['template'].affected_metrics],
                    fault_data['current_delta']
                ))
            })
        
        return {
            'active_faults': active_faults,
            'any_active': len(active_faults) > 0
        }


def analyse_fault_impact(results: List[Dict], baseline_values: Dict[str, float]):
    """Analyse the impact of fault injection on metrics."""
    print("\n" + "="*50)
    print("FAULT IMPACT ANALYSIS")
    print("="*50)
    
    normal_periods = [r for r in results if not r['any_fault_active']]
    fault_periods = [r for r in results if r['any_fault_active']]
    
    if not fault_periods:
        print("No fault periods detected.")
        return
    
    print(f"Baseline values: {baseline_values}")
    print(f"Normal period samples: {len(normal_periods)}")
    print(f"Fault period samples: {len(fault_periods)}")
    data_retuned = {}
    for metric in ['cpu', 'rtt', 'plr']:
        normal_vals = [r[metric] for r in normal_periods]
        fault_vals = [r[metric] for r in fault_periods]
        
        normal_mean = np.mean(normal_vals)
        fault_mean = np.mean(fault_vals)
        impact_percent = ((fault_mean - normal_mean) / normal_mean) * 100
        
        print(f"\n{metric.upper()}:")
        print(f"  Normal: μ={normal_mean:.4f}")
        print(f"  Fault:  μ={fault_mean:.4f}")
        print(f"  Impact: {impact_percent:+.4f}% change")
        print(
            {
                str(metric): {
                    "Normal": {
                    "μ":round(normal_mean, 4)
                    },
                    "Fault": {
                    "μ":round(fault_mean, 4)
                    },
                     "Impact":round(impact_percent, 4)
                }
            }
        )
        data_retuned[str(metric)] = {
                    "Normal": {
                        "μ": round(float(normal_mean), 4)
                    },
                    "Fault": {
                        "μ": round(float(fault_mean), 4)
                    },
                    "Impact": round(float(impact_percent), 4)
                }
    return data_retuned


def detect_anomalies(results: List[Dict], baseline_values: Dict[str, float], 
                    threshold_config: Optional[Dict[str, float]] = None):
    """Enhanced anomaly detection with metric-specific thresholds."""
    if threshold_config is None:
        threshold_config = {
            'cpu': {'std_threshold': 2.0, 'relative_threshold': 0.5},
            'rtt': {'std_threshold': 2.0, 'relative_threshold': 0.3},
            'plr': {'std_threshold': 1.5, 'relative_threshold': 1.0}  # More sensitive to PLR changes
        }
    
    print("\n" + "="*50)
    print("ENHANCED ANOMALY DETECTION")
    print("="*50)
    
    for metric in ['cpu', 'rtt', 'plr']:
        values = [r[metric] for r in results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        baseline = baseline_values[metric]
        
        std_threshold = threshold_config[metric]['std_threshold']
        relative_threshold = threshold_config[metric]['relative_threshold']
        
        anomalies = []
        for r in results:
            # Standard deviation-based detection
            z_score = abs(r[metric] - mean_val) / std_val
            
            # Relative change detection (especially important for PLR)
            relative_change = abs(r[metric] - baseline) / baseline if baseline > 0 else 0
            
            # Combined detection logic
            if (z_score > std_threshold) or (relative_change > relative_threshold):
                anomalies.append((r['step'], r[metric], z_score, relative_change, r['any_fault_active']))
        
        print(f"\n{metric.upper()} anomalies (>{std_threshold}σ or >{relative_threshold*100:.0f}% change):")
        if anomalies:
            for step, value, z_score, rel_change, fault_active in anomalies:
                status = "FAULT" if fault_active else "NORMAL"
                print(f"  Step {step:2d}: {value:.4f} (z={z_score:.2f}, Δ={rel_change*100:.1f}%) [{status}]")
        else:
            print("  No anomalies detected")

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
    
def run_complete_simulation(baseline_values:dict, max_values:dict, steps: int, seed: int):
    """Run complete simulation with YAML-based fault configuration."""
    print("Starting YAML-based Fault Injection Simulation...")
    print("="*50)
    
    # Setup
    metric_names = ["cpu", "rtt", "plr"]
    
    gen = MetricGenerator(
        metric_names, 
        baselines=[baseline_values['cpu'], baseline_values['rtt'], baseline_values['plr']], 
        noise_levels=[0.05, 5, 0.002], 
        seed=seed
    )
    
    # Create YAML-based fault injector
    injector = YAMLFaultInjector(
        'fault_injection/fault_templates_zero.yaml',
        #'fault_injection/templates.yaml',  
        metric_names,
        baseline_values,
        max_values,
        seed=seed
    )
    
    # Custom threshold configuration
    threshold_config = {
        'cpu': {'std_threshold': 2.0, 'relative_threshold': 0.5},
        'rtt': {'std_threshold': 2.0, 'relative_threshold': 0.3},
        'plr': {'std_threshold': 1.5, 'relative_threshold': 1.0}
    }
    
    # Run simulation
    results = []
    for t in range(steps):
        metrics = gen.step()
        observed = injector.maybe_inject(metrics)
        status = injector.get_fault_status()
        
        result = {
            'step': t,
            'cpu': observed[0],
            'rtt': observed[1],
            'plr': observed[2],
            'any_fault_active': status['any_active'],
            'active_faults': status['active_faults']
        }
        results.append(result)
        
        if status['any_active']:
            print(f"t={t:2d}: cpu={observed[0]:.3f}, rtt={observed[1]:.1f}, "
                  f"plr={observed[2]:.4f}, faults={[f['fault_name'] for f in status['active_faults']]}")
            print("\n⚠️ Fault detected — stopping simulation early.\n"); break 
        else:
            print(f"t={t:2d}: cpu={observed[0]:.3f}, rtt={observed[1]:.1f}, "
                  f"plr={observed[2]:.4f}, no faults")
    
    # Analysis
    analyse_fault_impact(results, baseline_values)
    detect_anomalies(results, baseline_values, threshold_config)
    create_visualisation(results, metric_names, baseline_values)
    #create_detailed_visualisation(results, metric_names, baseline_values)
    
    # Fault history
    print("\n" + "="*50)
    print("FAULT HISTORY")
    print("="*50)
    for i, fault in enumerate(injector.fault_history):
        print(f"Fault {i+1}: {fault['fault_name']} (ID: {fault['fault_id']})")
        print(f"  Started: step {fault['start_step']}")
        print(f"  Duration: {fault['duration']} steps")
        print(f"  Initial delta: {fault['initial_delta']}")
    
    if not injector.fault_history:
        print("No faults occurred during simulation")
    
    return results, injector, analyse_fault_impact(results, baseline_values)


"""if __name__ == "__main__":
    try:
        # Run simulation
        results, injector = run_complete_simulation(steps=100, seed=42)
        print(f"\nSimulation completed successfully!")
        print(f"Total steps: {len(results)}")
        print(f"Fault periods: {sum(1 for r in results if r['any_fault_active'])}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()"""
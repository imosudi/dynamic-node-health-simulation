
import time
import numpy as np
import yaml
from typing import List, Optional, Dict, Any, Tuple,    Union 
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
            #print("template: ", template.name, " id: ", template.id); time.sleep(10)
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

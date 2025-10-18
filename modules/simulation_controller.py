
#!/usr/bin/env python3
##fault_injection/fault_injection_sim.py 
"""
Complete Fault Injection Simulation with YAML-based Configuration
Enhanced with comprehensive fault templates from YAML
"""
"""
Pending tasks: health_monitor, HealthMonitor

"""
import time
import numpy as np
import random
import matplotlib

from modules.fault_injector import YAMLFaultInjector
from modules.node__data_generator import MetricGenerator_old
from modules.node_profiler import analyse_fault_impact, detect_anomalies
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import yaml
from typing import List, Optional, Dict, Any, Tuple,    Union 
from enum import Enum
from modules.node_operations.logic import NetworkHierarchy

"""class MetricGenerator:
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
        return self.baselines + noise"""

class HealthMonitor:
    def __init__(self, metric_names, signs, alpha=0.1, init_window=10, kappa=2.0):
        """
        alpha: EWMA smoothing for theta
        beta_sigma: EWMA smoothing for sigma_theta
        kappa: decision margin multiplier
        """
        
        self.metric_names = metric_names
        self.signs = np.array(signs)
        self.alpha = alpha
        self.init_window = init_window
        self.kappa = kappa
        self.values = []
        self.theta = None
        self.sigma_theta = None

    def normalise(self, metrics, baselines, stds):
        return [(metrics[i] - baselines[self.metric_names[i]]) / stds[i]
                for i in range(len(self.metric_names))]

    def update(self, metrics, baselines, stds):
        normed = self.normalise(metrics, baselines, stds)
        h = np.mean(self.signs * np.array(normed))
        self.values.append(h)

        if len(self.values) <= self.init_window:
            self.theta = np.mean(self.values)
            self.sigma_theta = np.std(self.values) if len(self.values) > 1 else 0.01
            return h, self.theta, "INIT"

        self.theta = self.alpha * h + (1 - self.alpha) * self.theta
        residual = abs(h - self.theta)
        self.sigma_theta = 0.9 * self.sigma_theta + 0.1 * residual

        if h < self.theta - 1.0:
            status = "Faulty"
        elif h >= 1.0:
            status = "Good"
        elif h >= 0:
            status = "Fair"
        else:
            status = "Poor"

        return h, self.theta, status

def run_complete_simulation(node_id: str,
                            default_weights,
                            layer_profiles: dict,
                            max_values: dict,
                            steps: int,
                            seed: int,
                            fault_templates: list,
                            stop_on_fault: bool = True):
    """
    Run complete simulation with YAML-based fault configuration.

    Parameters
    ----------
    default_weights : dict
        Weighting of metrics for health metric.
    baseline_values : dict
        Dict with 'cpu', 'rtt', 'plr' baseline values.
    max_values : dict
        Dict with 'cpu', 'rtt', 'plr' max values.
    steps : int
        Number of simulation steps.
    seed : int
        RNG seed.
    fault_templates : list
        Fault definitions from YAML.
    stop_on_fault : bool, default=True
        Whether to stop early when a real fault is detected.
    """
    print("\n Starting Fault Injection ...")
    print("="*60)

    
    # metric setup
    metric_names = ["cpu", "rtt", "plr"]
    if node_id == "CloudDBServer":
        baseline_values = layer_profiles["CLOUD"]["baseline"]
        noise_scales    = layer_profiles["CLOUD"]["noise"]
        noise_scales    = list(noise_scales.values())
    elif node_id == "L1N_01":
        baseline_values = layer_profiles["L1"]["baseline"]
        noise_scales    = layer_profiles["L1"]["noise"]
        noise_scales    = list(noise_scales.values())
    elif node_id.startswith("L2N"):
        baseline_values = layer_profiles["L2"]["baseline"]
        noise_scales    = layer_profiles["L2"]["noise"]
        noise_scales    = list(noise_scales.values())   
    elif node_id.startswith("L3N"):
        baseline_values = layer_profiles["L3"]["baseline"]
        noise_scales    = layer_profiles["L3"]["noise"] 
        noise_scales    = list(noise_scales.values())   
    elif node_id.startswith("L4N"):
        baseline_values = layer_profiles["L4"]["baseline"]
        noise_scales    = layer_profiles["L4"]["noise"] 
        noise_scales    = list(noise_scales.values())

    else:
        raise ValueError(f"Unknown node layer for {node_id}")
    
        

    '''print(f"Node ID: {node_id}"); 
    print(f"Baseline values: {baseline_values}")
    print(f"Max values: {max_values}")
    print(f"Simulation steps: {steps}")
    #print("layer_profiles: ", layer_profiles)
    print("noise_scales: ", noise_scales); # time.sleep(2000)'''
    '''for key, value in layer_profiles.items(): 
        print(f"  {key}: baseline={value['baseline']}, noise={value['noise']}")'''

    
    #time.sleep(1000)

    
    
    
    # metric setup
    #noise_scales = [0.05, 5, 0.002]  # Noise levels for cpu, rtt, plr
    

    # metric generator
    """gen = MetricGenerator_old(
        metric_names,
        baselines=[baseline_values['cpu'],
                   baseline_values['rtt'],
                   baseline_values['plr']],
        noise_levels=noise_scales,
        seed=seed
    )"""
    
    

    gen_old = MetricGenerator_old(
        metric_names,
        baselines=[baseline_values['cpu'],
                   baseline_values['rtt'],
                   baseline_values['plr']],
        noise_levels=noise_scales,
        seed=seed
    )

    #step1 = gen.step()
    #print("Shape:", step1.shape)  # (num_nodes, 3)
    #print("First 5 nodes' metrics:\n", step1); time.sleep(2)
    #print("gen.step() 1: ", gen_old.step(), "\n end of gen_old = MetricGenerator_old() "); time.sleep(2)
    #print("gen.step() 2: ", gen.step(), "\n end of net = NetworkHierarchy();  gen = MetricGenerator()"); time.sleep(2)

    # fault injector
    injector = YAMLFaultInjector(
        fault_templates,
        metric_names,
        baseline_values,
        max_values,
        seed=seed
    )

    # thresholds
    threshold_config = {
        'cpu': {'std_threshold': 2.0, 'relative_threshold': 0.5},
        'rtt': {'std_threshold': 2.0, 'relative_threshold': 0.3},
        'plr': {'std_threshold': 1.5, 'relative_threshold': 1.0}
    }

    # run loop
    results = []
    # I have jobs to do here by replacing the HealthMonitor with one in health_metrics.healthMetricCalculator
    health_monitor = HealthMonitor(metric_names, signs=[-1, -1, -1], alpha=0.1)
    history = []

    
    for t in range(steps):
        metrics = gen_old.step()
        observed = injector.maybe_inject(metrics)
        status = injector.get_fault_status()
        #print("observed: ", observed, " status: ", status); time.sleep(2); time.sleep(2)
        res = {
            'step': t,
            'cpu': float(observed[0]), # 'cpu': observed[0],
            'rtt': float(observed[1]), # 'rtt': observed[1],
            'plr': float(observed[2]), # 'plr': observed[2],
            'any_fault_active': status['any_active'],
            'active_faults': status['active_faults']  # list of dicts
        }
        #print("res: ", res); time.sleep(2)
        print("\n baseline_values: ", baseline_values); time.sleep(2)
        results.append(res)

        # update health monitor
        h, theta, _status = health_monitor.update(
            np.array([res['cpu'], res['rtt'], res['plr']], dtype=float),
            baseline_values,
            noise_scales
        )

        # collect per-step history
        history.append({
            'step': t,
            'baseline': baseline_values,
            'observed': np.array([res['cpu'], res['rtt'], res['plr']], dtype=float),
            'active_faults_ids': [f.get('fault_id') for f in res['active_faults']],
            'active_faults_names': [f.get('fault_name') for f in res['active_faults']],
            'health_metric': h,
            'adaptive_threshold': theta,
            'health_status': _status
        })

        """# print
        if res['any_fault_active']:
            fault_names = [f['fault_name'] for f in res['active_faults']]
            print(f"t={t:2d}: cpu={res['cpu']:.3f}, rtt={res['rtt']:.1f}, "
                  f"plr={res['plr']:.4f}, faults={fault_names}")
            if stop_on_fault:
                print("\n⚠️ Fault detected — stopping simulation early.\n")
                break
        else:
            print(f"t={t:2d}: cpu={res['cpu']:.3f}, rtt={res['rtt']:.1f}, "
                  f"plr={res['plr']:.4f}, no faults")"""

    # analysis
    analysis_data = analyse_fault_impact(results, baseline_values)
    tendency = detect_anomalies(results, baseline_values, threshold_config)

    # fault history (optional: in zero-fault mode this may be empty)
    """print("\n" + "="*60)
    print("FAULT HISTORY")
    print("="*60)
    if injector.fault_history:
        for i, fault in enumerate(injector.fault_history):
            print(f"Fault {i+1}: {fault['fault_name']} (ID: {fault['fault_id']})")
            #print(f"  Started: step {fault['start_step']}")
            #print(f"  Duration: {fault['duration']} steps")
            #print(f"  Initial delta: {fault['initial_delta']}")
    else:
        print("No faults occurred during simulation")"""

    return results, injector, analysis_data, history, tendency

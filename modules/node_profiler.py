from typing import List, Optional, Dict, Any, Tuple,    Union 
import time
import numpy as np


layer_profiles = {
            "CLOUD": {
                "baseline": {'cpu': 0.09, 'rtt': 22.8, 'plr': 0.0068}, #[0.090, 22.8, 0.0068],
                "noise":    {'cpu': 0.05, 'rtt': 5, 'plr': 0.002} #[0.05, 5, 0.002]
            },
            "L1": {
                "baseline": {'cpu': 0.45, 'rtt': 20, 'plr': 0.4},
                "noise":    {'cpu': 0.12, 'rtt': 6, 'plr': 0.009}
            },
            "L2": {
                "baseline": {'cpu': 0.40, 'rtt': 45, 'plr': 0.5},
                "noise":    {'cpu': 0.2, 'rtt': 10, 'plr': 0.05}
            },
            "L3": {
                "baseline": {'cpu': 0.35, 'rtt': 60, 'plr': 7.0},
                "noise":    {'cpu': 0.25, 'rtt': 15, 'plr': 0.075}
            },
            "L4": {
                "baseline": {'cpu': 0.25, 'rtt': 75, 'plr': 10.0},
                "noise":    {'cpu': 0.45, 'rtt': 20, 'plr': 0.085}
            }
            #"SENSOR": {
            #    "baseline": [5, 200, 15.0],
            #    "noise":    [2, 30, 4.0]
            #}
        }
    



def analyse_fault_impact(results: List[Dict], baseline_values: Dict[str, float]):
    """Analyse the impact of fault injection on metrics.
       Always returns analysis data, even if no faults are detected.
    """
    """print("\n" + "="*50)
    print("FAULT IMPACT ANALYSIS")
    print("="*50)"""
    
    normal_periods = [r for r in results if not r['any_fault_active']]
    fault_periods = [r for r in results if r['any_fault_active']]
    
    """print(f"Baseline values: {baseline_values}")
    print(f"Normal period samples: {len(normal_periods)}")
    print(f"Fault period samples: {len(fault_periods)}")"""
    
    data_retuned = {}
    for metric in ['cpu', 'rtt', 'plr']:
        normal_vals = [r[metric] for r in normal_periods] if normal_periods else [baseline_values[metric]]
        fault_vals = [r[metric] for r in fault_periods] if fault_periods else normal_vals
        
        normal_mean = np.mean(normal_vals)
        fault_mean = np.mean(fault_vals)
        
        # If no fault periods, impact is zero
        impact_percent = ((fault_mean - normal_mean) / normal_mean) * 100 if normal_mean != 0 else 0.0
        
        print(f"\n{metric.upper()}:")
        print(f"  Normal: μ={normal_mean:.4f}")
        print(f"  Fault:  μ={fault_mean:.4f}")
        print(f"  Impact: {impact_percent:+.4f}% change")
        
        metric_data = {
            "Normal": {"μ": round(float(normal_mean), 4)},
            "Fault": {"μ": round(float(fault_mean), 4)},
            "Impact": round(float(impact_percent), 4)
        }
        print(metric_data)
        data_retuned[metric] = metric_data
    
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
    
    """print("\n" + "="*50)
    print("ENHANCED ANOMALY DETECTION")
    print("="*50)"""
    results_lastdata = results[-1:]
    
    tendency_data = {}
    last_data = results_lastdata[0] if results_lastdata else {}
    node_data = {"step": last_data.get("step", -1), "cpu": round(float(last_data.get("cpu", 0.0)), 4), "rtt": round(float(last_data.get("rtt", 0.0)), 4), "plr": round(float(last_data.get("plr", 0.0)), 4)    }
    tendency_data["node_data"] = node_data
    #print("tendency_data: ", tendency_data); time.sleep(200)
    for metric in ['cpu', 'rtt', 'plr']:
        values = [r[metric] for r in results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        baseline = baseline_values[metric]
        
        std_threshold = threshold_config[metric]['std_threshold']
        relative_threshold = threshold_config[metric]['relative_threshold']
        print(f"{metric}: ", "mean_val: ", mean_val, " std_val: ", std_val)
        tendency_data[metric] = {"mean" : round(float(mean_val), 4), "std": round(float(std_val), 4)}
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
    #print("tendency_data: ", tendency_data)  
    return tendency_data      


#main.py
import os
import time
from collections import defaultdict
from enum import Enum

import pandas as pd
from typing import List, Optional, Dict, Any, Tuple,    Union 
from modules.simulation_controller import run_complete_simulation
from logs.csv_writer import write_detailed_csv
#import numpy as np
from modules.health_classifier import  healthMetricCalculator
from modules.node_operations.node_generators import create_node_list
from modules.node_operations.node_id_extract import extract_node_ids
from modules.node_profiler import layer_profiles

"""
Pending tasks: health_monitor, HealthMonitor

"""
try:
    all_node_ids = extract_node_ids('data/node_list.csv')    
except :
    node_list_path = "data/node_list.csv"
    if os.path.exists(node_list_path):
        node_list = pd.read_csv(node_list_path)
    else:
        print("Will make an attempt to generate node_list.csv ...")
        node_list = create_node_list()
    all_node_ids = extract_node_ids('data/node_list.csv') 


# Configuration
default_weights     = {'CPU': 0.3,  'RTT': 0.3,     'PLR': 0.4 } # weights must sum to 1.0
baseline_values     = {'cpu':0.090, 'rtt': 22.8,    'plr':0.0068} # cpu expressed in fraction: x/100, rtt in milliseconds, plr in fraction: x/100   
max_values          = {'cpu': 0.75, 'rtt': 150,     'plr': 0.5}
static_thresholds   = {'cpu': 0.73, 'rtt': 130.0,   'plr': 0.45 }   # Example: 70% CPU usage as threshold, Example: 100ms RTT as threshold  Example: 5% packet loss rate as threshold

    
# Fault injection templates
fault_templates = 'data/fault_templates_zero.yaml',
fault_templates = 'data/fault_templates.yaml',
#fault_templates = 'data/templates.yaml'
for node in all_node_ids:
    try:
        # Run simulation
        results, injector, data_returned, history, tendency_data \
            = run_complete_simulation(
                node,
                 default_weights,
                   layer_profiles,
                     max_values,
                        steps=2,
                          seed=1,
                            fault_templates='data/fault_templates_zero.yaml')
                            #fault_templates='data/fault_templates.yaml')
        print(f"\nSimulation completed successfully!")
        print(f"Total steps: {len(results)}")
        #print("results: ", results); time.sleep(500)
        """for index, item in enumerate(results):
            print("\n","node: ", node, " index: ", index, " item: ", item); time.sleep(2)"""
        
        # Write detailed CSV
        for node_id in all_node_ids:
            health_metric_calculator = healthMetricCalculator(
                 node_id, tendency_data, default_weights, static_thresholds
                 )                                                  
            health_metric = health_metric_calculator.healthMetric()
            #print(f"Computed Health Metric for {node_id}: {health_metric}")
            # Check if node_id exists in results    
            if node_id not in results:
                print(f"Warning: Node ID '{node_id}' not found in results.")
            else:
                print(f"Node ID '{node_id}' found in results.")
                
        # Write results to CSV

        dataset = health_metric[1]
        def get_metrics_dataset():
            """Return the collected metrics as a pandas DataFrame"""
            return pd.DataFrame(dataset)
        
        def save_metrics_to_csv(filename="node_metrics.csv"):
            """Save the collected metrics to a CSV file"""
            df = get_metrics_dataset()
            df.to_csv(filename, index=False)
            return df
        
        # Save metrics to CSV
        save_metrics_to_csv("node_metrics.csv")

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()


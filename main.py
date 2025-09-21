#main.py
import os
import time
from collections import defaultdict

import pandas as pd
from fault_injection.fault_injection_sim import run_complete_simulation
from logging_mod.csv_writer import write_detailed_csv
#import numpy as np
from fault_injection.health_metrics import  healthMetricCalculator
from node_operations.node_generators import create_node_list
from node_operations.node_id_extract import extract_node_ids


try:
    all_node_ids = extract_node_ids('data/node_list.csv')    
except :
    node_list_path = "data/node_list.csv"
    if os.path.exists(node_list_path):
        node_list = pd.read_csv(node_list_path)
    else:
        node_list = create_node_list()
    all_node_ids = extract_node_ids('data/node_list.csv') 



#print(all_node_ids); time.sleep(200) 
# ['CloudDB_Server', 'L1N_01', 'L2N_01', 'L2N_02', 'L2N_03', 'L2N_04', 'L3N_01', 'L3N_02', 'L3N_03', 'L3N_04', 'L3N_05', 'L3N_06', 'L3N_07', 'L3N_08', 'L3N_09', 'L3N_10', 'L3N_11', 'L3N_12', 'L4N_01', 'L4N_02', 'L4N_03', 'L4N_04', 'L4N_05', 'L4N_06', 'L4N_07', 'L4N_08', 'L4N_09', 'L4N_10', 'L4N_11', 'L4N_12', 'L4N_13', 'L4N_14', 'L4N_15', 'L4N_16', 'L4N_17', 'L4N_18', 'L4N_19', 'L4N_20', 'L4N_21', 'L4N_22', 'L4N_23', 'L4N_24', 'L4N_25', 'L4N_26', 'L4N_27', 'L4N_28', 'L4N_29', 'L4N_30', 'L4N_31', 'L4N_32', 'L4N_33', 'L4N_34', 'L4N_35', 'L4N_36']

# Configuration
default_weights     = {'CPU': 0.3,  'RTT': 0.3,     'PLR': 0.4 } # weights must sum to 1.0
baseline_values     = {'cpu':0.090, 'rtt': 22.8,    'plr':0.0068} # cpu expressed in fraction: x/100, rtt in milliseconds, plr in fraction: x/100   
max_values          = {'cpu': 0.75, 'rtt': 150,     'plr': 0.5}
static_thresholds   = {'cpu': 0.73,  'rtt': 130.0,   'plr': 0.45 }   # Example: 70% CPU usage as threshold, Example: 100ms RTT as threshold  Example: 5% packet loss rate as threshold

# Fault injection templates
fault_templates = 'fault_injection/fault_templates_zero.yaml',
fault_templates = 'fault_injection/fault_templates.yaml',
#fault_templates = 'fault_injection/templates.yaml'

try:
        # Run simulation
        results, injector, data_returned, history, tendency_data = run_complete_simulation(default_weights, baseline_values, max_values, steps=10, seed=10, fault_templates='fault_injection/fault_templates.yaml')
        print(f"\nSimulation completed successfully!")
        print(f"Total steps: {len(results)}")
        #print("results: ", results); time.sleep(500)


        #health_metric_calculator = healthMetricCalculator("L1N_01",tendency_data, default_weights, static_thresholds)                                                  
        #health_metric = health_metric_calculator.healthMetric()
        #print(f"Computed Health Metric: {health_metric}")
        


        #print("all_node_ids:", all_node_ids)

        for node_id in all_node_ids:
            health_metric_calculator = healthMetricCalculator(node_id, tendency_data, default_weights, static_thresholds)                                                  
            health_metric = health_metric_calculator.healthMetric()
            print(f"Computed Health Metric for {node_id}: {health_metric}")
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


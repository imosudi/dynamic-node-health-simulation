#main.py
import json
import os
import time
from collections import defaultdict
from enum import Enum

import pandas as pd
from typing import List, Optional, Dict, Any, Tuple,    Union 
from modules.node_operations.metrics_pre_processor import NodeMetricsProcessor
from modules.simulation_controller import run_complete_simulation
from logs.csv_writer import write_detailed_csv
#import numpy as np
from modules.health_classifier import  healthMetricCalculator
from modules.node_operations.node_generators import create_node_list
from modules.node_operations.node_id_extract import extract_node_ids
from modules.node_profiler import layer_profiles
from modules.transceive.transceive_processor import NodeTransceiveProcessor


import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
default_weights     = {'CPU': 0.3,  'RTT': 0.3,     'PLR': 0.4 } # weights must sum to 1.0
#baseline_values     = {'cpu':0.090, 'rtt': 22.8,    'plr':0.0068} # cpu expressed in fraction: x/100, rtt in milliseconds, plr in fraction: x/100   
max_values          = {'cpu': 0.75, 'rtt': 150,     'plr': 0.5}
static_thresholds   = {'cpu': 0.73, 'rtt': 130.0,   'plr': 0.45 }   # Example: 70% CPU usage as threshold, Example: 100ms RTT as threshold  Example: 5% packet loss rate as threshold

    
# Fault injection templates
fault_templates = 'data/fault_templates_zero.yaml',
fault_templates = 'data/fault_templates.yaml',
#fault_templates = 'data/templates.yaml'





"""
Pending tasks: health_monitor, HealthMonitor

"""
try:
    logging.info("Attempting to extract node IDs ...")
    all_node_ids = extract_node_ids('data/node_list.csv')    
except :
    logging.info("node_list.csv not found.")
    node_list_path = "data/node_list.csv"
    logging.info("Checking for node_list.csv at path: {}".format(node_list_path))
    if os.path.exists(node_list_path):
        logging("node_list.csv found. Loading node list...")
        logging.info("Loading node_list.csv from path: {}".format(node_list_path))
        node_list = pd.read_csv(node_list_path)
    else:
        logging.info("node_list.csv not found at path: {}.".format(node_list_path))
        logging.info("Will make an attempt to generate node_list.csv ...")
        node_list = create_node_list()
        logging.info("node_list.csv generated successfully.")

    logging.info("Saving generated node_list to data/node_list.csv ...")
    all_node_ids = extract_node_ids('data/node_list.csv') 


# Node Metrics Collection Processing...logging.info("Begin Node Metrics Collection Processing...")
try:
    node_metric_path = "data/node_metrics.csv" # should be "data/node_metrics.csv"
    if not os.path.exists(node_metric_path):
        # Initialise processor
        logging.info("Begin Node Metrics Collection Processing...")
        processor = NodeMetricsProcessor(
                fault_template_path='data/fault_templates_zero.yaml',
                output_filename='data/node_metrics.csv'
            )
        # Initialise processor

        logging.info("NodeMetricsProcessor initialised successfully!")

        processor.collect_node_metrics(
                all_node_ids=all_node_ids,
                run_complete_simulation=run_complete_simulation,
                healthMetricCalculator=healthMetricCalculator,
                default_weights=default_weights,
                layer_profiles=layer_profiles,
                max_values=max_values,
                static_thresholds=static_thresholds,
                steps=2,
                seed=1
            )
except:
    pass

logging.info("Begin Transceive Processing...") 

# Initialise processor
transceiveprocessor = NodeTransceiveProcessor('data/node_metrics.csv')
    
# Transceive initialisation
transceiveprocessor.transceive_initialisation(default_value=0)

# Get summary
logging.info("=" * 60)
logging.info("SUMMARY")
logging.info("=" * 60)
summary = transceiveprocessor.get_summary()
logging.info(
    json.dumps(
        summary, indent=2
        )
    )
# Export as DataFrame
print("\n" + "=" * 60)
print("DATAFRAME OUTPUT (first 5 rows)")
print("=" * 60)
df = transceiveprocessor.to_dataframe()
print(df.head())
print(f"\nDataFrame shape: {df.shape}")
    
# Export as CSV
print("\n" + "=" * 60)
print("CSV OUTPUT")
print("=" * 60)
csv_path = transceiveprocessor.to_csv('data/node_metrics_with_transceive.csv')
print(f"CSV saved to: {csv_path}")
print("CSV created successfully!")


    # Export as JSON
print("\n" + "=" * 60)
print("JSON OUTPUT (first 2 records)")
print("=" * 60)
json_output = transceiveprocessor.to_json()
json_data = json.loads(json_output)
print(json.dumps(json_data, indent=2)) #(json_data[:2], indent=2)
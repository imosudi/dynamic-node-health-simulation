"""
Transceive Processor Module

This module provides functionality to process node metrics data from JSON format.
It extracts key metrics and prepares them for further analysis.

Filename: modules/transceive/transceive_processor.py
"""

from cmath import sqrt
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
from logs import logging
import random

from modules.fault_injector import YAMLFaultInjector
from modules.node__data_generator import MetricGenerator_old
from modules.node_profiler import analyse_fault_impact, detect_anomalies
from modules.simulation_controller import HealthMonitor

class NodeTransceiveProcessorInit:
    """
    Processes transceive node metrics data from JSON input.
    """

    def __init__(self, json_data_dump: str):
        if not json_data_dump or not isinstance(json_data_dump, str):
            raise ValueError("json_data_dump must be a non-empty string")
        try:
            self.raw_data = json.loads(json_data_dump)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        if not isinstance(self.raw_data, list):
            raise ValueError("JSON data must be a list of objects")
        self.extracted_data = []

    def extract_metrics(self) -> List[Dict[str, Any]]:
        """Extract metrics from JSON records."""
        self.extracted_data = []
        #print("self.raw_data:, ", self.raw_data); time.sleep(200)
        for idx, record in enumerate(self.raw_data):
            try:
                extracted = {
                    'timestamp': record.get('timestamp'),
                    'node_id': record.get('node_id'),
                    'plr': float(record.get('plr', 0)),
                    'cpu': float(record.get('cpu', 0)),
                    'rtt': float(record.get('rtt', 0)),
                    'weighted_health_score': float(record.get('weighted_health_score', 0)),
                    'health_threshold': float(record.get('health_threshold', 0)),
                    'health_difference': float(record.get('health_difference', 0)),
                    'health_status': record.get('health_status', 'UNKNOWN'),
                    'anomaly_type': record.get('anomaly_type', 'NONE'),
                    'transceive_count': int(record.get('transceive_count', 0))
                }
                #print("Extracted Record:", extracted); time.sleep(200)
                self.extracted_data.append(extracted)
            except (ValueError, TypeError) as e:
                logging.warning(f"Skipping record {idx} due to extraction error: {e}")
                continue
        return self.extracted_data

    def _default_processing(self) -> Dict[str, List[Dict]]:
        """Group extracted data by node ID."""
        grouped = {}
        for record in self.extracted_data:
            node_id = record['node_id']
            grouped.setdefault(node_id, []).append(record)
        logging.info("Default processing completed: grouped data by node_id")
        #print("Grouped Data: ", grouped);# time.sleep(200)
        return grouped

    def _current_node_ids(self) -> List[str]:
        """Return a list of current node IDs."""
        grouped = self._default_processing()
        return list(grouped.keys())
    

    def get_baseline_and_noise(self, layer_profiles:dict, node_id: str) -> Dict[str, Any]:
        """
        Get baseline metrics and noise scales for a specific node.
        
        Args:
            node_id (str): The node identifier
            layer_profiles (dict): Layer profiles containing noise scales
            
        Returns:
            Dict containing:
                - baseline: Dict with cpu, rtt, plr from first record
                - noise_scales: List of noise scale values for the node's layer
                - layer: The identified layer name
                - node_id: The node identifier
                
        Raises:
            ValueError: If node layer cannot be identified or layer_profiles not set
        """
        grouped = self._default_processing()
        #print(grouped); time.sleep(200)
        #grouped = self._default_processing(); print(grouped); #time.sleep(500)
        group_data = grouped[node_id]
        if not layer_profiles:
            raise ValueError("Layer profiles not set. Initialise with layer_profiles parameter.")
        
        if not group_data:
            raise ValueError(f"No data provided for node {node_id}")
        
        # Extract baseline from first record
        baseline = {
            'cpu': group_data[0]['cpu'],
            'rtt': group_data[0]['rtt'],
            'plr': group_data[0]['plr']
        }
        
        # Determine layer and get noise scales
        layer = None
        noise_scales = None
        
        if node_id == "CloudDBServer":
            noise_scales    = layer_profiles["CLOUD"]["noise"]
            noise_scales    = list(noise_scales.values())
            layer = 0
        elif node_id == "L1N_01":
            noise_scales    = layer_profiles["L1"]["noise"]
            noise_scales    = list(noise_scales.values())
            layer = 1
        elif node_id.startswith("L2N"):
            noise_scales    = layer_profiles["L2"]["noise"]
            noise_scales    = list(noise_scales.values()) 
            layer = 2   
        elif node_id.startswith("L3N"):
            noise_scales    = layer_profiles["L3"]["noise"] 
            noise_scales    = list(noise_scales.values())
            layer = 3
        elif node_id.startswith("L4N"):
            noise_scales    = layer_profiles["L4"]["noise"] 
            noise_scales    = list(noise_scales.values())
            layer = 4
        else:
            raise ValueError(f"Unknown node layer for {node_id}")
        
        
        return {
            "transceive_count": group_data[0]['transceive_count'],
            'baseline': baseline,
            'noise_scales': noise_scales,
            'layer': layer,
            "node_id": node_id
        }

    def get_baseline_and_noise_(self, layer_profiles: dict, node_id: str) -> Dict[str, Any]:
        """Extract baseline and noise scales for given node."""
        grouped = self._default_processing()
        group_data = grouped[node_id]
        #print("grouped: ", grouped); time.sleep(200)
        if not layer_profiles:
            raise ValueError("Layer profiles not set. Initialise with layer_profiles parameter.")
        if not group_data:
            raise ValueError(f"No data provided for node {node_id}")

        baseline = {
            'cpu': group_data[0]['cpu'],
            'rtt': group_data[0]['rtt'],
            'plr': group_data[0]['plr']
        }

        layer_map = {
            "CloudDBServer": ("CLOUD", 0),
            "L1N_01": ("L1", 1)
        }

        # Determine layer dynamically
        if node_id.startswith("L2N"):
            layer_key, layer = "L2", 2
        elif node_id.startswith("L3N"):
            layer_key, layer = "L3", 3
        elif node_id.startswith("L4N"):
            layer_key, layer = "L4", 4
        elif node_id in layer_map:
            layer_key, layer = layer_map[node_id]
        else:
            raise ValueError(f"Unknown node layer for {node_id}")

        noise_scales = list(layer_profiles[layer_key]["noise"].values())

        return {
            "transceive_count": group_data[0]['transceive_count'],
            'baseline': baseline,
            'noise_scales': noise_scales,
            'layer': layer,
            "node_id": node_id
        }

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not self.extracted_data:
            self.extract_metrics()

        total_records = len(self.extracted_data)
        unique_nodes = len({r['node_id'] for r in self.extracted_data})

        health_status_counts = {}
        anomaly_type_counts = {}

        for record in self.extracted_data:
            health_status_counts[record['health_status']] = health_status_counts.get(record['health_status'], 0) + 1
            anomaly_type_counts[record['anomaly_type']] = anomaly_type_counts.get(record['anomaly_type'], 0) + 1

        return {
            'total_records': total_records,
            'unique_nodes': unique_nodes,
            'health_status_distribution': health_status_counts,
            'anomaly_type_distribution': anomaly_type_counts
        }


class NodeTransceiveProcessor:
    """
    Processes transceive node metrics data from JSON input.
    
    Attributes:
        raw_data (List[Dict]): The raw JSON data as a list of dictionaries
        extracted_data (List[Dict]): Extracted and validated metrics data
    """
    
    def __init__(self, layer_profiles, json_data_dump: str):
        """
        Initialise the processor with JSON data.
        
        Args:
            json_data_dump (str): JSON string containing node metrics data
            
        Raises:
            ValueError: If json_data_dump is invalid JSON or not a list
        """
        if not json_data_dump or not isinstance(json_data_dump, str):
            raise ValueError("json_data_dump must be a non-empty string")
        
        try:
            self.raw_data = json.loads(json_data_dump)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        if not isinstance(self.raw_data, list):
            raise ValueError("JSON data must be a list of objects")
        
        try:
            self.layer_profiles = layer_profiles
        except Exception as e:
            raise ValueError(f"Invalid layer_profiles: {e}")
        
        self.extracted_data = []
    
    def extract_metrics(self) -> List[Dict[str, Any]]:
        """
        Extract relevant metrics from each node record.
        
        Returns:
            List[Dict]: List of dictionaries containing extracted metrics
            
        Each extracted record contains:
            - timestamp: ISO format timestamp
            - node_id: Node identifier
            - plr: Packet Loss Rate (float)
            - cpu: CPU utilisation (float)
            - rtt: Round Trip Time (float)
            - weighted_health_score: Calculated health score (float)
            - health_status: Status string (HEALTHY/UNHEALTHY)
            - anomaly_type: Type of anomaly detected
            - transceive_count: Number of transceive operations (int)
        """
        self.extracted_data = []
        
        for idx, record in enumerate(self.raw_data):
            try:
                extracted = {
                    'timestamp': record.get('timestamp'),
                    'node_id': record.get('node_id'),
                    'plr': float(record.get('plr', 0)),
                    'cpu': float(record.get('cpu', 0)),
                    'rtt': float(record.get('rtt', 0)),
                    'plr_mean': float(record.get('plr_mean', float(record.get('plr', 0)))),
                    'cpu_mean': float(record.get('cpu_mean', float(record.get('cpu', 0)))),
                    'rtt_mean': float(record.get('rtt_mean', float(record.get('rtt', 0)))),
                    'plr_std': float(record.get('plr_std', 0)),
                    'cpu_std': float(record.get('cpu_std', 0)),
                    'rtt_std': float(record.get('rtt_std', 0)),
                    'plr_x-x^': float(record.get('plr_x-x^', 0)),
                    'rtt_x-x^': float(record.get('rtt_x-x^', 0)),
                    'cpu_x-x^': float(record.get('cpu_x-x^', 0)),

                    'weighted_health_score': float(record.get('weighted_health_score', 0)),
                    'health_threshold': float(record.get('health_threshold', 0)),
                    'health_difference': float(record.get('health_difference', 0)),
                    'health_status': record.get('health_status', 'UNKNOWN'),
                    'anomaly_type': record.get('anomaly_type', 'NONE'),
                    'transceive_count': int(record.get('transceive_count', 0))
                }
                
                # Handle additional fields that might exist in the data
                for key in ['plr_mean', 'cpu_mean', 'rtt_mean', 'plr_std', 'cpu_std', 'rtt_std',
                           'plr_x-x^', 'rtt_x-x^', 'cpu_x-x^', 'plr_x^-x', 'rtt_x^-x', 'cpu_x^-x']:
                    if key in record:
                        value = record[key]
                        # Convert complex numbers to float (real part only)
                        if isinstance(value, complex):
                            extracted[key] = float(value.real)
                        else:
                            try:
                                extracted[key] = float(value)
                            except (ValueError, TypeError):
                                extracted[key] = 0.0
                
                self.extracted_data.append(extracted)
            except (ValueError, TypeError) as e:
                print(f"Warning: Skipping record {idx} due to extraction error: {e}")
                continue
        
        return self.extracted_data
    
    def process_metrics(self, processing_func: Optional[callable] = None) -> Any:
        """
        Process extracted metrics using custom logic.
        
        Args:
            processing_func: Optional callable that takes extracted_data as input
                           If None, returns the extracted data as-is
        
        Returns:
            Processed data (format depends on processing_func)
            
        Example:
            def custom_logic(data):
                return [d for d in data if d['health_status'] == 'UNHEALTHY']
            
            processor.process_metrics(custom_logic)
        """
        if not self.extracted_data:
            self.extract_metrics()
        
        if processing_func is None:
            # Default: return extracted data grouped by node
            return self._default_processing()
        
        return processing_func(self.extracted_data)
    
    def _default_processing(self) -> Dict[str, List[Dict]]:
        """
        Default processing: Group metrics by node_id.
        
        Returns:
            Dict mapping node_id to list of metrics
        """
        grouped = {}
        for record in self.extracted_data:
            node_id = record['node_id']
            if node_id not in grouped:
                grouped[node_id] = []
            grouped[node_id].append(record)
        logging.info("Default processing completed: grouped data by node_id")
        return grouped
    
    def _current_node_ids(self) -> List[str]:
        """Return a list of current node IDs."""
        grouped = self._default_processing()
        return list(grouped.keys())
    
    def get_baseline_and_noise(self, node_id: str) -> Dict[str, Any]:
        """
        Get baseline metrics and noise scales for a specific node.
        
        Args:
            node_id (str): The node identifier
            layer_profiles (dict): Layer profiles containing noise scales
            
        Returns:
            Dict containing:
                - baseline: Dict with cpu, rtt, plr from first record
                - noise_scales: List of noise scale values for the node's layer
                - layer: The identified layer name
                - node_id: The node identifier
                
        Raises:
            ValueError: If node layer cannot be identified or layer_profiles not set
        """
        layer_profiles = self.layer_profiles
        grouped = self._default_processing()
        #print(grouped); time.sleep(200)
        #grouped = self._default_processing(); print(grouped); #time.sleep(500)
        group_data = grouped[node_id]
        if not layer_profiles:
            raise ValueError("Layer profiles not set. Initialise with layer_profiles parameter.")
        
        if not group_data:
            raise ValueError(f"No data provided for node {node_id}")
        
        # Extract baseline from first record
        baseline = {
            'cpu': group_data[0]['cpu'],
            'rtt': group_data[0]['rtt'],
            'plr': group_data[0]['plr']
        }
        
        # Determine layer and get noise scales
        layer = None
        noise_scales = None
        
        if node_id == "CloudDBServer":
            noise_scales    = layer_profiles["CLOUD"]["noise"]
            noise_scales    = list(noise_scales.values())
            layer = 0
        elif node_id == "L1N_01":
            noise_scales    = layer_profiles["L1"]["noise"]
            noise_scales    = list(noise_scales.values())
            layer = 1
        elif node_id.startswith("L2N"):
            noise_scales    = layer_profiles["L2"]["noise"]
            noise_scales    = list(noise_scales.values()) 
            layer = 2   
        elif node_id.startswith("L3N"):
            noise_scales    = layer_profiles["L3"]["noise"] 
            noise_scales    = list(noise_scales.values())
            layer = 3
        elif node_id.startswith("L4N"):
            noise_scales    = layer_profiles["L4"]["noise"] 
            noise_scales    = list(noise_scales.values())
            layer = 4
        else:
            raise ValueError(f"Unknown node layer for {node_id}")
        
        
        return {
            'baseline': baseline,
            'noise_scales': noise_scales,
            'layer': layer,
            "node_id": node_id
        }

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of the extracted data.
        
        Returns:
            Dict containing summary statistics
        """
        if not self.extracted_data:
            self.extract_metrics()
        
        total_records = len(self.extracted_data)
        unique_nodes = len(set(r['node_id'] for r in self.extracted_data))
        
        health_status_counts = {}
        anomaly_type_counts = {}
        
        for record in self.extracted_data:
            status = record['health_status']
            anomaly = record['anomaly_type']
            
            health_status_counts[status] = health_status_counts.get(status, 0) + 1
            anomaly_type_counts[anomaly] = anomaly_type_counts.get(anomaly, 0) + 1
        
        return {
            'total_records': total_records,
            'unique_nodes': unique_nodes,
            'health_status_distribution': health_status_counts,
            'anomaly_type_distribution': anomaly_type_counts
        }
    
    def meanEstimate(self, mean, score, count):
        return (mean * count + score) / (count + 1)
    
    def stdEstimate(self, mean, std, score, count):
        if count == 0:
            logging.info("stdEstimate: count is 0, returning 0.0")
            time.sleep(5)
            return 0.0
        
        # Validate inputs
        if std < 0:
            logging.warning(f"stdEstimate: negative std={std}, using abs value")
            std = abs(std)
        
        new_mean = self.meanEstimate(mean, score, count)
        
        # Calculate variance
        old_variance = std ** 2
        variance_numerator = old_variance * count + (score - new_mean) ** 2
        variance = variance_numerator / (count + 1)
        
        # Ensure non-negative (handles floating-point errors)
        if variance < 0:
            if variance < -1e-10:  # More than just rounding error
                logging.warning(
                    f"stdEstimate: significantly negative variance={variance}, "
                    f"setting to 0. Inputs: mean={mean}, std={std}, score={score}, count={count}"
                )
            variance = 0.0
        
        new_std = sqrt(variance)
        
        logging.info(
            f"New Std Dev Calculation: mean={mean}, std={std}, score={score}, "
            f"count={count} => new_std={new_std}"
        )
        
        return new_std
    
    def updatedNode(self, node_id: str):
        current_time = datetime.now().astimezone()
        baseline_and_noise  = self.get_baseline_and_noise(node_id)
        baseline_node_id             = baseline_and_noise['node_id']
        logging.info(f"Confirmin Node ID: {node_id} == : {baseline_node_id}"); #time.sleep(1000)
        process_metrics     = self._default_processing()
        process_metrics         = process_metrics[node_id][0]
        
        process_metrics['timestamp']    = current_time.isoformat()
        
        print("process_metrics['transceive_count']: ", process_metrics['transceive_count']); time.sleep(1)

        process_metrics['baseline']     = baseline_and_noise['baseline']
        process_metrics['noise_scales'] = baseline_and_noise['noise_scales']
        process_metrics['layer']        = baseline_and_noise['layer'] 

        process_metrics['plr']          = process_metrics['plr']
        process_metrics['rtt']          = process_metrics['rtt']
        process_metrics['cpu']          = process_metrics['cpu']

        process_metrics['plr_x-x^']     = process_metrics['plr'] - process_metrics['plr_mean'] #process_metrics['plr_x-x^'] or 0.0
        process_metrics['rtt_x-x^']     = process_metrics['rtt'] - process_metrics['rtt_mean'] #process_metrics['rtt_x-x^'] or 0.0
        process_metrics['cpu_x-x^']     = process_metrics['cpu'] - process_metrics['cpu_mean'] #process_metrics['cpu_x-x^'] or 0.0

        process_metrics['plr_mean']     = self.meanEstimate(process_metrics['plr_mean'], process_metrics['plr'], process_metrics['transceive_count'] ) #process_metrics['plr_mean'] or 0.0
        process_metrics['rtt_mean']     = self.meanEstimate(process_metrics['rtt_mean'], process_metrics['rtt'], process_metrics['transceive_count'] ) #process_metrics['rtt_mean'] or 0.0 
        process_metrics['cpu_mean']     = self.meanEstimate(process_metrics['cpu_mean'], process_metrics['cpu'], process_metrics['transceive_count']) #process_metrics['cpu_mean'] or 0.0

        process_metrics['plr_std']      = self.stdEstimate(process_metrics['plr_mean'], process_metrics['plr_std'], process_metrics['plr'], process_metrics['transceive_count']) #process_metrics['plr_std'] or 0.0
        process_metrics['rtt_std']      = self.stdEstimate(process_metrics['rtt_mean'], process_metrics['rtt_std'], process_metrics['rtt'], process_metrics['transceive_count']) #process_metrics['rtt_std'] or 0.0 
        process_metrics['cpu_std']      = self.stdEstimate(process_metrics['cpu_mean'], process_metrics['cpu_std'], process_metrics['cpu'], process_metrics['transceive_count']) #process_metrics['cpu_std'] or 0.0
        

        process_metrics['transceive_count']    = int( process_metrics['transceive_count']) +1  

        logging.info("process_metrics: ", process_metrics); time.sleep(4)    


        #self.transceive_inputs['transceive_count'] = str(default_value) + 1
        #self.transceive_inputs['plr_mean']      = process_metrics['plr']
        #self.transceive_inputs['rtt_mean']      = process_metrics['rtt']
        #self.transceive_inputs['cpu_mean']      = process_metrics['cpu']
        

        return process_metrics

    def processTransmissionNodes(self, processing_func: Optional[callable] = None) -> Any:
        """
        Process extracted metrics using custom logic.
        
        Args:
            processing_func: Optional callable that takes extracted_data as input
                           If None, returns the extracted data as-is
        
        Returns:
            Processed data (format depends on processing_func)
            
        Example:
            def custom_logic(data):
                return [d for d in data if d['health_status'] == 'UNHEALTHY']
            
            processor.process_metrics(custom_logic)
        """
        if not self.extracted_data:
            self.extract_metrics()
        
        print("self.extract_metrics(): ", self.extract_metrics()); #time.sleep(400)    

        processed_data = []
        for node_id in self._current_node_ids():
            print("Processing Node ID: ", node_id)  
            processed_data.append(
               #{ node_id :  self.updatedNode(node_id)} 
               self.updatedNode(node_id)
            )

        print("processed_data: ", processed_data); time.sleep(4)
        if processing_func is None:
            # Default: return extracted data grouped by node
            return processed_data #self._default_processing()
        
        return processing_func(self.extracted_data)
    
    def process_to_json(self) -> str:
        """
        Process all data and return as JSON string.
        
        Returns:
            JSON string containing processed data with baseline and noise information
        """
        processed_data = self.processTransmissionNodes()
        #print("Processed Data for JSON: ", processed_data); time.sleep(2000)
        return json.dumps(processed_data, indent=2)
    

# ============================================================================
# TESTS
# ============================================================================

def test_node_transceive_processor():
    """Test suite for NodeTransceiveProcessorInit class."""
    
    # Test data
    sample_json = json.dumps([
        {
            "timestamp": "2025-10-17 13:14:17.569243+00:00",
            "node_id": "CloudDBServer",
            "plr": "0.0077",
            "cpu": "0.0248",
            "rtt": "27.3268",
            "weighted_health_score": "0.39941517958613687",
            "health_threshold": "0.39941517958613687",
            "health_difference": "0.0",
            "health_status": "HEALTHY",
            "anomaly_type": "NONE",
            "transceive_count": "0"
        },
        {
            "timestamp": "2025-10-17 13:14:21.688208+00:00",
            "node_id": "L1N_01",
            "plr": "0.404",
            "cpu": "0.2936",
            "rtt": "25.4321",
            "weighted_health_score": "0.39999999999999714",
            "health_threshold": "0.39999999999999714",
            "health_difference": "0.0",
            "health_status": "HEALTHY",
            "anomaly_type": "NONE",
            "transceive_count": "0"
        }
    ])
    
    print("Running NodeTransceiveProcessorInit Tests...")
    print("=" * 60)
    
    # Test 1: Valid initialisation
    print("\n✓ Test 1: Valid initialisation")
    try:
        processor = NodeTransceiveProcessorInit(sample_json)
        assert len(processor.raw_data) == 2
        print("  PASSED: Processor initialised with 2 records")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 2: Invalid JSON
    print("\n✓ Test 2: Invalid JSON handling")
    try:
        processor = NodeTransceiveProcessorInit("invalid json")
        print("  FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"  PASSED: Correctly raised ValueError - {e}")
    
    # Test 3: Empty string
    print("\n✓ Test 3: Empty string handling")
    try:
        processor = NodeTransceiveProcessorInit("")
        print("  FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"  PASSED: Correctly raised ValueError - {e}")
    
    # Test 4: Extract metrics
    print("\n✓ Test 4: Extract metrics")
    try:
        processor = NodeTransceiveProcessorInit(sample_json)
        extracted = processor.extract_metrics()
        assert len(extracted) == 2
        assert extracted[0]['node_id'] == 'CloudDBServer'
        assert extracted[0]['plr'] == 0.0077
        print(f"  PASSED: Extracted {len(extracted)} records successfully")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 5: Process metrics with custom function
    print("\n✓ Test 5: Custom processing function")
    try:
        processor = NodeTransceiveProcessorInit(sample_json)
        processor.extract_metrics()
        
        def filter_high_plr(data):
            return [d for d in data if d['plr'] > 0.1]
        
        result = processor.process_metrics(filter_high_plr)
        assert len(result) == 1
        assert result[0]['node_id'] == 'L1N_01'
        print(f"  PASSED: Custom filter returned {len(result)} record(s)")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 6: Default processing (grouping)
    print("\n✓ Test 6: Default processing (grouping by node)")
    try:
        processor = NodeTransceiveProcessorInit(sample_json)
        grouped = processor.process_metrics()
        assert 'CloudDBServer' in grouped
        assert 'L1N_01' in grouped
        assert len(grouped['CloudDBServer']) == 1
        print(f"  PASSED: Grouped into {len(grouped)} nodes")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 7: Summary statistics
    print("\n✓ Test 7: Summary statistics")
    try:
        processor = NodeTransceiveProcessorInit(sample_json)
        stats = processor.get_summary_stats()
        assert stats['total_records'] == 2
        assert stats['unique_nodes'] == 2
        assert stats['health_status_distribution']['HEALTHY'] == 2
        print(f"  PASSED: Generated summary with {stats['total_records']} records")
        print(f"           Unique nodes: {stats['unique_nodes']}")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    print("\n" + "=" * 60)
    print("Test Suite Completed!")


if __name__ == "__main__":
    test_node_transceive_processor()
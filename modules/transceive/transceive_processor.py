"""
Transceive Processor Module

This module provides functionality to process node metrics data from JSON format.
It extracts key metrics and prepares them for further analysis.

Filename: modules/transceive/transceive_processor.py
"""

import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from logs import logging

class NodeTransceiveProcessor:
    """
    Processes transceive node metrics data from JSON input.
    
    Attributes:
        raw_data (List[Dict]): The raw JSON data as a list of dictionaries
        extracted_data (List[Dict]): Extracted and validated metrics data
    """
    
    def __init__(self, json_data_dump: str):
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
                    'weighted_health_score': float(record.get('weighted_health_score', 0)),
                    'health_threshold': float(record.get('health_threshold', 0)),
                    'health_difference': float(record.get('health_difference', 0)),
                    'health_status': record.get('health_status', 'UNKNOWN'),
                    'anomaly_type': record.get('anomaly_type', 'NONE'),
                    'transceive_count': int(record.get('transceive_count', 0))
                }
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


# ============================================================================
# TESTS
# ============================================================================

def test_node_transceive_processor():
    """Test suite for NodeTransceiveProcessor class."""
    
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
    
    print("Running NodeTransceiveProcessor Tests...")
    print("=" * 60)
    
    # Test 1: Valid initialisation
    print("\n✓ Test 1: Valid initialisation")
    try:
        processor = NodeTransceiveProcessor(sample_json)
        assert len(processor.raw_data) == 2
        print("  PASSED: Processor initialised with 2 records")
    except Exception as e:
        print(f"  FAILED: {e}")
    
    # Test 2: Invalid JSON
    print("\n✓ Test 2: Invalid JSON handling")
    try:
        processor = NodeTransceiveProcessor("invalid json")
        print("  FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"  PASSED: Correctly raised ValueError - {e}")
    
    # Test 3: Empty string
    print("\n✓ Test 3: Empty string handling")
    try:
        processor = NodeTransceiveProcessor("")
        print("  FAILED: Should have raised ValueError")
    except ValueError as e:
        print(f"  PASSED: Correctly raised ValueError - {e}")
    
    # Test 4: Extract metrics
    print("\n✓ Test 4: Extract metrics")
    try:
        processor = NodeTransceiveProcessor(sample_json)
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
        processor = NodeTransceiveProcessor(sample_json)
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
        processor = NodeTransceiveProcessor(sample_json)
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
        processor = NodeTransceiveProcessor(sample_json)
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
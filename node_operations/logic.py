#updated_logic.py

import networkx as nx
import time 
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from typing import Dict, List, Optional,Tuple
from collections import defaultdict
import logging
import math
from datetime import datetime, timedelta
from time import sleep
import json

# Define sensor types with descriptions and expected ranges
SENSOR_TYPES = {
    "001": {
        "name": "Temperature",
        "unit": "°C",
        "expected_range": (-20, 50)  # Example range for temperature
    },
    "002": {
        "name": "Humidity",
        "unit": "%RH",
        "expected_range": (0, 100)   # 0-100% for humidity
    },
    "003": {
        "name": "Pressure",
        "unit": "hPa",
        "expected_range": (900, 1100)  # Atmospheric pressure range
    },
    "004": {
        "name": "Motion",
        "unit": "count",
        "expected_range": (0, 1)      # Binary motion detection (0=no, 1=yes)
    }
}

# Define a simple FogNode class
class FogNode:
    def __init__(self, node_id, name, parent, children, energy_level, resources):
        self.node_id        = node_id       # Unique integer identifier
        self.name           = name          # Original node name from the network
        self.parent         = parent        # Parent node id (None for root)
        self.children       = children      # Set of child node ids
        self.energy_level   = energy_level
        self.resources      = resources

    def __repr__(self):
        return (f"FogNode(node_id={self.node_id}, name='{self.name}', "
                f"parent={self.parent}, children={self.children}, "
                f"energy_level={self.energy_level}, resources={self.resources})")


class NetworkHierarchy(object):
    def __init__(self, *args):
        self.hierarchy = {}  # Store the node -> children relationships
        self.G = None        # Cached networkx DiGraph
        super(NetworkHierarchy, self).__init__(*args)
    
    def nodes(self) -> tuple:
        """Dynamically generates all node names in the hierarchy."""
        # Layer 2: 4 nodes (L2N_01 to L2N_04)
        l2_nodes = [f"L2N_{i:02d}" for i in range(1, 5)]
        
        # Layer 3: 12 nodes (L3N_01 to L3N_12)
        l3_nodes = [f"L3N_{i:02d}" for i in range(1, 13)]
        
        # Layer 4: 36 nodes (L4N_01 to L4N_36)
        l4_nodes = [f"L4N_{i:02d}" for i in range(1, 37)]
        
        # Sensors: One per L4 node (Sen_L4N_XX_TypeCode)
        sensor_mappings = self.generate_sensor_mappings()
        sensor_nodes = [
            f"Sen_{l4_node}_{sensor_mappings[l4_node]}" 
            for l4_node in l4_nodes
        ]
        
        return (["CloudDBServer", "L1Node"] + 
                l2_nodes + l3_nodes + l4_nodes ) # the actual nodes
        return (["CloudDBServer", "L1Node"] + 
                l2_nodes + l3_nodes + l4_nodes + 
                sensor_nodes)   # fog nodes with sensors
        
    def get_all_descendants(self, node_name: str) -> List[str]:
        """Return all descendants (children, grandchildren, etc.) of a given node."""
        if not self.hierarchy:
            print("hierarchy is being built"); #time.sleep(1)
            self.create_network()  # Ensure hierarchy is built
        
        descendants = []
        try:
            self.hierarchy = {node: list(self.G.successors(node)) for node in self.G.nodes}
        except: 
            pass   
        def recurse(current_node):
            children = []
            if not current_node.startswith("L4"):
                children = self.hierarchy.get(current_node, [])
                for child in children:
                    if not child.startswith("Sen_"):
                        descendants.append(child)
                    recurse(child)
        recurse(node_name)
        return descendants

    def get_all_ancestors(self, node_name: str) -> List[str]:
        """
        Return all ancestors (parent, grandparent, etc.) of a given node.

        Args:
            node_name: Name of the node (e.g., 'L4N_12', 'Sen_L4N_01_001')

        Returns:
            List of ancestor node names in order from immediate parent to highest ancestor.
        """
        ancestors = []

        # Ensure network graph exists
        G, _, _ = self.create_network()

        def recurse(current_node):
            parents = list(G.predecessors(current_node))
            for parent in parents:
                ancestors.append(parent)
                recurse(parent)

        recurse(node_name)
        return ancestors

    def generate_sensor_mappings(self):
        """Assigns sensor types to L4 nodes in a repeating pattern using sensor codes."""
        sensor_codes = list(SENSOR_TYPES.keys())  # ['001', '002', '003', '004']
        type_cycle = sensor_codes * 9  # 4 types × 9 = 36 nodes
        return {
            f"L4N_{i:02d}": type_cycle[i-1]  # i starts at 1
            for i in range(1, 37)
        }
        
    def create_network(self):
        """Optimised network creation with position calculation and efficient node connections."""
        G = nx.DiGraph()
        pos = {}  # Node positions for visualisation
        
        # Layer definitions
        sensor_mappings = self.generate_sensor_mappings()
        layers = {
            'cloud': ["CloudDBServer"],
            'l1': ["L1Node"],
            'l2': [f"L2N_{i:02d}" for i in range(1, 5)],
            'l3': [f"L3N_{i:02d}" for i in range(1, 13)],
            'l4': [f"L4N_{i:02d}" for i in range(1, 37)],
            'sensors': [f"Sen_{l4}_{sensor_mappings[l4]}" for l4 in [f"L4N_{i:02d}" for i in range(1, 37)]]
        }
        #print("layers: ", layers); time.sleep(3)
        # Add all nodes with positions
        pos[layers['cloud'][0]] = (0, 5)
        pos[layers['l1'][0]] = (0, 4)
        
        # Position L2 nodes (-1.5 to 1.5 spread)
        for i, node in enumerate(layers['l2']):
            pos[node] = (-1.5 + i, 3)
            
        # Position L3 nodes (-2 to 2 spread)
        for i, node in enumerate(layers['l3']):
            pos[node] = (-2 + i * (4/11), 2)  # 11 intervals between 12 nodes
            
        # Position L4 nodes (-2 to 2 spread)
        for i, node in enumerate(layers['l4']):
            pos[node] = (-2 + i * (4/35), 1)  # 35 intervals between 36 nodes
            
        # Position sensors below their L4 nodes
        for sensor in layers['sensors']:
            # Extract L4 node name: "Sen_L4N_01_001" -> "L4N_01"
            l4_node = '_'.join(sensor.split('_')[1:3])  # Join "L4N" and "01"
            pos[sensor] = (pos[l4_node][0], 0)
            
        # Add all nodes and edges
        nodes = self.nodes()
        G.add_nodes_from(nodes)
        
        # Cloud → L1
        G.add_edge(layers['cloud'][0], layers['l1'][0])
        
        # L1 → L2 (all connections)
        for l2_node in layers['l2']:
            G.add_edge(layers['l1'][0], l2_node)
            
        # L2 → L3 (3 connections per L2)
        for l2_idx, l2_node in enumerate(layers['l2']):
            for l3_idx in range(3):
                target_idx = l2_idx * 3 + l3_idx
                if target_idx < len(layers['l3']):
                    G.add_edge(l2_node, layers['l3'][target_idx])
                    
        # L3 → L4 (3 connections per L3)
        for l3_idx, l3_node in enumerate(layers['l3']):
            for l4_idx in range(3):
                target_idx = l3_idx * 3 + l4_idx
                if target_idx < len(layers['l4']):
                    l4_node = layers['l4'][target_idx]
                    G.add_edge(l3_node, l4_node)
                    # Connect to corresponding sensor
                    G.add_edge(l4_node, f"Sen_{l4_node}_{sensor_mappings[l4_node]}")
        #print("G.nodes: ", G.nodes); #time.sleep(300)
        try:
            self.hierarchy = {node: list(self.G.successors(node)) for node in self.G.nodes}
            #print("self.hierarchy: ", self.hierarchy)
        except:
            pass
        
        
        self.G = G  # Cache the graph for later use
        #print("self.G.nodes: ", self.G.nodes); sleep(300)
            
        return G, pos, sensor_mappings

    def get_sensor_info(self, sensor_code) -> dict:
        """Helper method to get sensor information from sensor code."""
        if sensor_code in SENSOR_TYPES:
            return {
                "sensor_code": sensor_code,
                "sensor_name": SENSOR_TYPES[sensor_code]["name"],
                "unit": SENSOR_TYPES[sensor_code]["unit"],
                "expected_range": SENSOR_TYPES[sensor_code]["expected_range"]
            }
        return {}

    def create_fog_nodes(self) -> tuple:
        """
        Create a dictionary of FogNode objects for the whole network.
        Each node is assigned a unique integer ID and sensor nodes receive proper sensor type info.
        Node compute/network resources are realistically allocated based on hierarchical layer.
        """
        G, pos, sensor_mappings = self.create_network()
        fog_nodes = {}
        name_to_id = {}
        node_id_counter = 1

        # Layer-specific resource presets
        layer_resources = {
            "L4": {'cpu': 0.5, 'memory': 512, 'bandwidth': 10},      # Field Gateway
            "L3": {'cpu': 1.5, 'memory': 4096, 'bandwidth': 50},     # Micro Data Centre
            "L2": {'cpu': 3.0, 'memory': 8192, 'bandwidth': 100},    # Edge Aggregator
            "L1": {'cpu': 4.0, 'memory': 16384, 'bandwidth': 200},   # Fog Orchestrator
            "Cloud": {'cpu': 8.0, 'memory': 32768, 'bandwidth': 1000}# Cloud Server
        }

        for node in G.nodes():
            name_to_id[node] = node_id_counter
            energy_level = 100.0
            resources = {}

            # Determine node layer by prefix
            if node.startswith("L4N"):
                resources.update(layer_resources["L4"])
            elif node.startswith("L3N"):
                resources.update(layer_resources["L3"])
            elif node.startswith("L2N"):
                resources.update(layer_resources["L2"])
            elif node.startswith("L1N"):
                resources.update(layer_resources["L1"])
            elif node.startswith("Cloud"):
                resources.update(layer_resources["Cloud"])
            elif node.startswith("Sen_"):
                # Sensor node: extract and associate sensor type
                parts = node.split('_')
                if len(parts) >= 4:
                    sensor_code = parts[3]
                    sensor_info = self.get_sensor_info(sensor_code)
                    if sensor_info:
                        resources["sensor_info"] = sensor_info
                    else:
                        resources["sensor_code"] = sensor_code
                # Attach to L4 gateway defaults
                resources.update(layer_resources["L4"])
            else:
                # Default fallback (optional)
                resources.update({'cpu': 1.0, 'memory': 1024, 'bandwidth': 10})

            # Create FogNode object
            fog_nodes[node_id_counter] = FogNode(
                node_id=node_id_counter,
                name=node,
                parent=None,
                children=set(),
                energy_level=energy_level,
                resources=resources
            )
            node_id_counter += 1

        # Set parent and children relationships from the graph structure
        for node in G.nodes():
            current_id = name_to_id[node]
            predecessors = list(G.predecessors(node))
            if predecessors:
                fog_nodes[current_id].parent = name_to_id[predecessors[0]]
            for child in G.successors(node):
                child_id = name_to_id[child]
                fog_nodes[current_id].children.add(child_id)

        print(f"Created {len(fog_nodes)} fog nodes successfully")
        return fog_nodes, name_to_id

    def get_sensor_nodes_by_type(self, fog_nodes, sensor_type_code) -> list:
        """
        Retrieve all sensor nodes of a specific type.
        
        Args:
            fog_nodes: Dictionary of FogNode objects
            sensor_type_code: Sensor type code (e.g., '001' for Temperature)
            
        Returns:
            List of FogNode objects that are sensors of the specified type
        """
        sensor_nodes = []
        for node in fog_nodes.values():
            if (node.name.startswith('Sen_') and 
                'sensor_info' in node.resources and
                node.resources['sensor_info']['sensor_code'] == sensor_type_code):
                sensor_nodes.append(node)
        return sensor_nodes

    def display_sensor_summary(self, fog_nodes):
        """Display a summary of all sensor types in the network."""
        sensor_counts = {}
        
        for node in fog_nodes.values():
            if node.name.startswith('Sen_') and 'sensor_info' in node.resources:
                sensor_info = node.resources['sensor_info']
                sensor_name = sensor_info['sensor_name']
                if sensor_name not in sensor_counts:
                    sensor_counts[sensor_name] = 0
                sensor_counts[sensor_name] += 1



class fogNodeCharacterisation(object):
    """_summary_

    Args:
        object (_type_): Charactersation of network-wides fog nodes
    """
    def __init__(self, ALPHA=0.2, *args):
        self.G                  = {}
        self.WEIGHTS            = {"PLR": 0.3, "Response": 0.2, "CPU": 0.3 }
        self.STATIC_THRESHOLDS  = {"PLR": 10, "Response": 200, "CPU": 80 }
        if not (0 < ALPHA < 1):
            raise ValueError("alpha must be between 0 and 1.")
        self.ALPHA              = ALPHA  # Smoothing factor for EMA
        self.current_thresholds = {} # Current adaptive thresholds per node
        self.M2                 = 0.0  # Sum of squared differences
        
        self.metrics_stats      = {
                                    "PLR": {"mu": 5.0, "sigma": 2.5, "n": 0, "sum": 0, "sum_sq": 0},
                                    "Response": {"mu": 100.0, "sigma": 50.0, "n": 0, "sum": 0, "sum_sq": 0},
                                    "CPU": {"mu": 50.0, "sigma": 20.0, "n": 0, "sum": 0, "sum_sq": 0}
                                }
        # mu        Historical mean (μ) of the metric. Initial guess or prior. 
        #           PLR=5.0 (5% packet loss), Response=100, CPU, 50.0
        # sigma     Historical standard deviation (σ). Measures variability. PLR=2.5 (moderate variance)
        # n         Number of observations seen so far. Starts at 0 (no data). PLR,Response,CPU
        # sum       Running sum of all observed values. Used to compute mu.  PLR,Response,CPU
        # sum_sq    Running sum of squared values. Used to compute sigma.  PLR,Response,CPU
        self.threshold_bounds = {}  # Min/max bounds for thresholds
        super(fogNodeCharacterisation, self).__init__(*args)
    
    def _generate_cloud_metrics(self) -> Dict:
        """Generate optimised cloud server metrics."""
        return {
            "node": "CloudDBServer",
            "PLR": round(random.uniform(0, 2), 2),
            "Response": round(random.uniform(5, 30), 2),
            "CPU": round(random.uniform(10, 40), 2),
            "status": "HEALTHY"
        }

    def _generate_l1_metrics(self) -> Dict:
        """Generate L1 node metrics with realistic variations."""
        return {
            "node": "L1Node",
            "PLR": round(random.uniform(2, 6), 2),
            "Response": round(random.uniform(20, 45), 2),
            "CPU": round(random.uniform(30, 60), 2),
            "status": "HEALTHY"
        }

    def _generate_standard_metrics(self, failure_count: int, max_failures: int) -> Dict:
        """Generate standard node metrics with natural variations."""
        # Decide if we should force a failure
        force_failure = (failure_count < max_failures and random.random() < 0.1)
        
        if force_failure:
            failure_count+=1
            return {
                "PLR": round(random.uniform(10, 20), 2),
                "Response": round(random.uniform(200, 300), 2),
                "CPU": round(random.uniform(80, 100), 2),
                "status": "FAULTY", 
                "failure_count": failure_count
            }
        else:
            return {
                "PLR": round(random.uniform(0, 8), 2),
                "Response": round(random.uniform(50, 190), 2),
                "CPU": round(random.uniform(20, 75), 2),
                "status": "HEALTHY", 
                "failure_count": failure_count
            }
    
    def _standard_running_cloud_metrics(self, total_child_workload: float) -> Dict:
        """Generate realistic Cloud metrics based on cumulative child workload."""
        # Baseline values
        base_cpu = 10.0
        base_plr = 0.5
        base_response = 10.0

        # Scaling parameters
        cpu_scale = 3.0
        plr_scale = 0.015
        response_scale = 0.2

        # Adjusted metrics based on workload
        cpu = min(95.0, base_cpu + cpu_scale * math.log(1 + total_child_workload))
        plr = min(5.0, base_plr + plr_scale * math.sqrt(total_child_workload))
        response = min(100.0, base_response + response_scale * total_child_workload)

        return {
            "node": "CloudDBServer",
            "PLR": round(plr, 2),
            "Response": round(response, 2),
            "CPU": round(cpu, 2),
            "status": "HEALTHY"
        }

    def _standard_running_l1_metrics(self, total_child_workload: float) -> Dict:
        """Generate realistic L1 node metrics based on cumulative child workload."""
        # Baseline values
        base_cpu = 20.0
        base_plr = 1.0
        base_response = 15.0

        # Scaling parameters
        cpu_scale = 5.0
        plr_scale = 0.03
        response_scale = 0.4

        # Adjusted metrics based on workload
        cpu = min(98.0, base_cpu + cpu_scale * math.log(1 + total_child_workload))
        plr = min(10.0, base_plr + plr_scale * math.sqrt(total_child_workload))
        response = min(150.0, base_response + response_scale * total_child_workload)

        return {
            "node": "L1Node",
            "PLR": round(plr, 2),
            "Response": round(response, 2),
            "CPU": round(cpu, 2),
            "status": "HEALTHY"
        }

    def _standard_running_metrics(self, layer: int, num_children: int, failure_count: int, max_failures: int) -> Dict:
        """
        Generate runtime node metrics based on workload (layer, children) and simulate occasional faults.

        Parameters:
        - layer: depth of node in the hierarchy (lower = edge, higher = root)
        - num_children: number of child nodes (used to infer workload)
        - failure_count: current number of injected failures
        - max_failures: threshold to limit total faults

        Returns:
        - Dictionary with PLR, Response, CPU, status, and updated failure_count
        """
        
        # Failure injection
        force_failure = (failure_count < max_failures and random.random() < 0.1)
        if force_failure:
            failure_count += 1
            return {
                "PLR": round(random.uniform(10, 20), 2),
                "Response": round(random.uniform(200, 300), 2),
                "CPU": round(random.uniform(80, 100), 2),
                "status": "FAULTY",
                "failure_count": failure_count
            }

        # Simulated workload contributions
        receive_load = num_children * random.uniform(1.0, 2.0)
        process_load = receive_load * (1.0 + 0.1 * layer)
        forward_load = receive_load * (0.8 if layer > 0 else 0)

        # Base metric values
        base_plr = 1.0 + 0.3 * layer
        base_response = 60 + 5 * layer
        base_cpu = 30 + 3 * layer

        # Metric calculations
        plr = base_plr + 0.2 * receive_load + 0.1 * forward_load + random.uniform(-0.5, 0.5)
        response = base_response + 2.5 * process_load + 1.5 * forward_load + random.uniform(-10, 10)
        cpu = base_cpu + 2.8 * process_load + random.uniform(-5, 5)

        # Clamp and round values
        cpu = min(cpu, 100.0)
        plr = max(plr, 0.0)

        return {
            "PLR": round(plr, 2),
            "Response": round(response, 2),
            "CPU": round(cpu, 2),
            "status": "HEALTHY",
            "failure_count": failure_count
        }
    
    def initialise_metrics_global(self, node_metrics, node_details, failure_count: int, max_failures: int) -> Dict:
        """Process node metrics and calculate health metrics"""
        print(f"Processing {type(node_metrics).__name__} with {len(node_metrics)} nodes")
        initialisation_node_metrics = {}
        for node, metric in node_metrics.items():
            new_plr = new_response = new_cpu =  None
            
            if node == "CloudDBServer":
                #node_details = {"node_id": counter+1}
                cloud_node_metric = self._generate_cloud_metrics()
                for metric in ["PLR", "Response", "CPU",  "status"]: 
                    node_details[metric] = cloud_node_metric[metric]       
                node_metrics[node] = node_details
                initialisation_node_metrics[node] = node_details
                #counter+=1
            if node == "L1Node":
                #node_details = {"node_id": counter}
                l1_node_metric = self._generate_l1_metrics()
                print("l1_node_metric: ", l1_node_metric)
                for metric in ["PLR", "Response", "CPU",  "status"]:
                    node_details[metric] = l1_node_metric[metric]
                print("node_details: ", node_details)
                node_metrics[node] = node_details
                initialisation_node_metrics[node] = node_details
            
            if (node != "CloudDBServer" and node != "L1Node" and not node.startswith('Sen_')):   
                standard_metric = self._generate_standard_metrics(failure_count, max_failures )
                for metric in ["PLR", "Response", "CPU",  "status"]: 
                    node_details[metric] = standard_metric[metric]       
                
                '''for key in metric.keys(): 
                    new_plr = metric["PLR"]
                    new_response = metric["Response"]
                    new_cpu = metric["CPU"]'''

                # Calculate health metrics
                meanAndsd = self.muAndsigma_global(new_plr, new_response, new_cpu)
                healthmetric = self.healthMetric(
                    new_plr, new_response, new_cpu, 
                    meanAndsd["PLR"]["mu"], meanAndsd["PLR"]["sigma"],  
                    meanAndsd["Response"]["mu"], meanAndsd["Response"]["sigma"], 
                    meanAndsd["CPU"]["mu"], meanAndsd["CPU"]["sigma"]
                )
                
                node_id = metric["node_id"]
                threshold = self.update_threshold_with_bounds(node_id, healthmetric)
                            
                # Update metric with health data
                node_details["health_metrics"] = healthmetric
                node_details["threshold"] = threshold
                node_details["last_updated"] = datetime.now().isoformat()
                
                health_diff = healthmetric - threshold
                
                if health_diff != 0:
                    print(f"Anomaly detected for {node}, waiting 3 seconds...")
                    time.sleep(1)
                    metric["health_diff"] = health_diff
                #initialisation_node_metrics[node] = metric
                node_metrics[node] = node_details
                initialisation_node_metrics[node] = node_details
                
                time.sleep(3)  # Brief pause between nodes
                
                nh = NetworkHierarchy()
                nh.create_network()  # Ensure hierarchy is built
                print("self.hierarchy:", self.hierarchy); sleep(300)
                descendants = nh.get_all_descendants(node)
        with open('initialisation_node_metric.json', 'w') as fp:
            json.dump(initialisation_node_metrics, fp, indent=4)
        return node_metrics

    def initialise_metrics_per_node(self, node_metrics) -> Dict:
        """Initialise node cold startup metrics and calculate health metrics"""
        print(f"Processing {type(node_metrics).__name__} with {len(node_metrics)} nodes")
        print("self.G: ", self.G); sleep(3)
        initialisation_node_metrics = {}
        for node, metric in node_metrics.items():
            new_plr = new_response = new_cpu =  None            
            nh = NetworkHierarchy()
            #nh.create_network()  # Ensure hierarchy is built
            G, pos, sensor_mappings = nh.create_network()  # Ensure hierarchy is built
            self.G = G
            if not (node.startswith("L4") or node.startswith("Sen_")):
                descendants = nh.get_all_descendants(node)
                #print("node: ", node, "\ndescendants: ", descendants); time.sleep(10)
            
            if node == "CloudDBServer":
                node_details = {}
                #node_details = {"node_id": counter+1}
                cloud_node_metric = self._generate_cloud_metrics()
                for metrica in ["PLR", "Response", "CPU",  "status"]: 
                    node_details[metrica] = cloud_node_metric[metrica]       
                #print("node_details: ", node_details)
                
                for key in metric.keys(): 
                    new_plr = metric["PLR"]; new_response = metric["Response"]; new_cpu = metric["CPU"];  node_id = metric["node_id"]
                # Calculate health metrics
                meanAndsd = self.muAndsigma_per_node(node_id, new_plr, new_response, new_cpu)
                healthmetric = self.healthMetric(
                    new_plr, new_response, new_cpu, 
                    meanAndsd["PLR"]["mu"], meanAndsd["PLR"]["sigma"],  
                    meanAndsd["Response"]["mu"], meanAndsd["Response"]["sigma"], 
                    meanAndsd["CPU"]["mu"], meanAndsd["CPU"]["sigma"]
                )
                
                node_id = metric["node_id"]
                threshold = self.update_threshold_with_bounds(node_id, healthmetric)
                                
                # Update metric with health data
                metric["health_metrics"] = healthmetric; metric["threshold"] = threshold; metric["last_updated"] = datetime.now().isoformat(); health_diff = healthmetric - threshold 
                
                nodeStatus = self.healthStatusold(metric, healthmetric, threshold)
                #nodeStatus = self.healthStatus(metric, healthmetric, threshold)
                metric["healthyNode"] = nodeStatus
                
                initialisation_node_metrics[node] = metric                
                time.sleep(1)  # Brief pause between nodes
                #counter+=1
                
            if node == "L1Node":
                node_details = {}
                #node_details = {"node_id": counter}
                l1_node_metric = self._generate_l1_metrics()
                for metrica in ["PLR", "Response", "CPU",  "status"]:
                    node_details[metrica] = l1_node_metric[metrica]
                
                for key in metric.keys(): 
                    new_plr = metric["PLR"]; new_response = metric["Response"]; new_cpu = metric["CPU"];   node_id = metric["node_id"]
                
                # Calculate health metrics
                meanAndsd = self.muAndsigma_per_node(node_id, new_plr, new_response)
                healthmetric = self.healthMetric(
                    new_plr, new_response, new_cpu, 
                    meanAndsd["PLR"]["mu"], meanAndsd["PLR"]["sigma"],  
                    meanAndsd["Response"]["mu"], meanAndsd["Response"]["sigma"], 
                    meanAndsd["CPU"]["mu"], meanAndsd["CPU"]["sigma"]
                )
                
                node_id = metric["node_id"]
                threshold = self.update_threshold_with_bounds(node_id, healthmetric)
                                
                # Update metric with health data
                metric["health_metrics"] = healthmetric; metric["threshold"] = threshold; metric["last_updated"] = datetime.now().isoformat(); health_diff = healthmetric - threshold
                
                nodeStatus = self.healthStatus(metric, healthmetric, threshold)
                metric["healthyNode"] = nodeStatus
                
                initialisation_node_metrics[node] = metric
                time.sleep(1)  # Brief pause between nodes
            
            if (node != "CloudDBServer" and node != "L1Node" and not node.startswith('Sen_')):   
                node_details = {}
                for key in metric.keys(): 
                    new_plr = metric["PLR"]; new_response = metric["Response"]; new_cpu = metric["CPU"];  node_id = metric["node_id"]
                # Calculate health metrics
                meanAndsd = self.muAndsigma_per_node(node_id, new_plr, new_response, new_cpu)
                healthmetric = self.healthMetric(
                    new_plr, new_response, new_cpu, 
                    meanAndsd["PLR"]["mu"], meanAndsd["PLR"]["sigma"],  
                    meanAndsd["Response"]["mu"], meanAndsd["Response"]["sigma"], 
                    meanAndsd["CPU"]["mu"], meanAndsd["CPU"]["sigma"]
                )
                
                node_id = metric["node_id"]
                threshold = self.update_threshold_with_bounds(node_id, healthmetric)
                                
                # Update metric with health data
                metric["health_metrics"] = healthmetric
                metric["threshold"] = threshold
                metric["last_updated"] = datetime.now().isoformat()
                
                health_diff = healthmetric - threshold
                
                nodeStatus = self.healthStatusold(metric, healthmetric, threshold)
                #nodeStatus = self.healthStatusold(metric, healthmetric, threshold)
                metric["healthyNode"] = nodeStatus
                
                initialisation_node_metrics[node] = metric
                
                time.sleep(1)  # Brief pause between nodes

        with open('initialisation_node_metric.json', 'w') as fp:
            json.dump(initialisation_node_metrics, fp, indent=4)
        return node_metrics

    def muAndsigma_global(self, new_plr, new_response, new_cpu) -> Dict:
        metrics_stats = self.metrics_stats
        updated_metrics_stats = {
            "PLR":{},
            "Response":{},
            "CPU":{},
        }
        
        # Update PLR statistics
        metrics_stats["PLR"]["n"] += 1
        metrics_stats["PLR"]["sum"] += new_plr
        metrics_stats["PLR"]["sum_sq"] += new_plr ** 2

        # Recompute mu and sigma
        n = metrics_stats["PLR"]["n"]
        sum_val = metrics_stats["PLR"]["sum"]
        sum_sq  = metrics_stats["PLR"]["sum_sq"]

        metrics_stats["PLR"]["mu"] = sum_val / n 
        mu =  metrics_stats["PLR"]["mu"] # µ (mu)
        #metrics_stats["PLR"]["sigma"] = math.sqrt((sum_sq / n) - (sum_val / n) ** 2)
        metrics_stats["PLR"]["sigma"] = 0 if abs((sum_sq / n) - mu ** 2) < 1e-10 else math.sqrt((sum_sq / n) - (sum_val / n) ** 2)
        sigma   = metrics_stats["PLR"]["sigma"] # σ (sigma) 
        updated_metrics_stats["PLR"]["sigma"]   = sigma
        updated_metrics_stats["PLR"]["mu"]      = mu
        
        #time.sleep(1)
        
        # Update Response statistics
        metrics_stats["Response"]["n"] += 1
        metrics_stats["Response"]["sum"] += new_plr
        metrics_stats["Response"]["sum_sq"] += new_plr ** 2

        # Recompute mu and sigma
        n = metrics_stats["Response"]["n"]
        sum_val = metrics_stats["Response"]["sum"]
        sum_sq  = metrics_stats["Response"]["sum_sq"]

        metrics_stats["Response"]["mu"] = sum_val / n 
        mu =  metrics_stats["Response"]["mu"] # µ (mu)
        #metrics_stats["PLR"]["sigma"] = math.sqrt((sum_sq / n) - (sum_val / n) ** 2)
        metrics_stats["Response"]["sigma"] = 0 if abs((sum_sq / n) - mu ** 2) < 1e-10 else math.sqrt((sum_sq / n) - (sum_val / n) ** 2)
        sigma   = metrics_stats["Response"]["sigma"] # σ (sigma) 
        #print("sum_val: ", sum_val, "\n sum_sq: ", sum_sq, "\n n: ", n, "\n sigma", sigma)
        updated_metrics_stats["Response"]["sigma"]   = sigma
        updated_metrics_stats["Response"]["mu"]      = mu
        #time.sleep(1)
        
        # Update CPU statistics
        metrics_stats["CPU"]["n"] += 1
        metrics_stats["CPU"]["sum"] += new_plr
        metrics_stats["CPU"]["sum_sq"] += new_plr ** 2

        # Recompute mu and sigma
        n = metrics_stats["CPU"]["n"]
        sum_val = metrics_stats["CPU"]["sum"]
        sum_sq  = metrics_stats["CPU"]["sum_sq"]

        metrics_stats["CPU"]["mu"] = sum_val / n 
        mu =  metrics_stats["CPU"]["mu"] # µ (mu)
        #metrics_stats["PLR"]["sigma"] = math.sqrt((sum_sq / n) - (sum_val / n) ** 2)
        metrics_stats["CPU"]["sigma"] = 0 if abs((sum_sq / n) - mu ** 2) < 1e-10 else math.sqrt((sum_sq / n) - (sum_val / n) ** 2)
        sigma   = metrics_stats["CPU"]["sigma"] # σ (sigma) 
        #print("sum_val: ", sum_val, "\n sum_sq: ", sum_sq, "\n n: ", n, "\n sigma", sigma)
        updated_metrics_stats["CPU"]["sigma"]   = sigma
        updated_metrics_stats["CPU"]["mu"]      = mu
        #time.sleep(1)
        
        
        return updated_metrics_stats
                
    def muAndsigma_per_node(self, node_id: int, new_plr, new_response, new_cpu) -> Dict:
        if node_id not in self.metrics_stats:
            # Initialise if node_id is new
            self.metrics_stats[node_id] = {
                "PLR": {"mu": 5.0, "sigma": 2.5, "n": 0, "sum": 0.0, "sum_sq": 0.0},
                "Response": {"mu": 100.0, "sigma": 50.0, "n": 0, "sum": 0.0, "sum_sq": 0.0},
                "CPU": {"mu": 50.0, "sigma": 20.0, "n": 0, "sum": 0.0, "sum_sq": 0.0}
            }

        node_stats = self.metrics_stats[node_id]
        inputs = {
            "PLR": new_plr,
            "Response": new_response,
            "CPU": new_cpu
        }

        updated_metrics = {}

        for metric, value in inputs.items():
            stats = node_stats[metric]
            stats["n"] += 1
            stats["sum"] += value
            stats["sum_sq"] += value ** 2

            n = stats["n"]
            mean = stats["sum"] / n
            mean_sq = stats["sum_sq"] / n
            variance = max(0.0, mean_sq - mean ** 2)  # clamp to avoid small negative due to FP error
            sigma = math.sqrt(variance)

            stats["mu"] = mean
            stats["sigma"] = sigma

            updated_metrics[metric] = {"mu": mean, "sigma": sigma}

        return updated_metrics

    def healthMetric(self, plr, response, cpu,
                 plr_mean, plr_std, response_mean, response_std, 
                 cpu_mean, cpu_std) -> float:
        """
        Computes the health metric h_i(t) for a node at time t.
        
        Args:
            plr, response, cpu.
            *_mean, *_std: Historical mean and standard deviation for each metric.
        
        Returns:
            float: Health score h_i(t).
        """
        WEIGHTS = self.WEIGHTS # {"PLR": 0.3, "Response": 0.2, "CPU": 0.3 }
        
        # Standardise each metric (Z-score)
        z_plr       = (plr - plr_mean) / plr_std if plr_std != 0 else 0.0
        z_response  = (response - response_mean) / response_std if response_std != 0 else 0.0
        z_cpu       = (cpu - cpu_mean) / cpu_std if cpu_std != 0 else 0.0
        
        #print("z_plr: ", z_plr, "\n z_response: ", z_response, "\n z_cpu", z_cpu)
        
        # Apply weights and sum
        h = (WEIGHTS["PLR"] * z_plr + 
            WEIGHTS["Response"] * z_response + 
            WEIGHTS["CPU"] * z_cpu )
        
        return h

    def update_threshold_with_bounds(self, node_id: int, current_health: float) -> float:
        """Enhanced threshold update with bounds and stability checks."""
        if node_id not in self.current_thresholds:
            # Initialise with first health value
            self.current_thresholds[node_id] = current_health
            # Set initial bounds (±3 standard deviations from initial value)
            self.threshold_bounds[node_id] = {
                "min": current_health - 3.0,
                "max": current_health + 3.0
            } 
        else:
            # EMA update
            new_threshold = (self.ALPHA * current_health + 
                           (1 - self.ALPHA) * self.current_thresholds[node_id])
            
            # Apply bounds to prevent threshold drift
            bounds = self.threshold_bounds[node_id]
            new_threshold = max(bounds["min"], min(bounds["max"], new_threshold))
            
            self.current_thresholds[node_id] = new_threshold
        return self.current_thresholds[node_id]
    
    def update_metrics_per_node_old(self, node_metrics, update_node_metric):
        """Process node metrics and calculate health metrics"""
        print(f"Processing {type(node_metrics).__name__} with {len(node_metrics)} nodes")
        
        for node, metric in node_metrics.items():
            new_plr = new_response = new_cpu = None
            
            nh = NetworkHierarchy()
            if not (node.startswith("L4") or node.startswith("Sen_")):
                descendants = nh.get_all_descendants(node)
                print("node: ", node, "\ndescendants: ", descendants); time.sleep(10)
            
            if node == "CloudDBServer":
                node_details = {}
                #node_details = {"node_id": counter+1}
                cloud_node_metric = self._generate_cloud_metrics()
                for metric in ["PLR", "Response", "CPU",  "status"]: 
                    node_details[metric] = cloud_node_metric[metric]       
                #print("node_details: ", node_details)
                
                nodeStatus = self.healthStatusold(node_details)
                #print("node_details: ", node_details); sleep(300)
                node_details["healthyNode"] = nodeStatus
                
                node_metrics[node] = node_details
                update_node_metric[node] = node_details
                #counter+=1
            if node == "L1Node":
                node_details = {}
                #node_details = {"node_id": counter}
                l1_node_metric = self._generate_l1_metrics()
                for metric in ["PLR", "Response", "CPU",  "status"]:
                    node_details[metric] = l1_node_metric[metric]
                
                nodeStatus = self.healthStatusold(node_details)
                node_details["healthyNode"] = nodeStatus
                
                node_metrics[node] = node_details
                update_node_metric[node] = node_details
            
            if (node != "CloudDBServer" and node != "L1Node" and not node.startswith('Sen_')):   
                node_details = {}
                for key in metric.keys(): 
                    new_plr = metric["PLR"]; new_response = metric["Response"]; new_cpu = metric["CPU"];  node_id = metric["node_id"]
                # Calculate health metrics
                meanAndsd = self.muAndsigma_per_node(node_id, new_plr, new_response, new_cpu)
                healthmetric = self.healthMetric(
                    new_plr, new_response, new_cpu, 
                    meanAndsd["PLR"]["mu"], meanAndsd["PLR"]["sigma"],  
                    meanAndsd["Response"]["mu"], meanAndsd["Response"]["sigma"], 
                    meanAndsd["CPU"]["mu"], meanAndsd["CPU"]["sigma"]
                )
                
                node_id = metric["node_id"]
                threshold = self.update_threshold_with_bounds(node_id, healthmetric)
                                
                # Update metric with health data
                metric["health_metrics"] = healthmetric
                metric["threshold"] = threshold
                metric["last_updated"] = datetime.now().isoformat()
                
                health_diff = healthmetric - threshold
                
                nodeStatus = self.healthStatusold(metric)
                metric["healthyNode"] = nodeStatus
                
                update_node_metric[node] = metric
                
                time.sleep(3)  # Brief pause between nodes

        with open('update_node_metric.json', 'w') as fp:
            json.dump(update_node_metric, fp, indent=4)
        return node_metrics

    def update_metrics_per_node(self, processed_metrics: dict, failure_count: int, max_failures: int) -> Tuple[dict, int]:
        """
        Update metrics dynamically for all nodes in the network using standard running estimation methods.

        Args:
            processed_metrics - dict: Dictionary with current metrics per node.
            failure_count - int: Current failure injection count.
            max_failures - int: Maximum allowable fault injections.

        Returns:
            Tuple: (updated processed_metrics, updated failure_count)
        """
        #print("processed_metrics: ", processed_metrics); sleep(2)
        #print("self.G: ", self.G); sleep(300)
        
        # Step 1: Rebuild fog_nodes with child relationships
        #hierarchy = NetworkHierarchy()
        #G, _, _ = hierarchy.create_network()

        # Construct minimal fog_nodes with .children attribute
        fog_nodes = {}
        for node_name, details in processed_metrics.items():
            node_id = details.get("node_id")
            children = set(self.G.successors(node_name))
            fog_nodes[node_id] = type("TempFogNode", (), {"node_id": node_id, "name": node_name, "children": children})()
        print("fog_nodes: ", fog_nodes); sleep(2)
        
        '''for fog_node_id, fog_node_obj in fog_nodes.items():
            print("fog_node_id: ", fog_node_id)
            print("fog_node name: ", fog_node_obj.name)
            print("children: ", fog_node_obj.children)
            sleep(2)'''
        # Step 2: Update metrics per node based on role
        for node_name, details in processed_metrics.items():
            node_id = details["node_id"]

            if node_name == "CloudDBServer":
                descendants = self.get_all_current_descendants(node_name, self.G)
                workload = len([d for d in descendants if d in processed_metrics]); print("workload: ", workload)
                metrics = self._standard_running_cloud_metrics(total_child_workload=workload); 
                metrica={"node_id": node_id, "node": node_name}
                metrics = {**metrica, **metrics}
                print("metrics: ", metrics)
                #sleep(300)
            elif node_name == "L1Node":
                descendants = self.get_all_current_descendants(node_name, self.G)
                workload = len([d for d in descendants if d in processed_metrics]); print("workload: ", workload)
                metrics = self._standard_running_l1_metrics(total_child_workload=workload); 
                metrica={"node_id": node_id, "node": node_name}
                metrics = {**metrica, **metrics}
                print("metrics: ", metrics)
                #sleep(300)

            elif  (node_name != "L1Node" or node_name != "CloudDBServer") and not node_name.startswith("Sen_"):
                if node_name.startswith("L2N"):
                    layer = 2
                elif node_name.startswith("L3N"):
                    layer = 3
                elif node_name.startswith("L4N"):
                    layer = 4
                else:
                    layer = 1  # fallback

                num_children = len(fog_nodes[node_id].children)
                metrics = self._standard_running_metrics(layer, num_children, failure_count, max_failures)
                #sleep(300)
                failure_count = metrics.pop("failure_count", failure_count)
                metrica={"node_id": node_id, "node": node_name}
                metrics = {**metrica, **metrics}
                print("metrics: ", metrics)
            else:
                continue  # skip sensors

            # Step 3: Update stats, compute health
            updated_stats = self.muAndsigma_per_node(node_id, metrics["PLR"], metrics["Response"], metrics["CPU"])
            health = self.healthMetric(
                metrics["PLR"], metrics["Response"], metrics["CPU"], 
                updated_stats["PLR"]["mu"], updated_stats["PLR"]["sigma"],
                updated_stats["Response"]["mu"], updated_stats["Response"]["sigma"],
                updated_stats["CPU"]["mu"], updated_stats["CPU"]["sigma"]
            )
            threshold = self.update_threshold_with_bounds(node_id, health)

            print("health: ", health, "threshold: ",  threshold, "health - threshold ", health - threshold  ); sleep(3)
            # Step 4: Compose and assign updated values
            metrics.update({
                "node_id": node_id,
                "health_metrics": round(health, 4),
                "threshold": round(threshold, 4),
                "last_updated": datetime.now().isoformat(),
                "health_status": "healthy" if health >= threshold else "faulty",
                "healthyNode": health >= threshold
            })

            processed_metrics[node_name] = metrics
            
        with open('update_node_metric.json', 'w') as fp:
            json.dump(processed_metrics, fp, indent=4)
        return processed_metrics
    
        print("✅ All node metrics updated successfully.")
        return processed_metrics, failure_count
    
    def get_all_current_descendants(self, node_name: str, G) -> List[str]:
        """Return all descendants (children, grandchildren, etc.) of a given node."""
    
        descendants = []
        try:
            self.hierarchy = {node: list(G.successors(node)) for node in G.nodes}
        except: 
            pass   
        def recurse(current_node):
            children = []
            if not current_node.startswith("L4"):
                children = self.hierarchy.get(current_node, [])
                for child in children:
                    if not child.startswith("Sen_"):
                        descendants.append(child)
                    recurse(child)
        recurse(node_name)
        return descendants
    
    def healthStatus(self, node_details, healthmetric, adaptive_threshold) -> bool:
        """
        Evaluates the health status of a node based on various metrics.
        
        Args:
            node_details  - dict: Dictionary containing node metrics with keys:
                            ["PLR", "Response", "CPU" ]
        
        Returns:
            bool: True if node is faulty, False if healthy
        """
        
        STATIC_THRESHOLDS = self.STATIC_THRESHOLDS
        
        #["PLR", "Response", "CPU",  "status"]
        plr         = node_details["PLR"]
        response    = node_details["Response"]
        cpu         = node_details["CPU"]
        
        # = {"PLR": 10, "Response": 200, "CPU": 80}
        PLR_STATIC_THRESHOLDS       = STATIC_THRESHOLDS["PLR"]
        Response_STATIC_THRESHOLDS  = STATIC_THRESHOLDS["Response"]
        CPU_STATIC_THRESHOLDS       = STATIC_THRESHOLDS["CPU"]
        
        #status      = node_details["status"]
        #status      = "FAULTY"
                
        # Check if any metric violates its threshold
        is_faulty = (
            plr > PLR_STATIC_THRESHOLDS or
            response > Response_STATIC_THRESHOLDS or
            cpu > CPU_STATIC_THRESHOLDS
        )
        if not is_faulty:
            if 1==1:
            #try:
                if (healthmetric - adaptive_threshold) < 0:
                    node_details["health_status"] = "faulty"                    
                    is_faulty = True
                else:
                    node_details["health_status"] = "healthy"
                    is_faulty = False
            else:
            #except :
                node_details["health_status"] = "faulty"
                is_faulty = True
        
        return not is_faulty

    def healthStatusold(self, node_details, healthmetric, threshold) -> bool:
        """
        Evaluates the health status of a node based on various metrics.
        
        Args:
            node_details - dict: Dictionary containing node metrics with keys:
                            ["PLR", "Response", "CPU" ]
        
        Returns:
            bool: True if node is faulty, False if healthy
        """
        
        STATIC_THRESHOLDS = self.STATIC_THRESHOLDS
        
        #["PLR", "Response", "CPU",  "status"]
        plr         = node_details["PLR"]
        response    = node_details["Response"]
        cpu         = node_details["CPU"]
        
        # = {"PLR": 10, "Response": 200, "CPU": 80}
        PLR_STATIC_THRESHOLDS       = STATIC_THRESHOLDS["PLR"]
        Response_STATIC_THRESHOLDS  = STATIC_THRESHOLDS["Response"]
        CPU_STATIC_THRESHOLDS       = STATIC_THRESHOLDS["CPU"]
        
        #status      = node_details["status"]
        status      = "FAULTY"
                
        # Check if any metric violates its threshold
        is_faulty = (
            plr > PLR_STATIC_THRESHOLDS or
            response > Response_STATIC_THRESHOLDS or
            cpu > CPU_STATIC_THRESHOLDS 
        )
        if not is_faulty:
            try:
                if healthmetric - threshold < 0:
                    node_details["health_status"] = "faulty"                    
                    is_faulty = True
                else:
                    node_details["health_status"] = "healthy"
                    
                    is_faulty = False
            except :
                node_details["health_status"] = "faulty"

                is_faulty = False
        
        return not is_faulty

def save_results(node_metrics, output_file):
    """Save processed metrics to JSON file"""
    try:
        with open(output_file, 'w') as fp:
            json.dump(node_metrics, fp, indent=4)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

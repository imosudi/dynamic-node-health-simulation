import time
import numpy as np
from typing import List, Optional, Dict, Any, Tuple,    Union 

class MetricGenerator_old:
    def __init__(self, metric_names: List[str], baselines: List[float], 
                 noise_levels: List[float], seed: Optional[int], 
                 signs: Optional[List[float]] = None):
        self.metric_names = metric_names
        self.baselines = np.array(baselines)
        self.noise_levels = np.array(noise_levels)
        self.signs = np.array(signs) if signs is not None else np.ones_like(self.baselines)
        self.rng = np.random.default_rng(seed)
    
    def step(self) -> np.ndarray:
        #noise_scales = [0.05, 5, 0.002]
        #print("self.noise_levels: ", type(self.noise_levels) , self.noise_levels); time.sleep(1000)
        #self.noise_levels:  {'cpu': 0.05, 'rtt': 5, 'plr': 0.002}
        #noise_levels = self.noise_levels
        #noise_levels = list(noise_levels.values())
        #print("noise_levels: ", noise_levels); time.sleep(10000)
        noise = self.rng.normal(0, self.noise_levels)
        #print("self.baselines: ", self.baselines, " noise: ", noise); time.sleep(1)
        new_level = self.baselines + noise
        #print("self.baselines: ", self.baselines, " new_level: ", new_level); time.sleep(1)
        return new_level  


class MetricGenerator_modified:
    def __init__(self, metric_names: List[str], baselines: List[float], 
                 noise_levels: List[float], seed: Optional[int], 
                 signs: Optional[List[float]] = None):
        self.metric_names = metric_names
        self.baselines = np.array(baselines)
        self.noise_levels = np.array(noise_levels)
        self.signs = np.array(signs) if signs is not None else np.ones_like(self.baselines)
        self.rng = np.random.default_rng(seed)

    def _get_layer(self, node: str) -> str:
        """Infer the layer type from the node name."""
        if node == "CloudDBServer":
            return "CLOUD"
        elif node == "L1Node":
            return "L1"
        elif node.startswith("L2N"):
            return "L2"
        elif node.startswith("L3N"):
            return "L3"
        elif node.startswith("L4N"):
            return "L4"
        elif node.startswith("Sen_"):
            return "SENSOR"
        else:
            raise ValueError(f"Unknown node layer for {node}")

    
    def step(self) -> np.ndarray:
        noise = self.rng.normal(0, self.noise_levels)
        #print("self.baselines: ", self.baselines, " noise: ", noise); time.sleep(1)
        new_level = self.baselines + noise
        #print("self.baselines: ", self.baselines, " new_level: ", new_level); time.sleep(1)
        return new_level  


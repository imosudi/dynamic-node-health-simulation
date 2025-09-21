
from datetime import datetime, timedelta, timezone
import random


class fogNodeCharacterisation(object):
    """_summary_

    Args:
        object (_type_): Charactersation of network-wides fog nodes
    """
    def __init__(self, tendency_data, WEIGHTS, STATIC_THRESHOLDS, ALPHA=0.2, *args):
        self.G                  = {}
        self.WEIGHTS            = WEIGHTS #{"PLR": 0.3, "RTT": 0.2, "CPU": 0.3}
        self.STATIC_THRESHOLDS  = STATIC_THRESHOLDS #{"PLR": 10, "RTT": 200, "CPU": 80}
        if not (0 < ALPHA < 1):
            raise ValueError("alpha must be between 0 and 1.")
        self.ALPHA              = ALPHA  # Smoothing factor for EMA
        self.current_thresholds = {} # Current adaptive thresholds per node
        self.M2                 = 0.0  # Sum of squared differences
        
        self.metrics_stats      = {
                                    "PLR": {"mu": 5.0, "sigma": 2.5, "n": 0, "sum": 0, "sum_sq": 0},
                                    "RTT": {"mu": 100.0, "sigma": 50.0, "n": 0, "sum": 0, "sum_sq": 0},
                                    "CPU": {"mu": 50.0, "sigma": 20.0, "n": 0, "sum": 0, "sum_sq": 0}
                                }
        # mu        Historical mean (μ) of the metric. Initial guess or prior. 
        #           PLR=5.0 (5% packet loss), RTT=100, CPU, 50.0, Accuracy=90.0
        # sigma     Historical standard deviation (σ). Measures variability. PLR=2.5 (moderate variance)
        # n         Number of observations seen so far. Starts at 0 (no data). PLR,RTT,CPU,Accuracy=0
        # sum       Running sum of all observed values. Used to compute mu.  PLR,RTT,CPU,Accuracy=0
        # sum_sq    Running sum of squared values. Used to compute sigma.  PLR,RTT,CPU,Accuracy=0
        self.threshold_bounds = {}  # Min/max bounds for thresholds

        # Initialise dataset storage
        self.dataset = {
            'timestamp': [],
            'node_id': [],
            'plr': [],
            'cpu': [],
            'rtt': [],
            'weighted_health_score': [],
            'health_threshold': [],
            'health_difference': [],
            'health_status': [],
            'anomaly_type': []
        }

        super(fogNodeCharacterisation, self).__init__(*args)
    
    def _generate_cloud_metrics(self) -> dict:
        """Generate optimised cloud server metrics."""
        #  {'cpu': 0.2, 'rtt': 10, 'plr': 0.01}
        return {
            "node": "CloudDBServer",
            "PLR": round(random.uniform(0, 2), 2), #expresses packet loss in percentage: x%
            "RTT": round(random.uniform(5, 10), 2), #in milliseconds
            "CPU": round(random.uniform(10, 40), 2), #in percentage of CPU usage: x%
            "status": "HEALTHY"
        }

    def _generate_l1_metrics(self) -> dict:
        """Generate L1 node metrics with realistic variations."""
        return {
            "node": "L1Node",
            "PLR": round(random.uniform(2, 6), 2),
            "RTT": round(random.uniform(20, 45), 2),
            "CPU": round(random.uniform(30, 60), 2),
            "status": "HEALTHY"
        }

    def healthMetric(self, node_id, plr, rtt, cpu,
                 plr_mean, plr_std, rtt_mean, rtt_std, 
                 cpu_mean, cpu_std) -> float:
        """
        Computes the health metric h_i(t) for a node at time t.
        
        Args:
            plr, rtt, cpu, : Current observed values.
            *_mean, *_std: Historical mean and standard deviation for each metric.
        
        Returns:
            float: Health score h_i(t).
        """
        WEIGHTS = self.WEIGHTS #{"PLR": 0.4, "rtt": 0.3, "CPU": 0.3} #self.WEIGHTS 

        # Standardise each metric (Z-score)
        z_plr       = (plr - plr_mean) / plr_std if plr_std != 0 else 0.0
        z_rtt       = (rtt - rtt_mean) / rtt_std if rtt_std != 0 else 0.0
        z_cpu       = (cpu - cpu_mean) / cpu_std if cpu_std != 0 else 0.0
        
        
        #print("z_plr: ", z_plr, "\n z_rtt: ", z_rtt, "\n z_cpu", z_cpu, "\n z_: ", z_)
        
        # Apply weights and sum
        h = (WEIGHTS["PLR"] * z_plr + 
            WEIGHTS["rtt"] * z_rtt + 
            WEIGHTS["CPU"] * z_cpu )
        health_threshold = self.update_threshold_with_bounds(node_id, h)
        health_difference = h - health_threshold
        health_status = "HEALTHY" if h <= health_threshold else "FAULTY"
        anomaly_type = "NONE"
        current_time = datetime.now(timezone.utc)
        # Store metrics in dataset
        self.dataset['timestamp'].append(current_time)
        self.dataset['node_id'].append(node_id)
        self.dataset['plr'].append(plr)
        self.dataset['cpu'].append(cpu)
        self.dataset['rtt'].append(rtt)
        self.dataset['weighted_health_score'].append(weighted_health_score)
        self.dataset['health_threshold'].append(health_threshold)
        self.dataset['health_difference'].append(health_difference)
        self.dataset['health_status'].append(health_status)
        self.dataset['anomaly_type'].append(anomaly_type)

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

class healthMetricCalculator:
    def __init__(self, node_id, tendency_data, WEIGHTS, STATIC_THRESHOLDS, ALPHA=0.2, *args):
        self.tendency_data      = tendency_data
        self.WEIGHTS            = WEIGHTS 
        self.node_id            = node_id
        self.STATIC_THRESHOLDS  = STATIC_THRESHOLDS 
        if not (0 < ALPHA < 1):
            raise ValueError("alpha must be between 0 and 1.")
        self.ALPHA              = ALPHA  # Smoothing factor for EMA
        self.threshold_bounds = {}  # Min/max bounds for thresholds
        self.current_thresholds = {} # Current adaptive thresholds per node
        
        self.metrics_stats      = {
                                    "PLR": {"mu": 5.0, "sigma": 2.5, "n": 0, "sum": 0, "sum_sq": 0},
                                    "RTT": {"mu": 100.0, "sigma": 50.0, "n": 0, "sum": 0, "sum_sq": 0},
                                    "CPU": {"mu": 50.0, "sigma": 20.0, "n": 0, "sum": 0, "sum_sq": 0}
                                }
        # Initialise dataset storage
        self.dataset = {
            'timestamp': [],
            'node_id': [],
            'plr': [],
            'cpu': [],
            'rtt': [],
            'weighted_health_score': [],
            'health_threshold': [],
            'health_difference': [],
            'health_status': [],
            'anomaly_type': []
        }

        # mu        Historical mean (μ) of the metric. Initial guess or prior. 
        #           PLR=5.0 (5% packet loss), RTT=100, CPU, 50.0, Accuracy=90.0
        # sigma     Historical standard deviation (σ). Measures variability. PLR=2.5 (moderate variance)
        # n         Number of observations seen so far. Starts at 0 (no data). PLR,RTT,CPU,Accuracy=0
        # sum       Running sum of all observed values. Used to compute mu.  PLR,RTT,CPU,Accuracy=0
        # sum
    
    def vetting_returned_result(self):
        tendency_data = self.tendency_data
        if not tendency_data:
            print("No data returned from simulation.")
            return False
        if not isinstance(tendency_data, dict):
            print("Data returned is not a list.")
            return False
        if len(tendency_data) == 0:
            print("Data returned list is empty.")
            return False
        
        return tendency_data

    
    def health_metric_data(self):
        tendency_data = self.vetting_returned_result()

        #print(f"\n tendency_data: ", tendency_data ); #time.sleep(200)
        node_data = tendency_data.get("node_data", {})
        plr = node_data.get("plr", 0.0)
        rtt = node_data.get("rtt", 0.0)
        cpu = node_data.get("cpu", 0.0)
        step = node_data.get("step", -1)

        cpu_tendency = tendency_data.get("cpu", "N/A")
        plr_tendency = tendency_data.get("plr", "N/A")
        rtt_tendency = tendency_data.get("rtt", "N/A")

        return plr, rtt, cpu, step, cpu_tendency, plr_tendency, rtt_tendency
    
    def healthMetric( self) -> float:
        """
        Computes the health metric h_i(t) for a node at time t.
        
        Args:
            plr, rtt, cpu, : Current observed values.
            *_mean, *_std: Historical mean and standard deviation for each metric.
        
        Returns:
            float: Health score h_i(t).
        """
        plr, rtt, cpu, step, cpu_tendency, plr_tendency, rtt_tendency = self.health_metric_data()
        WEIGHTS = self.WEIGHTS 
        node_id = self.node_id

        # Extract means and stddevs from tendency data
        plr_mean = plr_tendency["mean"]; plr_std = plr_tendency["std"]; rtt_mean = rtt_tendency["mean"]; rtt_std = rtt_tendency["std"]; 
        cpu_mean =  cpu_tendency["mean"]; cpu_std = cpu_tendency["std"]

        sum_plr = plr_mean * step if step else 0.0
        sum_rtt = rtt_mean * step if step else 0.0
        sum_cpu = cpu_mean * step if step else 0.0

        self.metrics_stats      = {
                                    "PLR": {"mu": plr_mean, "sigma": plr_std, "n": step, "sum": sum_plr, "sum_sq": sum_plr ** 2},
                                    "RTT": {"mu": rtt_mean, "sigma": rtt_std, "n": step, "sum": sum_rtt, "sum_sq": sum_rtt ** 2},
                                    "CPU": {"mu": cpu_mean, "sigma": cpu_std, "n": step, "sum": sum_cpu, "sum_sq": sum_cpu ** 2}
                                }
        # Standardise each metric (Z-score)
        z_plr       = (plr - plr_mean) / plr_std if plr_std != 0 else 0.0
        z_rtt       = (rtt - rtt_mean) / rtt_std if rtt_std != 0 else 0.0
        z_cpu       = (cpu - cpu_mean) / cpu_std if cpu_std != 0 else 0.0
        
        
        #print("z_plr: ", z_plr, "\n z_rtt: ", z_rtt, "\n z_cpu", z_cpu, "\n z_: ", z_)
        
        # Apply weights and sum
        h = (WEIGHTS["PLR"] * z_plr + 
            WEIGHTS["RTT"] * z_rtt + 
            WEIGHTS["CPU"] * z_cpu )
        
        
        health_threshold    = self.update_threshold_with_bounds(node_id, h)
        health_difference   = h - health_threshold
        health_status       = "HEALTHY" if h <= health_threshold else "FAULTY"
        anomaly_type        = "NONE"
        current_time        = datetime.now(timezone.utc)
        # Store metrics in dataset
        self.dataset['timestamp'].append(current_time)
        self.dataset['node_id'].append(node_id)
        self.dataset['plr'].append(plr)
        self.dataset['cpu'].append(cpu)
        self.dataset['rtt'].append(rtt)
        self.dataset['health_threshold'].append(health_threshold)
        self.dataset['health_difference'].append(health_difference)
        self.dataset['health_status'].append(health_status)
        self.dataset['anomaly_type'].append(anomaly_type)
        self.dataset['weighted_health_score'].append(h)

        return h, self.dataset
    
    #def update_threshold_with_bounds(self, node_id: int, current_health: float) -> float:
    def update_threshold_with_bounds(self, node_id: str, current_health: float) -> float:
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
        

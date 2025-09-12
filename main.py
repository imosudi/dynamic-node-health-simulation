#main.py
from collections import defaultdict
from fault_injection.fault_injection_sim import run_complete_simulation
from logging_mod.csv_writer import write_detailed_csv
import numpy as np


#baseline_values = {'cpu': 0.2, 'rtt': 100, 'plr': 0.01}
#baseline_values = {'cpu': 0.2, 'rtt': 30, 'plr': 0.01}
#baseline_values = {"cpu":0.145, "rtt":26.4, "plr":0.0084}
#baseline_values = {"cpu":0.090, "rtt":22.8, "plr":0.0068}
#max_values = {'cpu': 1.0, 'rtt': 500, 'plr': 0.5}  # Maximum reasonable values

#baseline_values = {'cpu': 0.2, 'rtt': 100, 'plr': 0.01}
#baseline_values = {'cpu': 0.2, 'rtt': 30, 'plr': 0.01}
#baseline_values = {"cpu":0.145, "rtt":26.4, "plr":0.0084}
default_weights = {'PLR': 0.4, 'CPU': 0.3, 'RTT': 0.3}
baseline_values = {"cpu":0.090, "rtt":22.8, "plr":0.0068}
max_values = {'cpu': 0.75, 'rtt': 150, 'plr': 0.5}

fault_templates = 'fault_injection/fault_templates_zero.yaml',
#fault_templates = 'fault_injection/fault_templates.yaml',
#fault_templates = 'fault_injection/templates.yaml'

class VettingReturnedResult:
    def __init__(self, node_ids, max_display=16, ema_beta=0.3):
        self.node_ids = node_ids
        self.beta = ema_beta
        self.maxlen = 100

        # Initialise metric histories
        self.health_histories = defaultdict(list)
        self.cpu_histories = defaultdict(list)
        self.plr_histories = defaultdict(list)
        self.rtt_histories = defaultdict(list)
        self.ema_health = {nid: None for nid in self.node_ids}

    def vetting_returned_result(self, results):
        if not results:
            print("No data returned from simulation.")
            return False
        if not isinstance(results, list):
            print("Data returned is not a list.")
            return False
        if len(results) == 0:
            print("Data returned list is empty.")
            return False
        
        # Testing return values

        #print(f"\nresults last item: ", results[-1:] )
        if isinstance(results[-1:][0], dict) and 'cpu' in results[-1:][0]:
            result = results[-1:][0]
            #print(result['cpu'], result['rtt'], result['plr']  )
            result_metrics =  {
                    k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                    for k, v in result.items()
                }
        print(result)
        #self.beta = ema_beta
        #self.beta = ema_beta
        cpu_score = 1 - (result_metrics["cpu"] / 100)
        plr_score = 1 - min(result_metrics['plr'] / 0.2, 1.0)
        rtt_score = 1 - min(result_metrics['rtt'] / 400, 1.0) # Assuming 400ms as a threshold for RTT, I am concerned with my assumption here.

        # Weighted health score
        weighted_health_score = 0.3 * cpu_score + 0.3 * plr_score + 0.4 * rtt_score

        print("weighted_health_score: ", weighted_health_score)

        for i, node_id in enumerate(self.node_ids):
            

            # Update EMA health threshold
            if self.ema_health[node_id] is None:
                self.ema_health[node_id] = weighted_health_score
            else:
                self.ema_health[node_id] = (
                    self.beta * weighted_health_score + (1 - self.beta) * self.ema_health[node_id]
                )
        print("EMA Health Scores:", self.ema_health)
        return True

    def healthMetric(self, plr, rtt, cpu,
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
        WEIGHTS = self.WEIGHTS # {"PLR": 0.3, "rtt": 0.2, "CPU": 0.3}
        
        # Standardise each metric (Z-score)
        z_plr       = (plr - plr_mean) / plr_std if plr_std != 0 else 0.0
        z_rtt  = (rtt - rtt_mean) / rtt_std if rtt_std != 0 else 0.0
        z_cpu       = (cpu - cpu_mean) / cpu_std if cpu_std != 0 else 0.0
        
        
        #print("z_plr: ", z_plr, "\n z_rtt: ", z_rtt, "\n z_cpu", z_cpu, "\n z_: ", z_)
        
        # Apply weights and sum
        h = (WEIGHTS["PLR"] * z_plr + 
            WEIGHTS["rtt"] * z_rtt + 
            WEIGHTS["CPU"] * z_cpu )
        
        return h


try:
        # Run simulation
        results, injector, data_returned, history = run_complete_simulation(default_weights, baseline_values, max_values, steps=110, seed=10, fault_templates='fault_injection/templates.yaml')
        print(f"\nSimulation completed successfully!")
        print(f"Total steps: {len(results)}")
        print(f"Fault periods: {sum(1 for r in results if r['any_fault_active'])}")
        print("data_returned: ", data_returned, type(data_returned))
        #data_returned = [data_returned]
        #VettingReturnedData(["L1N_01"]).vetting_returned_data(data_returned) #, "L2N_01", "L3N_01", "L4N_01"]).vetting_returned_data(data_returned)
        
        #print(f"results: {results}")
        VettingReturnedResult(["L1N_01"]).vetting_returned_result(results) 
        #print(f"injector: {injector}")
        #print(f"history: {history}")
        #write_detailed_csv(history, filename='health_monitoring_detailed.csv')
except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

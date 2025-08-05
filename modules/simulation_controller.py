import json
import csv

class SimulationController:
    def __init__(self, node_profiles, fault_injector, health_scorer, adaptive_threshold, health_classifier):
        self.node_profiles = node_profiles
        self.fault_injector = fault_injector
        self.health_scorer = health_scorer
        self.adaptive_threshold = adaptive_threshold
        self.health_classifier = health_classifier
        self.simulation_log = []

    def run_simulation(self, total_timesteps):
        self.adaptive_threshold.initialise_thresholds(self.node_profiles)
        
        for node in self.node_profiles:
            self.fault_injector.generate_fault_probability_scenario(node.node_id, total_timesteps)

        for t in range(total_timesteps):
            for node in self.node_profiles:
                observed_metrics = {}
                for metric in node.metric_means:
                    observed_metrics[metric] = self.fault_injector.inject_fault(node, metric, t)
                
                health_score = self.health_scorer.compute_health_score(node, observed_metrics)
                threshold = self.adaptive_threshold.update_threshold(node.node_id, health_score)
                status = self.health_classifier.classify_health(health_score, threshold)

                self.simulation_log.append({
                    'time_step': t,
                    'node_id': node.node_id,
                    'health_score': health_score,
                    'threshold': threshold,
                    'status': status,
                    'observed_metrics': observed_metrics
                })

    def export_logs(self, json_path, csv_path):
        # Export JSON
        with open(json_path, 'w') as f_json:
            json.dump(self.simulation_log, f_json, indent=4)

        # Export CSV
        with open(csv_path, 'w', newline='') as f_csv:
            fieldnames = ['time_step', 'node_id', 'health_score', 'threshold', 'status', 'observed_metrics']
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.simulation_log:
                writer.writerow(entry)

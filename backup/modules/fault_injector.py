import numpy as np

class FaultInjector:
    def __init__(self, severity_multipliers):
        self.severity_multipliers = severity_multipliers
        self.fault_probabilities = {}  # node_id -> [p(t)]

    def generate_fault_probability_scenario(self, node_id, total_timesteps):
        # Example: sinusoidal stress scenario
        t = np.arange(total_timesteps)
        p_t = 0.05 + 0.05 * np.sin(2 * np.pi * t / 50)  # Periodic stress spikes
        self.fault_probabilities[node_id] = p_t.tolist()

    def inject_fault(self, node_profile, metric, time_step):
        node_id = node_profile.node_id
        p_t = self.fault_probabilities.get(node_id, [0.0])[time_step]
        fault_occurred = np.random.rand() < p_t
        baseline_value = node_profile.metric_means[metric]
        if fault_occurred:
            perturbation = self.severity_multipliers[metric] * node_profile.metric_stds[metric]
            return baseline_value + perturbation
        else:
            return baseline_value

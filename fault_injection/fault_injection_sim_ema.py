import numpy as np
import matplotlib.pyplot as plt
import yaml

class FaultTemplate:
    def __init__(self, config):
        self.id = config.get('id')
        self.name = config.get('name')
        self.type = config.get('type')
        self.subtype = config.get('subtype')
        self.affected_metrics = config.get('affected_metrics', [])
        self.severity_vector = np.array(config.get('severity_vector', []), dtype=float) if 'severity_vector' in config else None
        self.covariance = config.get('covariance', None)
        self.occurrence_model = config.get('occurrence_model', None)
        self.duration_dist = config.get('duration_dist', None)
        self.stuck_value = config.get('stuck_value', None)
        self.ramp_slope = config.get('ramp_slope', None)
        self.max_deviation = config.get('max_deviation', None)
        self.repeatable = config.get('repeatable', True)
        self.note = config.get('note', '')

class FaultInjector:
    def __init__(self, templates, metric_names):
        self.templates = templates
        self.metric_names = metric_names
        self.active_faults = []

    def sample_duration(self, dist):
        if dist['type'] == 'geometric':
            return np.random.geometric(dist['p'])
        elif dist['type'] == 'deterministic':
            return dist['n_steps']
        return 1

    def maybe_trigger_fault(self, template):
        if template.occurrence_model and template.occurrence_model['type'] == 'bernoulli':
            p = template.occurrence_model['p']
            if np.random.rand() < p:
                duration = self.sample_duration(template.duration_dist)
                self.active_faults.append({
                    'template': template,
                    'remaining': duration,
                    'step': 0
                })

    def apply_faults(self, baseline):
        observed = baseline.copy()
        to_remove = []
        for f in self.active_faults:
            tpl = f['template']
            if tpl.type == 'additive':
                if tpl.severity_vector is not None:
                    if tpl.covariance:
                        rho = tpl.covariance['rho']
                        size = len(tpl.affected_metrics)
                        cov_matrix = np.full((size, size), rho)
                        np.fill_diagonal(cov_matrix, 1.0)
                        L = np.linalg.cholesky(cov_matrix)
                        z = np.random.randn(size)
                        correlated = L @ z
                        correlated = correlated / np.std(correlated) * tpl.severity_vector
                        for i, m in enumerate(tpl.affected_metrics):
                            observed[m] += correlated[i]
                    else:
                        for i, m in enumerate(tpl.affected_metrics):
                            observed[m] += tpl.severity_vector[i]

            elif tpl.type == 'persistent':
                if tpl.subtype == 'stuck_at':
                    for i, m in enumerate(tpl.affected_metrics):
                        observed[m] = tpl.stuck_value
                elif tpl.subtype == 'ramp':
                    for i, m in enumerate(tpl.affected_metrics):
                        deviation = min(f['step'] * tpl.ramp_slope, tpl.max_deviation)
                        observed[m] += deviation

            f['step'] += 1
            f['remaining'] -= 1
            if f['remaining'] <= 0:
                to_remove.append(f)

        for f in to_remove:
            self.active_faults.remove(f)
        return observed

    def step(self, baseline):
        for tpl in self.templates:
            self.maybe_trigger_fault(tpl)
        return self.apply_faults(baseline)

class HealthMonitor:
    def __init__(self, metric_names, signs, alpha=0.1, init_window=10, kappa=2.0):
        self.metric_names = metric_names
        self.signs = np.array(signs)
        self.alpha = alpha
        self.init_window = init_window
        self.kappa = kappa
        self.values = []
        self.theta = None
        self.sigma_theta = None

    def normalize(self, metrics, baselines, stds):
        return [(metrics[i] - baselines[self.metric_names[i]]) / stds[i]
                for i in range(len(self.metric_names))]

    def update(self, metrics, baselines, stds):
        normed = self.normalize(metrics, baselines, stds)
        h = np.mean(self.signs * np.array(normed))
        self.values.append(h)

        if len(self.values) <= self.init_window:
            self.theta = np.mean(self.values)
            self.sigma_theta = np.std(self.values) if len(self.values) > 1 else 0.01
            return h, self.theta, "INIT"

        self.theta = self.alpha * h + (1 - self.alpha) * self.theta
        residual = abs(h - self.theta)
        self.sigma_theta = 0.9 * self.sigma_theta + 0.1 * residual

        if h < self.theta - 1.0:
            status = "Faulty"
        elif h >= 1.0:
            status = "Good"
        elif h >= 0:
            status = "Fair"
        else:
            status = "Poor"

        return h, self.theta, status

class DeterministicGenerator:
    def __init__(self, metric_names, base_values, noise_scales):
        self.metric_names = metric_names
        self.base_values = base_values
        self.noise_scales = noise_scales

    def generate(self):
        return np.array([self.base_values[m] + np.random.normal(0, s)
                         for m, s in zip(self.metric_names, self.noise_scales)])

def load_fault_templates(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return [FaultTemplate(t) for t in config['fault_templates']]

def run_complete_simulation(yaml_path, steps=1000):
    metric_names = ["RTT", "CPU", "PLR"]
    base_values = {"RTT": 50, "CPU": 20, "PLR": 0.01}
    noise_scales = [5, 3, 0.005]

    templates = load_fault_templates(yaml_path)
    generator = DeterministicGenerator(metric_names, base_values, noise_scales)
    injector = FaultInjector(templates, metric_names)
    health_monitor = HealthMonitor(metric_names, signs=[-1, -1, -1], alpha=0.1)

    history = []
    for t in range(steps):
        baseline = generator.generate()
        observed = injector.step(baseline.copy())
        h, theta, status = health_monitor.update(
            observed, base_values, noise_scales
        )
        result = {
            'step': t,
            'baseline': baseline,
            'observed': observed,
            'active_faults': [f['template'].id for f in injector.active_faults],
            'health_metric': h,
            'adaptive_threshold': theta,
            'health_status': status
        }
        history.append(result)
    return history

def plot_results(history, metric_names):
    steps = [h['step'] for h in history]
    observed = np.array([h['observed'] for h in history])
    health = [h['health_metric'] for h in history]
    threshold = [h['adaptive_threshold'] for h in history]

    fig, axs = plt.subplots(len(metric_names)+1, 1, figsize=(10, 8), sharex=True)
    for i, m in enumerate(metric_names):
        axs[i].plot(steps, observed[:, i], label=f"Observed {m}")
        axs[i].set_ylabel(m)
        axs[i].legend()

    axs[-1].plot(steps, health, label="Health Metric")
    axs[-1].plot(steps, threshold, label="Adaptive Threshold")
    axs[-1].set_ylabel("Health")
    axs[-1].legend()
    axs[-1].set_xlabel("Time step")
    plt.tight_layout()
    plt.savefig("simulation_results.png")

if __name__ == "__main__":
    yaml_path = 'fault_injection/templates.yaml'  # Adjust path as needed
    history = run_complete_simulation(yaml_path, steps=500)
    plot_results(history, ["RTT", "CPU", "PLR"])

    
"""if __name__ == "__main__":
    try:
        # Run simulation
        results, injector = run_complete_simulation(steps=100, seed=42)
        print(f"\nSimulation completed successfully!")
        print(f"Total steps: {len(results)}")
        print(f"Fault periods: {sum(1 for r in results if r['any_fault_active'])}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()"""
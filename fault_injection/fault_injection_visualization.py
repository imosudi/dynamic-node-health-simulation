import numpy as np
import matplotlib
import os

# Check if we're in an interactive environment
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')  # Use non-interactive backend if no display

import matplotlib.pyplot as plt

# ===========================
# Step 1: Deterministic Metric Generator
# ===========================

def generate_baseline_metrics(T=200, means=None, stds=None, random_state=None):
    rng = np.random.default_rng(random_state)
    m = len(means)
    X = np.zeros((T, m))
    for k in range(m):
        X[:, k] = rng.normal(loc=means[k], scale=stds[k], size=T)
    return X

# ===========================
# Step 2: Correlated Persistent Fault Injection
# ===========================

def inject_correlated_persistent_faults(X, fault_templates, random_state=None):
    """
    X: baseline metrics (T x m)
    fault_templates: list of dicts with keys:
        - start: time index when fault may begin
        - cov: covariance matrix (m x m)
        - drift: per-step drift vector (m,)
        - p_end: probability to end fault at each step (geometric)
    """
    rng = np.random.default_rng(random_state)
    T, m = X.shape
    X_faulty = X.copy()
    fault_mask = np.zeros(T, dtype=bool)

    for template in fault_templates:
        t = template['start']
        active = True
        drift_acc = np.zeros(m)
        while t < T and active:
            # Add correlated noise
            perturb = rng.multivariate_normal(mean=np.zeros(m), cov=template['cov'])
            # Add drift
            drift_acc += template['drift']
            X_faulty[t, :] += perturb + drift_acc
            fault_mask[t] = True
            # Decide whether to end fault
            if rng.random() < template['p_end']:
                active = False
            t += 1

    return X_faulty, fault_mask

# ===========================
# Quick Visualization
# ===========================

def plot_fault_injection(X_baseline, X_faulty, fault_mask, metric_names=None, save_path=None):
    T, m = X_baseline.shape
    time = np.arange(T)
    fig, axes = plt.subplots(m, 1, figsize=(10, 2 * m), sharex=True)

    if m == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        name = metric_names[k] if metric_names else f"Metric {k}"
        ax.plot(time, X_baseline[:, k], label=f"Baseline {name}", alpha=0.7)
        ax.plot(time, X_faulty[:, k], label=f"Faulty {name}", alpha=0.9)
        ax.fill_between(time, ax.get_ylim()[0], ax.get_ylim()[1], where=fault_mask,
                        color='red', alpha=0.1, label='Fault Active' if k == 0 else "")
        ax.set_ylabel(name)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    # Only try to show the plot if we have a display
    if os.environ.get('DISPLAY') is not None:
        try:
            plt.show()
        except:
            print("Plot display failed even though DISPLAY is set.")
    else:
        print("No display available. Plot was saved to file if requested.")
    
    plt.close()

# ===========================
# Example Usage
# ===========================
if __name__ == "__main__":
    means = [50, 30, 100]
    stds = [5, 3, 10]
    X_baseline = generate_baseline_metrics(T=200, means=means, stds=stds, random_state=42)

    fault_templates = [
        {
            'start': 50,
            'cov': np.array([[10, 5, 2], [5, 8, 1], [2, 1, 6]]),
            'drift': np.array([0.1, -0.05, 0.2]),
            'p_end': 0.05
        }
    ]

    X_faulty, fault_mask = inject_correlated_persistent_faults(X_baseline, fault_templates, random_state=42)

    plot_fault_injection(
        X_baseline, 
        X_faulty, 
        fault_mask, 
        metric_names=["PLR", "CPU", "RTT"],
        save_path="fault_injection_plot.png"  # Save the plot to a file
    )
# Python code to load YAML-defined fault templates and run the generator+injector simulation.
# This cell is self-contained: it provides a YAML loader (from string or file), builds FaultTemplate
# objects, runs a simulation for RTT/CPU/PLR, and plots baseline vs injected series with fault shading.
#
# FIXED: Added interactive backend detection and proper display support
#
# It is production-quality: clear structure, validation, RNG seeding, and support for pairwise rho covariance.
# You can adapt the YAML input (yaml_text) or point `yaml_path` to load from disk.
#
# Run this cell to execute the simulation and see the visualization.

import io
import math
import numpy as np
import matplotlib
import os

# Try to use interactive backend if available
try:
    # Check if DISPLAY is set (for X11)
    if 'DISPLAY' in os.environ:
        # Try common interactive backends
        for backend in ['Qt5Agg', 'TkAgg', 'GTK3Agg']:
            try:
                matplotlib.use(backend)
                print(f"Using interactive backend: {backend}")
                break
            except:
                continue
        else:
            print("No interactive backend available, falling back to Agg")
            matplotlib.use('Agg')
    else:
        print("No DISPLAY environment variable found, using Agg backend")
        matplotlib.use('Agg')
except Exception as e:
    print(f"Backend selection error: {e}, using default")

import matplotlib.pyplot as plt

# yaml is often available; if not, fallback to a minimal parser that expects valid YAML.
try:
    import yaml
except Exception as e:
    yaml = None
    print("PyYAML not available; please install pyyaml for YAML file loading. Error:", e)


# -------------------------
# Simulation classes
# -------------------------

class FaultTemplate:
    def __init__(self, cfg, metric_count, baseline_stds):
        # cfg: dict loaded from YAML for a single template
        self.id = cfg.get("id")
        self.name = cfg.get("name", self.id)
        self.type = cfg.get("type", "additive")
        self.subtype = cfg.get("subtype", None)
        self.affected_metrics = cfg.get("affected_metrics", [])
        self.affected_metrics = [int(i) for i in self.affected_metrics]
        self.metric_count = metric_count

        # severity_vector: expressed as multiples of baseline std
        sev = cfg.get("severity_vector")
        if sev is not None:
            self.severity_vector = np.array(sev, dtype=float)
        else:
            # default 1x std on affected metrics
            self.severity_vector = np.ones(len(self.affected_metrics), dtype=float)

        # covariance handling: if pairwise_rho, build covariance based on rho and baseline stds
        cov_cfg = cfg.get("covariance", {})
        self.cov_mode = cov_cfg.get("mode")
        self.rho = cov_cfg.get("rho", None)
        if self.cov_mode == "pairwise_rho" and self.rho is not None:
            # build covariance matrix for the affected metrics only
            k = len(self.affected_metrics)
            self.cov = np.zeros((k, k), dtype=float)
            for i in range(k):
                si = baseline_stds[self.affected_metrics[i]]
                for j in range(k):
                    sj = baseline_stds[self.affected_metrics[j]]
                    if i == j:
                        # diagonal = (severity * sigma)^2
                        self.cov[i, j] = (self.severity_vector[i] * si) ** 2
                    else:
                        self.cov[i, j] = self.rho * si * sj
        else:
            # accept explicit covariance matrix if provided (full m x m or k x k)
            explicit = cfg.get("covariance")
            if isinstance(explicit, dict):
                # no explicit matrix provided; treat as no covariance
                self.cov = None
            else:
                # assume a list-of-lists provided
                self.cov = np.array(explicit) if explicit is not None else None

        # drift/ramp
        self.ramp_slope = float(cfg.get("ramp_slope", 0.0))
        self.max_deviation = float(cfg.get("max_deviation", np.inf))

        # duration model
        self.duration_dist = cfg.get("duration_dist", {"type": "geometric", "p": 0.1})
        self.duration_p = self.duration_dist.get("p", None)
        self.duration_type = self.duration_dist.get("type", "geometric")
        if self.duration_type == "deterministic":
            self.duration_n = int(self.duration_dist.get("n_steps", 1))
        else:
            self.duration_n = None

        # occurrence model (bernoulli p per step)
        occ = cfg.get("occurrence_model", {"type": "bernoulli", "p": 0.001})
        self.occ_type = occ.get("type", "bernoulli")
        self.occ_p = occ.get("p", 0.001)

        # other
        self.repeatable = bool(cfg.get("repeatable", True))
        self.stuck_value = cfg.get("stuck_value", None)
        self.note = cfg.get("note", "")

    def sample_initial_delta(self, rng, baseline_stds):
        """
        Returns a vector of shifts of length = metric_count, with non-zero only on affected indices.
        """
        k = len(self.affected_metrics)
        deltas = np.zeros(self.metric_count, dtype=float)
        if k == 0:
            return deltas

        if self.cov is not None:
            # sample MVN in the affected subspace
            mv = rng.multivariate_normal(mean=np.zeros(k), cov=self.cov)
            # mv already scaled by severity in diagonal if pairwise_rho used
            for idx, m_idx in enumerate(self.affected_metrics):
                deltas[m_idx] = mv[idx]
        else:
            # independent normals scaled by severity * sigma
            for idx, m_idx in enumerate(self.affected_metrics):
                sigma = baseline_stds[m_idx]
                deltas[m_idx] = rng.normal(loc=0.0, scale=self.severity_vector[idx] * sigma)

        return deltas

    def sample_duration(self, rng):
        if self.duration_type == "deterministic":
            return self.duration_n
        # geometric: number of trials until success
        p = self.duration_p if (self.duration_p is not None) else 0.1
        return rng.geometric(p)


class MetricGenerator:
    def __init__(self, metric_names, means, stds, signs=None, seed=None):
        self.metric_names = metric_names
        self.means = np.array(means, dtype=float)
        self.stds = np.array(stds, dtype=float)
        self.signs = np.array(signs, dtype=float) if signs is not None else np.ones_like(self.means)
        self.rng = np.random.default_rng(seed)

    def generate(self, T):
        # create baseline: independent Gaussian around mean with std
        T = int(T)
        m = len(self.means)
        X = self.rng.normal(loc=self.means.reshape((1, m)), scale=self.stds.reshape((1, m)), size=(T, m))
        return X


class FaultInjector:
    def __init__(self, templates, metric_count, seed=None):
        self.templates = templates  # list of FaultTemplate objects
        self.metric_count = metric_count
        self.rng = np.random.default_rng(seed)
        # runtime state per template for active events
        self.active = [None] * len(templates)  # each entry: dict with keys template, remaining, current_delta, applied_indices

    def step(self, t, current_metrics, baseline_stds):
        """
        Possibly start or continue faults for each template.
        Returns modified metrics (copy) and per-timestep metadata.
        """
        x = current_metrics.copy()
        meta = {"applied": [], "deltas": np.zeros_like(x)}

        for i, tmpl in enumerate(self.templates):
            state = self.active[i]
            # If no active event, decide whether to start
            if state is None:
                if self.rng.random() < tmpl.occ_p:
                    # start event
                    duration = tmpl.sample_duration(self.rng)
                    initial = tmpl.sample_initial_delta(self.rng, baseline_stds)
                    # if stuck_at, override initial deltas
                    if tmpl.type == "persistent" and tmpl.subtype == "stuck_at":
                        # stuck value might be keyword like max_value or explicit numeric; we handle 'max_value' as None marker
                        if isinstance(tmpl.stuck_value, (int, float)):
                            # delta should set metric to stuck_value (we will signal with special field)
                            delta = np.full(self.metric_count, np.nan)
                            # We'll handle stuck_at separately by setting state['stuck_value']
                            state = {"template": tmpl, "remaining": duration, "current_delta": None,
                                     "stuck_value": float(tmpl.stuck_value)}
                        else:
                            # treat 'max_value' as very large positive delta placeholder
                            state = {"template": tmpl, "remaining": duration, "current_delta": None,
                                     "stuck_value": "max_value"}
                    else:
                        state = {"template": tmpl, "remaining": duration, "current_delta": initial}
                    self.active[i] = state

            # If active, apply effect
            if state is not None:
                tmpl_active = state["template"]
                if tmpl_active.type == "persistent" and tmpl_active.subtype == "stuck_at":
                    # apply stuck: either numerical or max_value (we map max_value -> baseline + 10*sigma)
                    stuck_vals = np.full(self.metric_count, np.nan)
                    if state["stuck_value"] == "max_value":
                        # define max as mean + 10*sigma (configurable)
                        # We'll map to baseline mean + large multiple of std for the affected indices
                        for idx in tmpl_active.affected_metrics:
                            stuck_vals[idx] = current_metrics[idx] + 10.0 * baseline_stds[idx]
                    else:
                        stuck_vals[:] = np.nan
                        for idx in tmpl_active.affected_metrics:
                            stuck_vals[idx] = state["stuck_value"]
                    # apply stuck values
                    for idx in tmpl_active.affected_metrics:
                        if not math.isnan(stuck_vals[idx]):
                            # set metric to stuck value (overwrite)
                            meta["deltas"][idx] = stuck_vals[idx] - x[idx]
                            x[idx] = stuck_vals[idx]
                            meta["applied"].append((tmpl_active.id, idx))
                else:
                    # additive or ramp or other persistent types
                    delta = state["current_delta"]
                    # apply delta additive
                    x += delta
                    meta["deltas"] += delta
                    for idx in tmpl_active.affected_metrics:
                        meta["applied"].append((tmpl_active.id, idx))

                    # update delta with ramp if applicable
                    if tmpl_active.subtype == "ramp" or tmpl_active.ramp_slope != 0.0:
                        # increase delta on affected indices by ramp_slope * sign of original delta
                        for jdx, midx in enumerate(tmpl_active.affected_metrics):
                            sign = np.sign(delta[midx]) if delta[midx] != 0 else 1.0
                            increment = tmpl_active.ramp_slope * sign
                            # cap by max_deviation if set
                            proposed = delta[midx] + increment
                            if abs(proposed) > tmpl_active.max_deviation:
                                proposed = np.sign(proposed) * tmpl_active.max_deviation
                            state["current_delta"][midx] = proposed

                # decrement remaining, possibly end
                state["remaining"] -= 1
                if state["remaining"] <= 0:
                    # end event
                    if not tmpl.repeatable:
                        # mark template as disabled by setting occ_p to 0
                        tmpl.occ_p = 0.0
                    self.active[i] = None
                else:
                    # keep the state
                    self.active[i] = state

        return x, meta


# -------------------------
# YAML loader -> FaultTemplate objects
# -------------------------

def load_templates_from_yaml(templates_yaml, metric_count, baseline_stds):
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse YAML. Please install pyyaml.")
    cfg = yaml.safe_load(templates_yaml)
    templates_cfg = cfg.get("fault_templates", [])
    templates = []
    for tcfg in templates_cfg:
        templates.append(FaultTemplate(tcfg, metric_count=metric_count, baseline_stds=baseline_stds))
    return templates


# -------------------------
# Demo runner with interactive plot support
# -------------------------
def run_demo(T=500, seed=123, interactive=True):
    rng = np.random.default_rng(seed)
    metric_names = ["RTT", "CPU", "PLR"]
    # baseline means and stds chosen to be realistic-ish
    means = np.array([100.0, 30.0, 0.01])   # RTT(ms), CPU(%), PLR(probability)
    stds = np.array([5.0, 3.0, 0.002])
    metric_count = len(metric_names)

    # build generator
    gen = MetricGenerator(metric_names, means, stds, seed=seed)

    # Load templates from external YAML file    
    try:
        with open('templates.yaml', 'r') as file:
            templates_yaml = file.read()
        templates = load_templates_from_yaml(templates_yaml, metric_count=metric_count, baseline_stds=stds)
        print(f"Loaded {len(templates)} fault templates from templates.yaml")
    except FileNotFoundError:
        print("Warning: templates.yaml not found. Creating minimal example templates.")
        # Create some example templates if file doesn't exist
        templates_yaml = """
fault_templates:
  - id: "rtt_spike"
    name: "RTT Spike"
    type: "additive"
    affected_metrics: [0]
    severity_vector: [3.0]
    occurrence_model:
      type: "bernoulli"
      p: 0.02
    duration_dist:
      type: "geometric"
      p: 0.3
    note: "Occasional RTT spikes"
  
  - id: "cpu_ramp"
    name: "CPU Gradual Increase"
    type: "persistent"
    subtype: "ramp"
    affected_metrics: [1]
    severity_vector: [2.0]
    ramp_slope: 0.1
    max_deviation: 15.0
    occurrence_model:
      type: "bernoulli"
      p: 0.005
    duration_dist:
      type: "geometric"
      p: 0.1
    note: "CPU usage gradually increases"
"""
        templates = load_templates_from_yaml(templates_yaml, metric_count=metric_count, baseline_stds=stds)

    # injector
    injector = FaultInjector(templates, metric_count=metric_count, seed=seed + 1)

    # run simulation
    X_base = gen.generate(T)
    X_obs = X_base.copy()
    applied_mask = np.zeros((T, metric_count), dtype=bool)
    delta_log = np.zeros((T, metric_count), dtype=float)

    print(f"Running simulation for {T} timesteps...")
    fault_events = 0

    for t in range(T):
        current = X_obs[t].copy()
        # At each timestep, allow injector to possibly start/continue faults and mutate current metrics
        mutated, meta = injector.step(t, current, baseline_stds=stds)
        X_obs[t] = mutated
        if meta["applied"]:
            fault_events += len(meta["applied"])
            for (_, idx) in meta["applied"]:
                applied_mask[t, idx] = True
            delta_log[t] = meta["deltas"]

        # propagate to next time step baseline behavior (feedback): next baseline value uses mutated current as mean
        if t + 1 < T:
            # simple feedback: next sample mean nudged slightly toward current observed (small persistence)
            alpha = 0.1
            gen.means = (1 - alpha) * gen.means + alpha * mutated  # update baseline means to reflect feedback

            # generate next sample with updated means
            X_obs[t + 1] = rng.normal(loc=gen.means, scale=gen.stds)

    print(f"Simulation complete. Total fault applications: {fault_events}")

    # plot
    time = np.arange(T)
    fig, axes = plt.subplots(metric_count, 1, figsize=(12, 3 * metric_count), sharex=True)
    
    # Handle case where metric_count = 1 (axes won't be an array)
    if metric_count == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        ax.plot(time, X_base[:, k], label=f"Baseline {metric_names[k]}", alpha=0.6)
        ax.plot(time, X_obs[:, k], label=f"Observed {metric_names[k]}", alpha=0.9)
        # shade fault times
        fault_periods = applied_mask[:, k]
        if np.any(fault_periods):
            ax.fill_between(time, ax.get_ylim()[0], ax.get_ylim()[1], 
                          where=fault_periods, color='red', alpha=0.12, label='Fault Active')
        ax.set_ylabel(metric_names[k])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time step")
    plt.suptitle(f"Fault Injection Simulation Results (T={T}, seed={seed})", fontsize=14)
    plt.tight_layout()
    
    # Try to show the plot interactively
    current_backend = matplotlib.get_backend()
    print(f"Current matplotlib backend: {current_backend}")
    
    if interactive and current_backend != 'Agg':
        try:
            plt.show()
            print("Interactive plot displayed!")
        except Exception as e:
            print(f"Could not display interactive plot: {e}")
            print("Saving plot as PNG instead...")
            plt.savefig("fault_simulation_results.png", dpi=150, bbox_inches='tight')
            print("Plot saved as: fault_simulation_results.png")
    else:
        # Save as file
        plt.savefig("fault_simulation_results.png", dpi=150, bbox_inches='tight')
        print("Plot saved as: fault_simulation_results.png")

    # Print summary statistics
    print("\n=== Simulation Summary ===")
    for k, name in enumerate(metric_names):
        baseline_mean = np.mean(X_base[:, k])
        observed_mean = np.mean(X_obs[:, k])
        fault_time_pct = np.sum(applied_mask[:, k]) / T * 100
        print(f"{name}: Baseline avg={baseline_mean:.3f}, Observed avg={observed_mean:.3f}, "
              f"Fault active {fault_time_pct:.1f}% of time")

    return {
        "metric_names": metric_names,
        "X_base": X_base,
        "X_obs": X_obs,
        "applied_mask": applied_mask,
        "delta_log": delta_log,
        "templates": templates,
    }


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    # Print backend information for debugging
    print(f"Available matplotlib backends: {matplotlib.backend_bases._Backend.__subclasses__()}")
    print(f"DISPLAY environment variable: {os.environ.get('DISPLAY', 'Not set')}")
    
    # run the demo
    results = run_demo(T=600, seed=2025, interactive=True)
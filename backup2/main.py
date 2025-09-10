import os
from modules.sample_data_generator import generate_sample_node_list_csv
from modules.node_profile import NodeProfile
from modules.fault_injector import FaultInjector
from modules.health_scorer import HealthScorer
from modules.adaptive_threshold import AdaptiveThreshold
from modules.health_classifier import HealthClassifier
from modules.simulation_controller import SimulationController
from modules.live_plotter import LivePlotter

def main():
    node_list_csv = 'data/node_list.csv'

    # Generate sample node_list.csv if it doesn't exist
    if not os.path.exists(node_list_csv):
        print(f"{node_list_csv} not found. Generating sample data...")
        generate_sample_node_list_csv(node_list_csv)
        print(f"Sample node_list.csv generated successfully.")

    # Load Node Profiles
    node_profiles = NodeProfile.load_profiles_from_csv(node_list_csv)

    # Initialize Modules
    fault_injector = FaultInjector(severity_multipliers={'PLR': 3, 'CPU': 2, 'RTT': 1.5})
    health_scorer = HealthScorer()
    adaptive_threshold = AdaptiveThreshold(alpha=0.4)
    health_classifier = HealthClassifier()
    live_plotter = LivePlotter()

    # Run Simulation
    controller = SimulationController(node_profiles, fault_injector, health_scorer, adaptive_threshold, health_classifier)
    controller.run_simulation(total_timesteps=100)

    # Export Logs
    controller.export_logs('logs/node_health_log.json', 'logs/node_health_log.csv')

    # Plot Results
    live_plotter.plot_simulation_results(controller.simulation_log, node_profiles)

if __name__ == "__main__":
    main()

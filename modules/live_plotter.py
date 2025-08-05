import matplotlib.pyplot as plt

class LivePlotter:
    def plot_simulation_results(self, simulation_log, node_profiles):
        node_ids = [node.node_id for node in node_profiles]
        node_data = {node_id: {'time': [], 'health': [], 'threshold': [], 'status': []} for node_id in node_ids}

        for entry in simulation_log:
            node_id = entry['node_id']
            node_data[node_id]['time'].append(entry['time_step'])
            node_data[node_id]['health'].append(entry['health_score'])
            node_data[node_id]['threshold'].append(entry['threshold'])
            node_data[node_id]['status'].append(entry['status'])

        fig, axs = plt.subplots(len(node_ids), 1, figsize=(10, 3 * len(node_ids)), sharex=True)
        if len(node_ids) == 1:
            axs = [axs]  # Ensure axs is always iterable
        
        for ax, node_id in zip(axs, node_ids):
            ax.plot(node_data[node_id]['time'], node_data[node_id]['health'], label='Health Score', color='blue')
            ax.plot(node_data[node_id]['time'], node_data[node_id]['threshold'], label='Threshold', color='orange')
            
            # Mark Faulty Points
            faulty_times = [t for t, s in zip(node_data[node_id]['time'], node_data[node_id]['status']) if s == 'Faulty']
            faulty_scores = [h for h, s in zip(node_data[node_id]['health'], node_data[node_id]['status']) if s == 'Faulty']
            ax.scatter(faulty_times, faulty_scores, color='red', label='Faulty', marker='x')

            ax.set_title(f"Node {node_id} Health Metrics")
            ax.legend()

        plt.xlabel("Time Step")
        plt.tight_layout()
        plt.savefig('logs/simulation_plot.png')
        print("Plot saved to logs/simulation_plot.png")


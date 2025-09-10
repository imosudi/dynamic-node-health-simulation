import csv
import os

def generate_sample_node_list_csv(filepath):
    # Define node entries with tiers L0–L4
    sample_nodes = [
        # L0 Cloud Node
        {'node_id': 'CloudDB_Server', 'layer': 0},
        
        # L1 Farm Office Node
        {'node_id': 'L1N_01', 'layer': 1},
        
        # L2 Zone Supervisors
        {'node_id': 'L2N_01', 'layer': 2},
        {'node_id': 'L2N_02', 'layer': 2},
        {'node_id': 'L2N_03', 'layer': 2},
        {'node_id': 'L2N_04', 'layer': 2},
        
        # L3 Field Units (this is where the problem was)
    ] + [{'node_id': f'L3N_{i:02d}', 'layer': 3} for i in range(1, 13)]


    # L4 Field Sensors
    for i in range(1, 37):
        sample_nodes.append({'node_id': f'L4N_{i:02d}', 'layer': 4})

    # Static metric profiles for all nodes (can be extended for realistic profiling)
    default_means = {'PLR': 0.02, 'CPU': 30.0, 'RTT': 100.0}
    default_stds = {'PLR': 0.01, 'CPU': 10.0, 'RTT': 20.0}
    default_weights = {'PLR': 0.4, 'CPU': 0.3, 'RTT': 0.3}

    # Ensure data directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = [
            'node_id', 'layer',
            'mean_PLR', 'std_PLR', 'weight_PLR',
            'mean_CPU', 'std_CPU', 'weight_CPU',
            'mean_RTT', 'std_RTT', 'weight_RTT'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for node in sample_nodes:
            writer.writerow({
                'node_id': node['node_id'],
                'layer': node['layer'],
                'mean_PLR': default_means['PLR'],
                'std_PLR': default_stds['PLR'],
                'weight_PLR': default_weights['PLR'],
                'mean_CPU': default_means['CPU'],
                'std_CPU': default_stds['CPU'],
                'weight_CPU': default_weights['CPU'],
                'mean_RTT': default_means['RTT'],
                'std_RTT': default_stds['RTT'],
                'weight_RTT': default_weights['RTT'],
            })

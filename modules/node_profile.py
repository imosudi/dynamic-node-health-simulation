import csv

class NodeProfile:
    def __init__(self, node_id, layer, metric_means, metric_stds, metric_weights):
        self.node_id = node_id
        self.layer = layer
        self.metric_means = metric_means
        self.metric_stds = metric_stds
        self.metric_weights = metric_weights

    @staticmethod
    def from_csv_row(row):
        node_id = row['node_id']
        layer = int(row['layer'])
        metric_means = {
            'PLR': float(row['mean_PLR']),
            'CPU': float(row['mean_CPU']),
            'RTT': float(row['mean_RTT'])
        }
        metric_stds = {
            'PLR': float(row['std_PLR']),
            'CPU': float(row['std_CPU']),
            'RTT': float(row['std_RTT'])
        }
        metric_weights = {
            'PLR': float(row['weight_PLR']),
            'CPU': float(row['weight_CPU']),
            'RTT': float(row['weight_RTT'])
        }
        return NodeProfile(node_id, layer, metric_means, metric_stds, metric_weights)

    @staticmethod
    def load_profiles_from_csv(filepath):
        profiles = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                profiles.append(NodeProfile.from_csv_row(row))
        return profiles

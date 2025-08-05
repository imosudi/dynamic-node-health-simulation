class AdaptiveThreshold:
    def __init__(self, alpha):
        self.alpha = alpha
        self.thresholds = {}  # node_id -> theta

    def initialise_thresholds(self, node_profiles):
        for profile in node_profiles:
            self.thresholds[profile.node_id] = 0.0  # Start with healthy score

    def update_threshold(self, node_id, current_health_score):
        prev_theta = self.thresholds[node_id]
        new_theta = self.alpha * current_health_score + (1 - self.alpha) * prev_theta
        self.thresholds[node_id] = new_theta
        return new_theta

class HealthScorer:
    def compute_health_score(self, node_profile, observed_metrics):
        health_score = 0.0
        for metric in observed_metrics:
            mean = node_profile.metric_means[metric]
            std = node_profile.metric_stds[metric]
            weight = node_profile.metric_weights[metric]
            z_score = (observed_metrics[metric] - mean) / std
            health_score += weight * z_score
        return health_score

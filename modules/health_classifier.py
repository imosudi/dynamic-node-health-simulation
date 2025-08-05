class HealthClassifier:
    def classify_health(self, health_score, threshold):
        if health_score >= 1.0:
            return "Good"
        elif 0 <= health_score < 1.0:
            return "Fair"
        elif threshold <= health_score < 0:
            return "Poor"
        else:
            return "Faulty"

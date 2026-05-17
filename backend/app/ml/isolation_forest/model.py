import os
import joblib
import logging
import numpy as np

logger = logging.getLogger(__name__)

class IFAnomalyScorer:
    def __init__(self):
        self.models_dir = "models_trained/"
        self.models = {}
        self.logged_classes = set()
        classes = ['camera', 'sensor', 'thermostat', 'access_control', 'medical', 'industrial']
        loaded_count = 0
        
        for cls in classes:
            model_path = os.path.join(self.models_dir, f"if_{cls}.joblib")
            if os.path.exists(model_path):
                try:
                    self.models[cls] = joblib.load(model_path)
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to load IF model for {cls}: {e}")
                    
        logger.info(f"Loaded {loaded_count}/6 Isolation Forest models successfully.")

    def score(self, device_class: str, feature_vector: list[float]) -> float:
        if device_class not in self.models:
            return 0.0
            
        model = self.models[device_class]
        features_2d = np.array(feature_vector).reshape(1, -1)
        
        raw_score = model.decision_function(features_2d)[0]
        
        if device_class not in self.logged_classes:
            logger.info(f"First IF Score for {device_class}: {raw_score}")
            self.logged_classes.add(device_class)
            
        # Invert and normalize to 0-1 (higher means more anomalous)
        normalized_score = max(0.0, min(1.0, 0.5 - raw_score))
        with open('debug_if.log', 'a') as f:
            f.write(f"IF DEBUG: class={device_class} raw={raw_score} norm={normalized_score}\n")
        return float(normalized_score)

    def score_anomaly(self, device_class: str, feature_vector: list[float]) -> float:
        """Alias to support existing pipeline calls."""
        return self.score(device_class, feature_vector)

if_scorer = IFAnomalyScorer()

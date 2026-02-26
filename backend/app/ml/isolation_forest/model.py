import os
import joblib
import logging
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

MODELS_DIR = "models_trained/"

class IF_AnomalyScorer:
    """
    Scikit-Learn based Statistical anomaly checker using Isolation Forests.
    Focuses specifically on detecting out-of-distribution volumetric or categorical
    network anomalies per device class (e.g. C2 Botnet port/IP changes).
    """

    def __init__(self):
        self.class_models = {}

    def load_model(self, device_class: str) -> IsolationForest:
        """Load an Isolation Forest trained explicitly for a class of devices."""
        if device_class in self.class_models:
            return self.class_models[device_class]
            
        model_path = os.path.join(MODELS_DIR, f"iforest_{device_class}.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Isolation Forest missing for class '{device_class}'")
            
        model = joblib.load(model_path)
        self.class_models[device_class] = model
        return model

    def score_anomaly(self, device_class: str, current_features: list[float]) -> float:
        """
        Calculate probability that the incoming features represent anomaly behavior.
        Isolation Forest explicitly targets out-of-bounds structural attacks.
        Returns a normalized anomaly float from 0.0 to 1.0 (1.0 = Highly Anomalous).
        """
        try:
            model = self.load_model(device_class)
        except FileNotFoundError:
            return 0.0 # Return perfectly normal until baseline finishes training
            
        features_2d = np.array(current_features).reshape(1, -1)
        
        # Scikit-learn outputs scores where lower values mean higher anomaly
        # Typically around 0.0 (normal) to -0.2 (anomaly)
        raw_score = model.decision_function(features_2d)[0]
        
        # Invert and normalize to 0-1 range for uniform scoring format across the AI engine
        normalized_score = max(0.0, min(1.0, 0.5 - (raw_score * 2.0)))
        
        # Add penalty modifier if the model explicitly classified it as a structural outlier (-1)
        prediction = model.predict(features_2d)[0]
        if prediction == -1:
            normalized_score = max(0.8, normalized_score)
            
        return float(normalized_score)

# Global Instance
if_scorer = IF_AnomalyScorer()

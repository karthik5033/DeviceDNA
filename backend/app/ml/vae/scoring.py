import os
import json
import logging
import torch
import torch.nn.functional as F
from app.ml.vae.model import DeviceVAE

logger = logging.getLogger(__name__)

class VAETwinScorer:
    def __init__(self):
        self.models_dir = "models_trained/"
        self.twins = {}
        loaded_count = 0
        
        for i in range(1, 51):
            device_id = f"SIM-{i:04d}"
            pt_path = os.path.join(self.models_dir, f"vae_{device_id}.pt")
            json_path = os.path.join(self.models_dir, f"vae_{device_id}_norm.json")
            
            if os.path.exists(pt_path) and os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        norm_params = json.load(f)
                        
                    model = DeviceVAE(input_dim=14)
                    model.load_state_dict(torch.load(pt_path))
                    model.eval()
                    
                    self.twins[device_id] = {
                        'model': model,
                        'norm': norm_params
                    }
                    loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to load twin for {device_id}: {e}")
                    
        logger.info(f"Loaded {loaded_count} VAE Digital Twins successfully.")

    def score(self, device_id: str, feature_vector: list[float]) -> float:
        if device_id not in self.twins:
            return 0.0
            
        twin = self.twins[device_id]
        model = twin['model']
        norm = twin['norm']
        
        # Normalize features
        normalized_features = []
        for val, f_min, f_max in zip(feature_vector, norm['min'], norm['max']):
            if f_max - f_min == 0:
                normalized_features.append(0.0)
            else:
                normalized_features.append((val - f_min) / (f_max - f_min))
                
        with torch.no_grad():
            tensor_x = torch.FloatTensor(normalized_features).unsqueeze(0)
            recon_x, mu, logvar = model(tensor_x)
            mse = F.mse_loss(recon_x, tensor_x, reduction='mean').item()
            
            # Normalize to 0-1 anomaly score with threshold 0.5
            anomaly_score = max(0.0, min(1.0, mse / 0.5))
            return float(anomaly_score)

    def score_deviation(self, device_id: str, feature_vector: list[float]) -> float:
        """Alias to support existing pipeline calls."""
        return self.score(device_id, feature_vector)

twin_scorer = VAETwinScorer()

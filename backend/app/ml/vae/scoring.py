import os
import logging

try:
    import torch
    import torch.nn.functional as F
    from app.ml.vae.model import DeviceVAE
except ImportError:
    torch = None
    F = None
    DeviceVAE = None

logger = logging.getLogger(__name__)

class VAE_TwinScorer:
    """
    Evaluates new incoming device telemetry (14D vectors) against the 
    pre-trained VAE logic model to generate a Twin Deviation Score 0.0-1.0.
    1.0 means practically identical to baseline (No Anomaly).
    0.0 means completely out of bounds (100% Anomaly).
    """
    
    def __init__(self, models_dir: str = "models_trained/"):
        self.models_dir = models_dir
        self.device_twins = {} # Cache loaded PyTorch models

    def load_twin(self, device_id: str) -> DeviceVAE:
        """Hydrate the correct Digital Twin into memory."""
        if device_id in self.device_twins:
            return self.device_twins[device_id]
            
        model_path = os.path.join(self.models_dir, f"vae_{device_id}.pt")
        
        # In a real environment, wait or trigger baseline training if missing
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Digital Twin not found for {device_id}")

        model = DeviceVAE(input_dim=14)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        self.device_twins[device_id] = model
        return model

    def score_deviation(self, device_id: str, current_features: list[float]) -> float:
        """
        Calculates how heavily a new piece of telemetry data deviates from
        what the Digital Twin predicts is normal behavior for this device.
        """
        try:
            model = self.load_twin(device_id)
        except FileNotFoundError:
            # Not scored if baseline isn't finished training
            return -1.0 

        # Forward pass without calculating gradients
        with torch.no_grad():
            tensor_x = torch.FloatTensor(current_features)
            recon_x, mu, logvar = model(tensor_x)
            
            # Reconstruction Mean Squared Error
            mse = F.mse_loss(recon_x, tensor_x, reduction='sum').item()
            
            # Normalize MSE into a bounded 0-1 Trust Score
            # If MSE is near 0, score is 1.0 (perfectly normal).
            # Example heuristic: Exponential decay based on empirical MSE thresholds.
            # Assuming average normal MSE ~ 0.5, anomalous ~ >5.0
            
            # Bound it using exponential mapping
            deviation_score = max(0.0, min(1.0, 1.0 - (mse / 10.0)))

            return deviation_score

# Singleton Instance
twin_scorer = VAE_TwinScorer()

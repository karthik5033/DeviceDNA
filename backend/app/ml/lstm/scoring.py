import os
import json
import logging
import torch
import torch.nn.functional as F
from app.ml.lstm.model import TimeSeriesLSTM

logger = logging.getLogger(__name__)

class LSTMScorer:
    def __init__(self):
        self.models_dir = "models_trained/"
        self.model_path = os.path.join(self.models_dir, "lstm_shared.pt")
        self.norm_path = os.path.join(self.models_dir, "lstm_shared_norm.json")
        self.model = None
        self.mins = None
        self.maxs = None
        self.ranges = None
        self.threshold = 3.0  # Same empirical threshold logic as VAE
        
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = TimeSeriesLSTM(input_dim=14, hidden_dim=64, num_layers=2, output_dim=14)
                self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
                self.model.eval()
                logger.info("Loaded LSTM Shared Model successfully.")
            except Exception as e:
                logger.error(f"Failed to load LSTM model: {e}")
                self.model = None
        else:
            logger.warning(f"LSTM model not found at {self.model_path}. LSTM scoring disabled.")

        if os.path.exists(self.norm_path):
            try:
                with open(self.norm_path, 'r') as f:
                    norm_params = json.load(f)
                self.mins = torch.FloatTensor(norm_params['mins'])
                self.maxs = torch.FloatTensor(norm_params['maxs'])
                self.ranges = self.maxs - self.mins
                self.ranges[self.ranges == 0] = 1.0
                logger.info("Loaded LSTM normalization params successfully.")
            except Exception as e:
                logger.error(f"Failed to load LSTM norm params: {e}")
        else:
            logger.warning(f"LSTM norm params not found at {self.norm_path}. Will use raw features.")

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply min-max normalization using stored params."""
        if self.mins is not None:
            return (tensor - self.mins) / self.ranges
        return tensor

    def score(self, device_id: str, feature_sequence: list[list[float]]) -> float:
        """
        Accepts a sequence of up to 12 recent feature vectors.
        Uses the first N-1 to predict the Nth, then computes MSE against the actual Nth vector.
        Returns anomaly score normalized to [0, 1].
        """
        if self.model is None or len(feature_sequence) < 2:
            return 0.0
            
        # Cap at the most recent 12 vectors to match training window
        seq = feature_sequence[-12:]
        
        # The first N-1 vectors are the input X, the last is the target Y
        x_data = seq[:-1]
        y_data = seq[-1]
        
        with torch.no_grad():
            tensor_x = self._normalize(torch.FloatTensor(x_data)).unsqueeze(0)  # (1, seq_len, 14)
            tensor_y = self._normalize(torch.FloatTensor(y_data)).unsqueeze(0)  # (1, 14)
            
            prediction = self.model(tensor_x)
            mse = F.mse_loss(prediction, tensor_y, reduction='mean').item()
            
            # Normalize to 0-1 using dynamic threshold (same approach as VAE)
            anomaly_score = max(0.0, min(1.0, mse / self.threshold))
                
            return float(anomaly_score)

lstm_scorer = LSTMScorer()

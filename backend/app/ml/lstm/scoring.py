import os
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from app.ml.lstm.model import TimeSeriesLSTM

logger = logging.getLogger(__name__)

# Feature distribution profiles for calibration (same as train_isolation_forest.py)
_PROFILES = {
    'camera': {
        'means': [80.0, 1200000.0, 1520.0, 782.0, 2500.0, 0.85, 0.15, 0.25, 0.15, 0.10, 0.50, 3.0, 4.0, 0.15],
        'stds':  [15.0, 300000.0,  300.0,  100.0, 500.0,  0.05, 0.05, 0.05, 0.05, 0.02, 0.05, 1.0, 1.0, 0.05]
    },
    'sensor': {
        'means': [20.0, 10000.0, 60.0, 160.0, 2500.0, 0.80, 0.20, 0.20, 0.0, 0.15, 0.65, 2.0, 2.0, 0.01],
        'stds':  [ 5.0,  2000.0, 10.0,  20.0,  500.0, 0.05, 0.05, 0.05, 0.0, 0.05, 0.05, 0.0, 0.0, 0.01]
    },
    'thermostat': {
        'means': [15.0, 15000.0, 50.0, 288.0, 2500.0, 0.60, 0.40, 0.0, 0.50, 0.25, 0.25, 3.0, 3.0, 0.30],
        'stds':  [ 5.0,  5000.0, 15.0,  40.0,  500.0, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 1.0, 1.0, 0.10]
    },
    'access_control': {
        'means': [30.0, 60000.0, 110.0, 544.0, 2500.0, 0.70, 0.30, 0.0, 0.40, 0.20, 0.40, 4.0, 3.0, 0.10],
        'stds':  [10.0, 20000.0,  30.0,  50.0,  500.0, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 1.0, 1.0, 0.05]
    },
    'medical': {
        'means': [40.0, 320000.0, 150.0, 2080.0, 2500.0, 0.80, 0.20, 0.0, 0.30, 0.10, 0.60, 5.0, 4.0, 0.05],
        'stds':  [12.0, 100000.0,  40.0,  200.0,  500.0, 0.05, 0.05, 0.0, 0.05, 0.05, 0.05, 2.0, 1.0, 0.03]
    },
    'industrial': {
        'means': [50.0, 150000.0, 140.0, 1056.0, 2500.0, 0.80, 0.20, 0.0, 0.0, 0.10, 0.90, 3.0, 3.0, 0.02],
        'stds':  [15.0,  40000.0,  35.0,  100.0,  500.0, 0.05, 0.05, 0.0, 0.0, 0.05, 0.05, 1.0, 0.0, 0.01]
    }
}


class LSTMScorer:
    def __init__(self):
        self.models_dir = "models_trained/"
        self.model_path = os.path.join(self.models_dir, "lstm_shared.pt")
        self.norm_path = os.path.join(self.models_dir, "lstm_shared_norm.json")
        self.model = None
        self.mins = None
        self.maxs = None
        self.ranges = None
        self.threshold = 1.0  # Fallback; overwritten by calibration
        
        self._load_model()
        self._calibrate_threshold()
        
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

    def _calibrate_threshold(self, n_passes: int = 100, seq_len: int = 12, percentile: float = 95.0):
        """
        Generate temporally-correlated normal sequences via random walk around class means,
        run inference, and set the anomaly threshold to the given percentile of MSE values.
        
        Uses random walk (not independent samples) because the LSTM was trained on sliding
        windows of gradually-varying traffic — each timestep is a small perturbation of
        the previous one. Independent Gaussian draws produce unrealistically high MSE.
        """
        if self.model is None:
            logger.warning("LSTM model not loaded — skipping calibration.")
            return
        
        rng = np.random.default_rng(seed=42)
        class_names = list(_PROFILES.keys())
        mse_values = []

        with torch.no_grad():
            for _ in range(n_passes):
                cls = rng.choice(class_names)
                means = np.array(_PROFILES[cls]['means'], dtype=np.float32)
                stds = np.array(_PROFILES[cls]['stds'], dtype=np.float32)

                # Build a temporally-correlated sequence via random walk:
                # start near the class mean, then apply small drift each step
                vectors = np.zeros((seq_len + 1, len(means)), dtype=np.float32)
                vectors[0] = np.maximum(rng.normal(loc=means, scale=stds * 0.5), 0.0)
                for t in range(1, seq_len + 1):
                    # Small perturbation: 10% of std per timestep (mimics 5-min window drift)
                    drift = rng.normal(loc=0.0, scale=stds * 0.1, size=len(means))
                    vectors[t] = np.maximum(vectors[t - 1] + drift, 0.0)

                x_data = vectors[:seq_len]
                y_data = vectors[seq_len]

                tensor_x = self._normalize(torch.FloatTensor(x_data)).unsqueeze(0)
                tensor_y = self._normalize(torch.FloatTensor(y_data)).unsqueeze(0)

                prediction = self.model(tensor_x)
                mse = F.mse_loss(prediction, tensor_y, reduction='mean').item()
                mse_values.append(mse)

        self.threshold = float(np.percentile(mse_values, percentile))
        logger.info(
            f"LSTM calibration complete: {n_passes} passes, "
            f"p50={np.percentile(mse_values, 50):.6f}, "
            f"p95={self.threshold:.6f}, "
            f"max={max(mse_values):.6f}"
        )

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
        if self.model is None or len(feature_sequence) < 6:
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

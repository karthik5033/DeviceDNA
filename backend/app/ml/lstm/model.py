import torch
import torch.nn as nn

class TimeSeriesLSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) network designed to learn the sequential
    temporal patterns of device network features over sliding time windows.
    Detects anomalies by trying to predict the next `feature_vector` and scoring
    the exact error difference.
    """
    def __init__(self, input_dim=14, hidden_dim=64, num_layers=2, output_dim=14):
        super(TimeSeriesLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        # batch_first=True expects shape (batch_size, sequence_length, features)
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Fully connected layer to map LSTM outputs back to the original feature dimension
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length = 12 (1 hour of 5-min flows), features = 14)
        """
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # We only care about the output predictions, not the raw states
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        # out[:, -1, :] shape: (batch_size, hidden_dim)
        out = self.fc(out[:, -1, :])
        return out

def lstm_anomaly_score(predicted_features, actual_features) -> float:
    """
    Computes Mean Squared Error between the LSTM's prediction of what the NEXT
    5-minute window telemetry should be vs what actually arrived.
    """
    mse = torch.nn.functional.mse_loss(predicted_features, actual_features, reduction='none')
    # Averages over the 14 feature dimensions to return a single temporal error metric
    return torch.mean(mse).item()

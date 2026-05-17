import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader, TensorDataset

# To use simulator code from backend root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.lstm.model import TimeSeriesLSTM
from simulator.device_profiles import FLEET
from simulator.traffic_generator import generate_flow
from app.services.feature_extraction import extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "models_trained/"
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_class_sequences(device_class: str, num_samples: int = 1000, seq_len: int = 12):
    """
    Generate `num_samples` of sequence length `seq_len` for a given device class, 
    plus the +1 next target vector.
    """
    # Find a representative device profile for this class
    dev = next(d for d in FLEET if d['device_class'] == device_class)
    
    # Generate continuous feature vectors
    # We need (num_samples + seq_len) vectors to create overlapping windows
    total_vectors = num_samples + seq_len
    
    vectors = []
    for _ in range(total_vectors):
        flows = [generate_flow(dev) for _ in range(25)]
        feat_vec = extract_features(dev['id'], dev['device_class'], flows)
        vectors.append(feat_vec.to_tensor_list())
        
    tensor_data = torch.FloatTensor(vectors)
    
    # Create sliding windows
    x_data = []
    y_data = []
    for i in range(num_samples):
        x_data.append(tensor_data[i : i + seq_len])
        y_data.append(tensor_data[i + seq_len])
        
    return torch.stack(x_data), torch.stack(y_data)

def normalize_sequences(x: torch.Tensor, y: torch.Tensor):
    """
    Min-Max normalize across the feature dimension (dim=2 for x, dim=1 for y).
    Flattens x to 2D to compute global mins/maxs across all timesteps and samples.
    Returns (normalized_x, normalized_y, mins, maxs).
    """
    # x shape: (N, seq_len, 14), y shape: (N, 14)
    # Flatten x to (N * seq_len, 14) to get global min/max per feature
    flat_x = x.reshape(-1, x.shape[-1])
    all_data = torch.cat([flat_x, y], dim=0)  # (N*seq_len + N, 14)
    
    mins, _ = all_data.min(dim=0)
    maxs, _ = all_data.max(dim=0)
    
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # Avoid division by zero for constant features
    
    norm_x = (x - mins) / ranges
    norm_y = (y - mins) / ranges
    
    return norm_x, norm_y, mins, maxs

if __name__ == "__main__":
    logger.info(f"Beginning LSTM Shared Training. Target Device: {device}")
    
    # Find unique device classes
    device_classes = list(set(d['device_class'] for d in FLEET))
    logger.info(f"Found {len(device_classes)} device classes: {device_classes}")
    
    all_x = []
    all_y = []
    
    for cls in device_classes:
        logger.info(f"Generating 1000 sequences for class: {cls}")
        x, y = generate_class_sequences(cls, num_samples=1000, seq_len=12)
        all_x.append(x)
        all_y.append(y)
        
    final_x = torch.cat(all_x, dim=0)
    final_y = torch.cat(all_y, dim=0)
    
    logger.info(f"Total Training Data (raw): X shape {final_x.shape}, Y shape {final_y.shape}")
    
    # Normalize to [0, 1] to prevent gradient explosion
    final_x, final_y, mins, maxs = normalize_sequences(final_x, final_y)
    logger.info(f"Normalized. Feature mins range: [{mins.min():.2f}, {mins.max():.2f}], maxs range: [{maxs.min():.2f}, {maxs.max():.2f}]")
    
    dataset = TensorDataset(final_x, final_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = TimeSeriesLSTM(input_dim=14, hidden_dim=64, num_layers=2, output_dim=14).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    epochs = 20
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_loss = train_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.6f}")
        
    save_path = os.path.join(MODELS_DIR, "lstm_shared.pt")
    torch.save(model.state_dict(), save_path)
    
    # Save normalization parameters for inference
    norm_path = os.path.join(MODELS_DIR, "lstm_shared_norm.json")
    with open(norm_path, 'w') as f:
        json.dump({
            'mins': mins.tolist(),
            'maxs': maxs.tolist()
        }, f)
    
    logger.info(f"✅ LSTM Shared Model Saved to {save_path}")
    logger.info(f"✅ LSTM Norm Params Saved to {norm_path}")

import os
import json
import torch
import torch.optim as optim
import logging
from torch.utils.data import DataLoader, TensorDataset

# To use simulator code from backend root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.vae.model import DeviceVAE, vae_loss_function
from simulator.device_profiles import FLEET
from simulator.traffic_generator import generate_flow
from app.services.feature_extraction import extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = "models_trained/"
os.makedirs(MODELS_DIR, exist_ok=True)

# Try using the RTX 4060 if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_baseline_dataset(dev: dict, num_windows: int = 1000):
    """
    Accelerates 7 days of behavior. 
    1 window = 5 minutes = ~200 flows.
    """
    data_tensors = []
    
    for w in range(num_windows):
        flows = [generate_flow(dev) for _ in range(25)] # Roughly a 5 minute slice
        feature_vector = extract_features(dev['id'], dev['device_class'], flows)
        tensor_list = feature_vector.to_tensor_list()
        data_tensors.append(tensor_list)
        
    return torch.FloatTensor(data_tensors)

def normalize_dataset(data: torch.Tensor):
    """
    Min-Max normalize each feature column to [0, 1] range.
    Returns (normalized_data, mins, maxs) so we can save the scaling params.
    """
    mins, _ = data.min(dim=0)
    maxs, _ = data.max(dim=0)
    
    # Avoid division by zero for constant features
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    
    normalized = (data - mins) / ranges
    return normalized, mins, maxs

def train_device_twin(dev: dict, epochs: int = 50, batch_size: int = 64):
    """
    Train a specialized VAE network strictly on the specific normal baseline 
    dataset of a single endpoint IoT device.
    """
    logger.info(f"Generating synthetic baseline for {dev['name']} ({dev['id']})...")
    baseline_tensors = generate_baseline_dataset(dev)
    
    # Normalize to [0, 1] to prevent gradient explosion
    normalized_data, mins, maxs = normalize_dataset(baseline_tensors)
    
    dataset = TensorDataset(normalized_data, normalized_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DeviceVAE(input_dim=14).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_features, _ in dataloader:
            batch_features = batch_features.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(batch_features)
            loss, mse = vae_loss_function(recon_batch, batch_features, mu, logvar)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            train_loss += loss.item()
            optimizer.step()
        
        avg_loss = train_loss / len(dataloader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  [{dev['id']}] Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f}")
            
    # Save the PyTorch Model state dict
    save_path = os.path.join(MODELS_DIR, f"vae_{dev['id']}.pt")
    torch.save(model.state_dict(), save_path)
    
    # Save normalization parameters alongside the model for inference
    norm_path = os.path.join(MODELS_DIR, f"vae_{dev['id']}_norm.json")
    with open(norm_path, 'w') as f:
        json.dump({
            'mins': mins.tolist(),
            'maxs': maxs.tolist()
        }, f)
    
    final_loss = train_loss / len(dataloader.dataset)
    logger.info(f"Compiled Twin {dev['id']} -> {save_path} (Final Loss: {final_loss:.4f})")

if __name__ == "__main__":
    logger.info(f"Beginning VAE Digital Twin Training Engine. Target Device: {device}")
    
    # Train Digital Twins for all 50 simulated devices in our fleet
    try:
        for idx, dev in enumerate(FLEET):
            logger.info(f"Training Progress: Twin {idx + 1} / {len(FLEET)}")
            train_device_twin(dev, epochs=20) # Keep low for fast dev iteration
            
        logger.info("✅ All 50 Digital Twins Successfully Trained.")
    except KeyboardInterrupt:
        logger.warning("Training halted natively.")

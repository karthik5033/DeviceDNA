import os
import json
import torch
import torch.optim as optim
import logging
from torch.utils.data import DataLoader, TensorDataset

# Map simulator imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.gmvae.model import DeviceGMVAE, gmvae_loss_function
from simulator.device_profiles import FLEET
from simulator.traffic_generator import generate_flow
from app.services.feature_extraction import extract_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_trained")
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_TO_DEVICES = {}
for dev in FLEET:
    cls = dev['device_class']
    if cls not in CLASS_TO_DEVICES:
        CLASS_TO_DEVICES[cls] = []
    CLASS_TO_DEVICES[cls].append(dev)

def generate_class_baseline_dataset(class_name: str, num_windows_per_device: int = 150):
    """
    Generates synthetic baseline telemetry vectors strictly for one device class.
    """
    logger.info(f"Generating synthetic baseline for class '{class_name}' ({num_windows_per_device} windows per device)...")
    feature_list = []
    
    devices = CLASS_TO_DEVICES.get(class_name, [])
    for idx, dev in enumerate(devices):
        logger.info(f"  [{idx + 1}/{len(devices)}] Simulating flows for specialist {dev['id']}...")
        for w in range(num_windows_per_device):
            # Generate a 5-minute slice of traffic flows (~15 flows)
            flows = [generate_flow(dev) for _ in range(15)]
            feature_vector = extract_features(dev['id'], class_name, flows)
            feature_list.append(feature_vector.to_tensor_list())
            
    return torch.FloatTensor(feature_list)

def train_specialist(class_name: str, epochs: int = 25, batch_size: int = 64):
    """
    Trains a specialized single-component GMVAE model strictly on a single device class behavioral space.
    """
    features = generate_class_baseline_dataset(class_name)
    if len(features) == 0:
        logger.warning(f"No baseline data generated for class '{class_name}'. Skipping.")
        return
        
    # Min-Max Normalization
    mins, _ = features.min(dim=0)
    maxs, _ = features.max(dim=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized_features = (features - mins) / ranges
    
    dataset = TensorDataset(normalized_features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Specialist GMVAE has K=1 cluster representation (single-mode class representation)
    model = DeviceGMVAE(input_dim=14, num_clusters=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    logger.info(f"Optimizing Specialist GMVAE for class: {class_name}...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            # Enforce single class cluster (class_idx=0)
            recon_x, _, pi, mus, logvars, chosen_idx = model(batch_x, class_idx=0)
            
            loss, mse, kld, entropy = gmvae_loss_function(
                recon_x, batch_x, pi, mus, logvars, chosen_idx, entropy_beta=0.0
            )
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader.dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"  [{class_name}] Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
            
    # Save specialized weights
    model_path = os.path.join(MODELS_DIR, f"gmvae_specialist_{class_name}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"[OK] Specialist GMVAE for {class_name} saved to {model_path}\n")

if __name__ == "__main__":
    # Train specialists across all 6 classes
    for class_name in CLASS_TO_DEVICES.keys():
        train_specialist(class_name, epochs=20)

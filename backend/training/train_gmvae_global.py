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

MODELS_DIR = "models_trained/"
os.makedirs(MODELS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_MAP = {
    'camera': 0,
    'sensor': 1,
    'thermostat': 2,
    'access_control': 3,
    'medical': 4,
    'industrial': 5
}

def generate_global_baseline_dataset(num_windows_per_device: int = 80):
    """
    Generates synthetic baseline datasets across all 50 devices in the fleet,
    returning feature tensors and corresponding class index labels.
    """
    logger.info(f"Generating global synthetic baseline dataset ({num_windows_per_device} windows per device)...")
    feature_list = []
    class_labels = []
    
    for idx, dev in enumerate(FLEET):
        cls_name = dev['device_class']
        class_idx = CLASS_MAP.get(cls_name, 0)
        logger.info(f" [{idx + 1}/{len(FLEET)}] Simulating baseline flows for {dev['id']} ({cls_name})...")
        
        for w in range(num_windows_per_device):
            # Generate a 5-minute slice of traffic flows (~15 flows)
            flows = [generate_flow(dev) for _ in range(15)]
            feature_vector = extract_features(dev['id'], cls_name, flows)
            feature_list.append(feature_vector.to_tensor_list())
            class_labels.append(class_idx)
            
    return torch.FloatTensor(feature_list), torch.LongTensor(class_labels)

def train_global_gmvae(epochs: int = 30, batch_size: int = 64):
    """
    Trains the main Global GMVAE cluster routing and network autoencoder.
    """
    features, labels = generate_global_baseline_dataset()
    
    # Min-Max Normalization across the full global dataset
    mins, _ = features.min(dim=0)
    maxs, _ = features.max(dim=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normalized_features = (features - mins) / ranges
    
    dataset = TensorDataset(normalized_features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DeviceGMVAE(input_dim=14, num_clusters=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    logger.info("Beginning Global GMVAE model optimization...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_mse = 0
        epoch_kld = 0
        epoch_ent = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: enforce class routing during supervised pre-training
            recon_x, _, pi, mus, logvars, chosen_idx = model(batch_x, class_idx=batch_y)
            
            loss, mse, kld, entropy = gmvae_loss_function(
                recon_x, batch_x, pi, mus, logvars, chosen_idx, entropy_beta=0.15
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += mse.item()
            epoch_kld += kld.item()
            epoch_ent += entropy.item()
            
        n_samples = len(dataloader.dataset)
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/n_samples:.4f} | "
                    f"MSE: {epoch_mse/n_samples:.4f} | KLD: {epoch_kld/n_samples:.4f} | "
                    f"Entropy: {epoch_ent/n_samples:.4f}")
                    
    # Save Model state parameters
    model_path = os.path.join(MODELS_DIR, "gmvae_global.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Global GMVAE model saved to {model_path}")
    
    # Save Normalization params for live stream processing
    norm_path = os.path.join(MODELS_DIR, "gmvae_global_norm.json")
    with open(norm_path, 'w') as f:
        json.dump({
            'mins': mins.tolist(),
            'maxs': maxs.tolist()
        }, f)
    logger.info(f"Normalization constants saved to {norm_path}")

if __name__ == "__main__":
    train_global_gmvae(epochs=25)

import os
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

def train_device_twin(dev: dict, epochs: int = 50, batch_size: int = 64):
    """
    Train a specialized VAE network strictly on the specific normal baseline 
    dataset of a single endpoint IoT device.
    """
    logger.info(f"Generating synthetic baseline for {dev['name']} ({dev['id']})...")
    baseline_tensors = generate_baseline_dataset(dev)
    
    dataset = TensorDataset(baseline_tensors, baseline_tensors)
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
            train_loss += loss.item()
            optimizer.step()
            
    # Save the PyTorch Model state dict
    save_path = os.path.join(MODELS_DIR, f"vae_{dev['id']}.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Compiled Twin {dev['id']} -> {save_path} (Final Loss: {train_loss / len(dataloader.dataset):.4f})")

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

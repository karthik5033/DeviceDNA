import os
import json
import logging
import torch
import torch.nn.functional as F
import numpy as np
from app.ml.gmvae.model import DeviceGMVAE

logger = logging.getLogger(__name__)

# Map class names to GMM cluster indices
CLASS_TO_IDX = {
    'camera': 0,
    'sensor': 1,
    'thermostat': 2,
    'access_control': 3,
    'medical': 4,
    'industrial': 5
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

class GMVAEHierarchicalScorer:
    """
    Hierarchical GMVAE scoring engine implementing the 6-Signal Comparison Engine.
    Coordinates Global and Specialized models to catch stealthy behavioral drifts.
    """
    def __init__(self):
        self.models_dir = "models_trained/"
        self.global_model = None
        self.specialists = {}
        self.norm_params = {}
        
        # In-memory history for latent variables to compute temporal drift and centroids
        self.device_latent_history = {}
        self.device_centroids = {}
        
        # Load Global Model
        global_pt = os.path.join(self.models_dir, "gmvae_global.pt")
        global_json = os.path.join(self.models_dir, "gmvae_global_norm.json")
        
        if os.path.exists(global_pt) and os.path.exists(global_json):
            try:
                with open(global_json, 'r') as f:
                    self.global_norm = json.load(f)
                self.global_model = DeviceGMVAE(input_dim=14, num_clusters=6)
                self.global_model.load_state_dict(torch.load(global_pt, map_location=torch.device('cpu')))
                self.global_model.eval()
                logger.info("Loaded Global GMVAE model successfully.")
            except Exception as e:
                logger.error(f"Failed to load Global GMVAE: {e}")
                
        # Load Specialist Models
        for cls_name in CLASS_TO_IDX.keys():
            spec_pt = os.path.join(self.models_dir, f"gmvae_specialist_{cls_name}.pt")
            if os.path.exists(spec_pt):
                try:
                    spec_model = DeviceGMVAE(input_dim=14, num_clusters=1) # Specialist has 1 component representation
                    spec_model.load_state_dict(torch.load(spec_pt, map_location=torch.device('cpu')))
                    spec_model.eval()
                    self.specialists[cls_name] = spec_model
                    logger.info(f"Loaded Specialist GMVAE for class '{cls_name}'.")
                except Exception as e:
                    logger.error(f"Failed to load Specialist GMVAE for {cls_name}: {e}")

        # Load Device-level min-max normalizations for features
        self.load_device_norms()

    def load_device_norms(self):
        """Loads min-max norms for all 50 devices from models_trained directory."""
        for i in range(1, 51):
            device_id = f"SIM-{i:04d}"
            json_path = os.path.join(self.models_dir, f"vae_{device_id}_norm.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        self.norm_params[device_id] = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load normalization parameters for {device_id}: {e}")

    def score_deviation(self, device_id: str, device_class: str, feature_vector: list[float]) -> float:
        """
        Runs the 6-Signal Comparison Engine across Global and Specialist networks.
        Computes composite anomaly score [0.0 - 1.0].
        """
        # Graceful fallback: If models aren't loaded/trained, return 0.0 anomaly penalty
        if self.global_model is None:
            return 0.0
            
        # 1. Normalize features using the device's specific min/max bounds
        norm = self.norm_params.get(device_id)
        if not norm:
            logger.warning(f"No normalization parameters found for {device_id}. Using [0, 1] fallback.")
            norm = {'min': [0.0]*14, 'max': [1.0]*14}
            
        f_mins = norm.get('min', norm.get('mins', norm.get('feature_mins', [0]*14)))
        f_maxs = norm.get('max', norm.get('maxs', norm.get('feature_maxs', [1]*14)))
        
        normalized_features = []
        for val, f_min, f_max in zip(feature_vector, f_mins, f_maxs):
            if f_max - f_min == 0:
                normalized_features.append(0.0)
            else:
                normalized_features.append((val - f_min) / (f_max - f_min))
                
        tensor_x = torch.FloatTensor(normalized_features).unsqueeze(0)
        
        with torch.no_grad():
            # ---------------------------------------------
            # STEP 1: Global GMVAE Inference & Probabilities
            # ---------------------------------------------
            recon_g, z_g, pi_g, mus_g, logvars_g, chosen_idx_g = self.global_model(tensor_x)
            mse_g = F.mse_loss(recon_g, tensor_x, reduction='mean').item()
            
            # Global anomaly threshold (scale reconstruction MSE to 0.0 - 1.0)
            L_g = min(1.0, mse_g / 3.0)
            
            # Cluster probabilities & Entropy
            pi_arr = pi_g[0].numpy()
            max_prob = float(np.max(pi_arr))
            
            # Signal 3: Routing ambiguity (max probability should be high)
            routing_conf_anomaly = 1.0 - max_prob
            
            # Signal 4: Latent entropy (measures category confusion)
            eps = 1e-7
            entropy = float(-np.sum(pi_arr * np.log(pi_arr + eps)))
            scaled_entropy = min(1.0, entropy / 1.79) # Max entropy log(6) ≈ 1.79
            
            # ---------------------------------------------
            # STEP 2: Specialist GMVAE Inference (Signal 1)
            # ---------------------------------------------
            # Route to specialist based on active device class
            mse_l = mse_g
            specialist = self.specialists.get(device_class)
            if specialist is not None:
                recon_l, z_l, _, _, _, _ = specialist(tensor_x, class_idx=0)
                mse_l = F.mse_loss(recon_l, tensor_x, reduction='mean').item()
                
            L_l = min(1.0, mse_l / 3.0)
            
            # Reconstruction Difference (globally acceptable but abnormal for class)
            recon_diff = max(0.0, L_l - L_g)
            
            # ---------------------------------------------
            # STEP 3: Latent Identity & Centroid Drift (Signal 2)
            # ---------------------------------------------
            z_g_np = z_g[0].numpy()
            
            # Maintain rolling historical centroids for the device to measure drift
            if device_id not in self.device_latent_history:
                self.device_latent_history[device_id] = []
            
            self.device_latent_history[device_id].append(z_g_np)
            if len(self.device_latent_history[device_id]) > 50:
                self.device_latent_history[device_id].pop(0)
                
            # Centroid is the average normal vector observed during the baseline window
            if device_id not in self.device_centroids:
                # Use class mu head as the default starting centroid
                class_idx = CLASS_TO_IDX.get(device_class, 0)
                self.device_centroids[device_id] = mus_g[0, class_idx].numpy()
                
            centroid = self.device_centroids[device_id]
            D_z = float(np.linalg.norm(z_g_np - centroid))
            # Normalize D_z: observed normal distance < 1.0; anomalous > 3.0
            latent_drift_score = min(1.0, max(0.0, D_z / 4.0))
            
            # ---------------------------------------------
            # STEP 4: Temporal Latent Velocity (Signal 5)
            # ---------------------------------------------
            latent_velocity = 0.0
            if len(self.device_latent_history[device_id]) >= 2:
                prev_z = self.device_latent_history[device_id][-2]
                vel = float(np.linalg.norm(z_g_np - prev_z))
                latent_velocity = min(1.0, vel / 2.0) # normal velocity is very low
                
            # ---------------------------------------------
            # STEP 5: Aggregation & Multi-Signal Fusion
            # ---------------------------------------------
            # Weighted formula blending the hierarchical GMVAE metrics:
            # 30% Local Recon, 20% Global Recon, 20% Latent Drift, 15% Routing uncertainty, 15% Entropy
            composite_score = (
                (L_l * 0.30) +
                (L_g * 0.20) +
                (latent_drift_score * 0.20) +
                (routing_conf_anomaly * 0.15) +
                (scaled_entropy * 0.15)
            )
            
            # If temporal velocity spikes sharply, amplify the score by up to 20%
            if latent_velocity > 0.6:
                composite_score = min(1.0, composite_score + (latent_velocity * 0.20))
                
            # Log GMVAE Signals for forensic tracking
            with open('debug_gmvae.log', 'a') as f:
                f.write(f"GMVAE DEBUG: device={device_id} class={device_class} L_g={L_g:.4f} L_l={L_l:.4f} "
                        f"diff={recon_diff:.4f} drift={latent_drift_score:.4f} conf_anom={routing_conf_anomaly:.4f} "
                        f"entropy={scaled_entropy:.4f} vel={latent_velocity:.4f} final={composite_score:.4f}\n")
                        
            return float(composite_score)

    def score(self, device_id: str, feature_vector: list[float]) -> float:
        """Alias for simple VAE single-device compatibility. Routes to correct class."""
        # Retrieve class index from database or simulator fleet mapping
        # For simplicity, defaults to 'camera' or maps SIM-XXXX range
        try:
            sim_num = int(device_id.split('-')[1])
            if sim_num <= 12:
                device_class = 'camera'
            elif sim_num <= 22:
                device_class = 'sensor'
            elif sim_num <= 30:
                device_class = 'thermostat'
            elif sim_num <= 36:
                device_class = 'access_control'
            elif sim_num <= 44:
                device_class = 'medical'
            else:
                device_class = 'industrial'
        except Exception:
            device_class = 'camera'
            
        return self.score_deviation(device_id, device_class, feature_vector)

gmvae_scorer = GMVAEHierarchicalScorer()

import logging
from app.ml.vae.scoring import twin_scorer

try:
    import numpy as np
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    np = None
    torch = None
    cosine_similarity = None
logger = logging.getLogger(__name__)

class DNAFingerprintService:
    """
    Computes and compares high-dimensional DNA representations of device behavior.
    Uses the embedded latent space of the Digital Twin (VAE Autoencoder) 
    combined with static feature averages to form a static signature string.
    """
    def __init__(self):
        self._class_averages = {}

    def compute_dna_vector(self, device_id: str, current_features: list[float]) -> np.ndarray:
        """
        Pass the 14D raw feature vector through the VAE and compute the DNA signature.
        The DNA is the combination of the raw attributes + the latent vector (16D) 
        outputting a 30-Dimension behavior profile.
        """
        try:
            model = twin_scorer.load_twin(device_id)
        except Exception:
            return np.zeros((1, 30))

        with torch.no_grad():
            tensor_x = torch.FloatTensor(current_features)
            
            # Extract the latent representation (mu) which represents the "essence" of behavior
            mu, _ = model.encode(tensor_x)
            latent_vector = mu.numpy()

        # Concatenate original features with latent meaning
        # Normalize the raw features to match the latent space scale (~ -5.0 to 5.0) for pure comparison
        norm_features = (np.array(current_features) - np.mean(current_features)) / (np.std(current_features) + 1e-8)
        
        dna_signature = np.concatenate([norm_features, latent_vector]).reshape(1, -1)
        return dna_signature

    def verify_identity(self, device_id: str, new_dna: np.ndarray, baseline_dna: np.ndarray) -> float:
        """
        Compare the live running DNA trace to the officially enrolled DNA trace.
        Outputs a Cosine Similarity score from -1.0 to 1.0. 
        If it drops below 0.85, the device has likely been compromised with new firmware.
        """
        if baseline_dna.sum() == 0 or new_dna.sum() == 0:
            return 0.0
            
        similarity = cosine_similarity(new_dna, baseline_dna)[0][0]
        return float(similarity)

    def classify_unknown_device(self, unknown_dna: np.ndarray, known_classes: dict[str, np.ndarray]) -> str:
        """
        Predict the class of a newly attached, unknown IoT device by comparing its DNA 
        to the class-average DNA pool (Cameras vs Thermostats vs Monitors).
        Returns the class string with the highest similarity.
        """
        best_match = 'unknown'
        highest_sim = -1.0
        
        for dev_class, class_dna in known_classes.items():
            sim = cosine_similarity(unknown_dna, class_dna)[0][0]
            if sim > highest_sim:
                highest_sim = sim
                best_match = dev_class
                
        # If it doesn't match anything well, it's a completely unknown device
        if highest_sim < 0.65:
            return "unknown_alien_device"
            
        return best_match

dna_engine = DNAFingerprintService()

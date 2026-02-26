import torch
import torch.nn as nn
import torch.nn.functional as F

class DeviceVAE(nn.Module):
    """
    Variational Autoencoder used to build a 'Digital Twin' of an IoT device.
    Learns the 14-Dimensional behavioral distribution of a device's baseline telemetry.
    Input matches the 14 flat float list from FeatureVector.
    """
    def __init__(self, input_dim=14, hidden_dim=32, latent_dim=16):
        super(DeviceVAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    """
    Combines Reconstruction Error (MSE) and Kullback-Leibler Divergence (KLD).
    """
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Return total loss, and reconstruction error separately for scoring
    return MSE + KLD, MSE 

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeviceGMVAE(nn.Module):
    """
    Gaussian Mixture Variational Autoencoder (GMVAE).
    Implements a K-component Gaussian Mixture prior in the latent space.
    Used for both Global and Specialist behavioral modeling.
    """
    def __init__(self, input_dim=14, hidden_dim1=64, hidden_dim2=128, latent_dim=16, num_clusters=6):
        super(DeviceGMVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        
        # Shared Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        
        # Cluster Routing Head
        self.fc_logits = nn.Linear(hidden_dim2, num_clusters)
        
        # K specialized mean and log-variance heads for the mixture components
        self.mu_heads = nn.ModuleList([nn.Linear(hidden_dim2, latent_dim) for _ in range(num_clusters)])
        self.logvar_heads = nn.ModuleList([nn.Linear(hidden_dim2, latent_dim) for _ in range(num_clusters)])
        
        # Shared Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim1)
        self.fc4 = nn.Linear(hidden_dim1, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        
        logits = self.fc_logits(h2)
        pi = F.softmax(logits, dim=-1) # Cluster probabilities shape: [batch, K]
        
        mus = torch.stack([head(h2) for head in self.mu_heads], dim=1)        # [batch, K, latent_dim]
        logvars = torch.stack([head(h2) for head in self.logvar_heads], dim=1)  # [batch, K, latent_dim]
        
        return pi, mus, logvars

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x, class_idx=None):
        pi, mus, logvars = self.encode(x)
        
        # Determine which cluster to sample from:
        # If class_idx is provided (e.g. specialized training), route to that cluster.
        # Otherwise, route to the highest-probability cluster (argmax).
        if class_idx is not None:
            if isinstance(class_idx, int):
                chosen_idx = torch.full((x.size(0),), class_idx, dtype=torch.long, device=x.device)
            else:
                chosen_idx = class_idx.long()
        else:
            chosen_idx = torch.argmax(pi, dim=-1) # Shape: [batch]
            
        # Extract the chosen cluster's parameters
        batch_size = x.size(0)
        batch_indices = torch.arange(batch_size, device=x.device)
        
        chosen_mu = mus[batch_indices, chosen_idx]        # Shape: [batch, latent_dim]
        chosen_logvar = logvars[batch_indices, chosen_idx]  # Shape: [batch, latent_dim]
        
        z = self.reparameterize(chosen_mu, chosen_logvar)
        recon_x = self.decode(z)
        
        return recon_x, z, pi, mus, logvars, chosen_idx

def gmvae_loss_function(recon_x, x, pi, mus, logvars, chosen_idx, entropy_beta=0.1):
    """
    Computes GMVAE Loss: Reconstruction Loss (MSE) + KL Divergence + Cluster Entropy.
    """
    # 1. Reconstruction Loss (MSE)
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # 2. KL Divergence for the chosen cluster components
    batch_size = x.size(0)
    batch_indices = torch.arange(batch_size, device=x.device)
    
    chosen_mu = mus[batch_indices, chosen_idx]
    chosen_logvar = logvars[batch_indices, chosen_idx]
    
    KLD = -0.5 * torch.sum(1 + chosen_logvar - chosen_mu.pow(2) - chosen_logvar.exp())
    
    # 3. Cluster Entropy Regularization (to encourage confident cluster assignment)
    eps = 1e-7
    entropy = -torch.sum(pi * torch.log(pi + eps))
    
    total_loss = MSE + KLD + entropy_beta * entropy
    return total_loss, MSE, KLD, entropy

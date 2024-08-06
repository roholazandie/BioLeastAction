import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleCellVAE(nn.Module):

    def __init__(self, config):
        super(SingleCellVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.layer_sizes[0], config.layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(config.layer_sizes[1], config.layer_sizes[2]),
            nn.ReLU()
        )
        #mean of the Gaussian:
        self.fc_mu = nn.Linear(config.layer_sizes[2], config.layer_sizes[3])
        #log of the variance of the Gaussian:
        self.fc_logvar = nn.Linear(config.layer_sizes[2], config.layer_sizes[3])

        self.decoder = nn.Sequential(
            nn.Linear(config.layer_sizes[3], config.layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(config.layer_sizes[2], config.layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(config.layer_sizes[1], config.layer_sizes[0]),
            nn.Sigmoid()  # To keep the output in the range [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return F.relu(self.fc_mu(h)), F.relu(self.fc_logvar(h))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        #reparametrization trick
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    latent_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + latent_loss
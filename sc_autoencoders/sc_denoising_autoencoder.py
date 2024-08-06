import torch
import torch.nn as nn


class DenoisingAutoEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.layer_sizes[0], config.layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(config.layer_sizes[1], config.layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(config.layer_sizes[2], config.layer_sizes[3]),
            nn.LayerNorm(config.layer_sizes[3], eps=config.layer_norm_epsilon),
            nn.Dropout(config.embed_dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.layer_sizes[3], config.layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(config.layer_sizes[2], config.layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(config.layer_sizes[1], config.layer_sizes[0])
        )

    def forward(self, x):
        x_noisy = self.add_noise(x)  # Add noise to the input
        x_encoded = self.encoder(x_noisy)
        x_reconstructed = self.decoder(x_encoded)
        return x_reconstructed

    def encode(self, x):
        x_noisy = self.add_noise(x)  # Add noise to the input
        return self.encoder(x_noisy)

    def decode(self, x):
        return self.decoder(x)

    def add_noise(self, x, noise_factor=0.3):
        """
        Add random noise to the input data.
        
        Parameters:
        x (Tensor): Input tensor.
        noise_factor (float): The standard deviation of the noise.
        
        Returns:
        Tensor: Noisy input tensor.
        """
        noise = torch.randn_like(x) * noise_factor
        return x + noise

    def load_from_checkpoint(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))
        self.eval()
        return self
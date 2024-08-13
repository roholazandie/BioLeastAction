import torch
import torch.nn as nn


class SingleCellAutoEncoder(nn.Module):

    def __init__(self, config):
        super(SingleCellAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.layer_sizes[0], config.layer_sizes[1]),
            nn.BatchNorm1d(config.layer_sizes[1]),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(config.layer_sizes[1], config.layer_sizes[2]),
            nn.BatchNorm1d(config.layer_sizes[2]),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(config.layer_sizes[2], config.layer_sizes[3]),
            nn.BatchNorm1d(config.layer_sizes[3]),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(config.layer_sizes[3], config.layer_sizes[4]),  # Additional Layer
            nn.BatchNorm1d(config.layer_sizes[4]),  # Batch Normalization for Additional Layer
            nn.LayerNorm(config.layer_sizes[4], eps=config.layer_norm_epsilon),
            nn.Dropout(config.embed_dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.layer_sizes[4], config.layer_sizes[3]),  # Additional Layer
            nn.BatchNorm1d(config.layer_sizes[3]),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(config.layer_sizes[3], config.layer_sizes[2]),
            nn.BatchNorm1d(config.layer_sizes[2]),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(config.layer_sizes[2], config.layer_sizes[1]),
            nn.BatchNorm1d(config.layer_sizes[1]),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(config.layer_sizes[1], config.layer_sizes[0]),
            nn.Softmax(dim=1)  # Softmax Activation
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def load_from_checkpoint(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))
        self.eval()
        return self

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Linear(out_channels, out_channels)
        
        # Shortcut connection (identity mapping)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class SingleCellResidualAutoEncoder(nn.Module):
    def __init__(self, config):
        super(SingleCellResidualAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            ResidualBlock(config.layer_sizes[0], config.layer_sizes[1]),
            ResidualBlock(config.layer_sizes[1], config.layer_sizes[2]),
            ResidualBlock(config.layer_sizes[2], config.layer_sizes[3]),
            nn.LayerNorm(config.layer_sizes[3], eps=config.layer_norm_epsilon),
            nn.Dropout(config.embed_dropout)
        )
        self.decoder = nn.Sequential(
            ResidualBlock(config.layer_sizes[3], config.layer_sizes[2]),
            ResidualBlock(config.layer_sizes[2], config.layer_sizes[1]),
            ResidualBlock(config.layer_sizes[1], config.layer_sizes[0])
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
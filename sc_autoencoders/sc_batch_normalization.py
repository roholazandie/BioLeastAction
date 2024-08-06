import torch
import torch.nn as nn

#currently using batch_norm with elu and alpha=0.1
class SingleCellBatchAutoEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.layer_sizes[0], config.layer_sizes[1]),
            nn.BatchNorm1d(config.layer_sizes[1]), 
            nn.ELU(alpha=0.1),
            nn.Linear(config.layer_sizes[1], config.layer_sizes[2]),
            nn.BatchNorm1d(config.layer_sizes[2]),
            nn.ELU(alpha=0.1),
            nn.Linear(config.layer_sizes[2], config.layer_sizes[3]),
            nn.BatchNorm1d(config.layer_sizes[3]),
            nn.Dropout(config.embed_dropout)
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.layer_sizes[3], config.layer_sizes[2]),
            nn.BatchNorm1d(config.layer_sizes[2]),
            nn.ELU(alpha=0.1),
            nn.Linear(config.layer_sizes[2], config.layer_sizes[1]),
            nn.BatchNorm1d(config.layer_sizes[1]),
            nn.ELU(alpha=0.1),
            nn.Linear(config.layer_sizes[1], config.layer_sizes[0]),
            nn.BatchNorm1d(config.layer_sizes[0])
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

from __future__ import print_function
import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.nn import functional as F
from tqdm import tqdm
from normflows.flows import Planar, Radial, MaskedAffineFlow, BatchNorm
import argparse
from datetime import datetime
import os
from normflows import nets
import pandas as pd
import random
import anndata
import wandb
from torchdiffeq import odeint_adjoint as odeint
from scipy.sparse import issparse
import numpy as np
import scanpy as sc
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Initialize wandb
wandb.init(project="Conditional_VAE_with_Flows")
save_dir = "Checkpoints/conditional_VAE_Checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class SingleCellDataset(Dataset):
    def __init__(self, adata):
        if issparse(adata.X):
            self.data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            self.data = torch.tensor(adata.X, dtype=torch.float32)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
adata = anndata.read_h5ad('../data/reprogramming_schiebinger_serum_computed.h5ad')    
cell_sets = adata.obs['cell_sets'].to_numpy()
# One-hot encode the cell_sets
encoder = OneHotEncoder(sparse_output=False)
condition_tensor = torch.tensor(encoder.fit_transform(cell_sets.reshape(-1, 1)), dtype=torch.float32)

dataset = SingleCellDataset(adata)
combined_dataset = TensorDataset(dataset.data, condition_tensor)

train_size = int(0.9 * len(dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class SimpleFlowModel(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        ld = 0.0
        for flow in self.flows:
            z, ld_ = flow(z)
            ld += ld_
        return z, ld

class BinaryTransform:
    def __init__(self, thresh=0.5):
        self.thresh = thresh

    def __call__(self, x):
        return (x > self.thresh).type(x.type())

class ColourNormalize:
    def __init__(self, a=0.0, b=0.0):
        self.a = a
        self.b = b

    def __call__(self, x):
        return (self.b - self.a) * x / 255 + self.a

input_dim = adata.shape[1]
latent_dim = 128

class FlowVAE(nn.Module):
    def __init__(self, flows, condition_dim):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.f1 = nn.Linear(256, 128)
        self.f2 = nn.Linear(256, 128)
        self.decode = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
            
        )
        self.flows = flows

    def forward(self, x,condition):
        x_cond = torch.cat([x, condition], dim=1)
        # Encode
        mu, log_var = self.f1(
            self.encode(x_cond)
        ), self.f2(self.encode(x_cond))

        # Reparameterize variables
        std = torch.exp(0.5 * log_var)
        norm_scale = torch.randn_like(std)
        z_0 = mu + norm_scale * std

        # Flow transforms
        z_, log_det = self.flows(z_0)
        z_ = z_.squeeze()

        # Q0 and prior
        q0 = Normal(mu, torch.exp((0.5 * log_var)))
        p = Normal(0.0, 1.0)

        # KLD including logdet term
        kld = (
            -torch.sum(p.log_prob(z_), -1)
            + torch.sum(q0.log_prob(z_0), -1)
            - log_det.view(-1)
        )
        self.test_params = [
            torch.mean(-torch.sum(p.log_prob(z_), -1)),
            torch.mean(torch.sum(q0.log_prob(z_0), -1)),
            torch.mean(log_det.view(-1)),
            torch.mean(kld),
        ]

        # Decode
        z_cond = torch.cat([z_.view(z_.size(0), 128), condition], dim=1)
        zD = self.decode(z_)
        out = torch.sigmoid(zD)

        return out, kld

def logit(x):
    return torch.log(x / (1 - x))

def bound(rce, x, kld, beta):
    mse_loss = nn.MSELoss() 
    recon_loss = mse_loss(rce, x)
    return recon_loss + beta*kld

condition_dim = condition_tensor.shape[1] 
flows = SimpleFlowModel([Planar((latent_dim,)) for k in range(10)])
model = FlowVAE(flows, condition_dim).to(device) 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, condition in train_loader:
        x, condition = x.to(device), condition.to(device)
        optimizer.zero_grad()
        out, kld = model(x, condition)
        loss = bound(out, x, kld, beta=1.0)  
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss}')
    
    # Evaluate on test set
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, condition in test_loader:
            x, condition = x.to(device), condition.to(device)
            out, kld = model(x, condition)
            loss = bound(out, x, kld, beta=1.0)
            loss = loss.mean()
            test_loss += loss.item()
    
    test_loss /= len(test_loader.dataset)
    print(f'Epoch {epoch+1}, Test Loss: {test_loss}')
    # Log the loss to wandb
    wandb.log({"Epoch": epoch+1, "Loss": avg_loss, "Test Loss": test_loss})
    model_path = os.path.join(save_dir, f"vae_with_flows_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)

# Finalize wandb run
wandb.finish() 

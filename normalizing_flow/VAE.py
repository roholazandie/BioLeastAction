import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import anndata
import wandb
from torchdiffeq import odeint_adjoint as odeint
from scipy.sparse import issparse
import os
import random
import numpy as np
import scanpy as sc

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Initialize wandb
wandb.init(project="VAE_with_Flows")

save_dir = "Checkpoints/norm_sc_VAE_Checkpoints"
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Step 1: Data Loading
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

#adata = anndata.read_h5ad('../data/E9.5_E1S1.MOSTA.h5ad')
adata = anndata.read_h5ad('../data/reprogramming_schiebinger_serum_computed.h5ad')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
dataset = SingleCellDataset(adata)
# Split into train and test sets (e.g., 90% train, 10% test)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Step 2: VAE with Normalizing Flows
class ODEFunc(nn.Module):
    def forward(self, t, z):
        return -0.5 * z

class VAE_model(nn.Module):
    def __init__(self, input_dim, latent_dim, flow_steps=4):
        super(VAE_model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),            
            nn.Linear(1024, 256),
            nn.ReLU(),        
            nn.Linear(256, latent_dim * 2)  # For mean and log variance
        )
        
        self.flow_steps = flow_steps
        self.latent_dim = latent_dim
        self.odefunc = ODEFunc()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, input_dim)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encoding
        q_params = self.encoder(x)
        mu, logvar = torch.chunk(q_params, 2, dim=-1)
        z0 = self.reparameterize(mu, logvar)
        
        # Normalizing Flow
        zk = z0
        for _ in range(self.flow_steps):
            zk = self.flow_layer(zk)
        
        # Decoding
        recon_x = self.decoder(zk)
        return recon_x, mu, logvar, zk
    
    def flow_layer(self, z):
        z = odeint(self.odefunc, z, torch.tensor([0., 1.]), method='rk4')[1]
        return z

# Instantiate the model
input_dim = adata.shape[1]
latent_dim = 128
model = VAE_model(input_dim, latent_dim)
model.to(device)

# Step 3: Training Loop
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

def loss_function(recon_x, x, mu, logvar, zk):
    # Reconstruction Loss
    recon_loss = loss_fn(recon_x, x)
    
    # KL Divergence with normalizing flow correction
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Add normalizing flow corrections to KL divergence if needed
    
    return recon_loss + kld

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            recon_x, mu, logvar, zk = model(batch)
            loss = loss_function(recon_x, batch, mu, logvar, zk)
            total_loss += loss.item()
    return total_loss / len(dataloader)

for epoch in range(50):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device) 
        recon_x, mu, logvar, zk = model(batch)
        loss = loss_function(recon_x, batch, mu, logvar, zk)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Loss: {avg_loss}')
    
    # Evaluate on test set
    test_loss = evaluate(model, test_loader)
    print(f'Epoch {epoch+1}, Test Loss: {test_loss}')
    
    # Log the loss to wandb
    wandb.log({"Epoch": epoch+1, "Loss": avg_loss, "Test Loss": test_loss})
    model_path = os.path.join(save_dir, f"vae_with_flows_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)

# Finalize wandb run
wandb.finish() 

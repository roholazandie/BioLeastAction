import torch
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.nn import functional as F
from torchdiffeq import odeint_adjoint as odeint

from normflows.flows import Planar, Radial, MaskedAffineFlow, BatchNorm
from normflows import nets

import random
import anndata
import wandb
from scipy.sparse import issparse
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#Initialize wandb
wandb.init(project="Conditional_VAE_with_Planar_Flows")
save_dir = "Checkpoints/planar_c_vae"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Turn anndata from a dense/sparse matrix to a PyTorch tensor. 
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

#Conditioning on cell_sets. Using interval [0,1] and splitting it up by ['Epithelial' 'IPS' 'MEF/other' 'MET' 'Neural' 'Stromal' 'Trophoblast'] in order. 
cell_sets = adata.obs['cell_sets'].to_numpy()
unique_cell_sets = np.unique(cell_sets)
interval_width = 1.0 / len(unique_cell_sets)
cell_set_mapping = {cell_set: (i + 0.5) * interval_width for i, cell_set in enumerate(unique_cell_sets)}
mapped_cell_sets = np.array([cell_set_mapping[cell] for cell in cell_sets])
condition_tensor = torch.tensor(mapped_cell_sets, dtype=torch.float32).unsqueeze(1)

# Create dataset and dataloader
dataset = SingleCellDataset(adata)
combined_dataset = TensorDataset(dataset.data, condition_tensor)

train_size = int(0.9 * len(dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#Flow model:
class FlowModel(nn.Module):
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, z):
        log_det = 0.0
        for flow in self.flows:
            z, ld = flow(z)
            log_det += ld
        return z, log_det

class FlowVAE(nn.Module):
    def __init__(self, flows, condition_dim):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.f1 = nn.Linear(256, 128)
        self.f2 = nn.Linear(256, 128)
        self.decode = nn.Sequential(
            nn.Linear(128 + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
            
        )
        self.flows = flows

    def forward(self, x,condition):
        # Encode
        encoded = self.encode(x)
        mu, log_var = self.f1(encoded), self.f2(encoded)

        # Reparameterize
        std = torch.exp(0.5 * log_var)
        z_0 = mu + torch.randn_like(std) * std

        # Flow transforms
        z_, log_det = self.flows(z_0)

        # Apply condition right before decoding
        z_cond = torch.cat([z_, condition], dim=1)
            
        # Decode with condition
        decoded = self.decode(z_cond)

        # Prior and KLD calculations
        q0 = Normal(mu, torch.exp(0.5 * log_var))
        p = Normal(0.0, 1.0)
        kld = (
            -torch.sum(p.log_prob(z_), -1)
            + torch.sum(q0.log_prob(z_0), -1)
            - log_det.view(-1)
        )
        return decoded, kld
    
def recon_loss_calculation(rce, x):
    mse_loss = nn.MSELoss() 
    recon_loss = mse_loss(rce, x)
    return recon_loss

input_dim = adata.shape[1]
latent_dim = 128
condition_dim = condition_tensor.shape[1] 
flows = FlowModel([Planar((latent_dim,)) for _ in range(10)])
model = FlowVAE(flows, condition_dim).to(device) 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss, total_recon_loss, total_kld = 0, 0, 0

    for x, condition in train_loader:
        x, condition = x.to(device), condition.to(device)
        optimizer.zero_grad()
        out, kld = model(x, condition)
        recon_loss = recon_loss_calculation(out,x)
        loss = recon_loss + kld.mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kld += kld.mean().item()

    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_kld = total_kld / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}, Recon Loss: {avg_recon_loss}, KLD: {avg_kld}')
    
    # Evaluate on test set
    model.eval()
    test_loss, test_recon_loss, test_kld = 0, 0, 0
    total_precision, total_recall, total_f1 = 0, 0, 0
    with torch.no_grad():
        for x, condition in test_loader:
            x, condition = x.to(device), condition.to(device)
            out, kld = model(x, condition)
            recon_loss = recon_loss_calculation(out, x)
            loss = recon_loss + kld.mean()
            test_loss += loss.item()
            test_recon_loss += recon_loss.item()
            test_kld += kld.mean().item()

            x_single = x[0:1].to(device)  # Take the first sample and maintain the batch dimension
            condition_single = condition[0:1].to(device)
            out_single, kld_single = model(x_single, condition_single)

            y_pred = (out_single != 0).cpu().flatten().tolist()
            y_true = (x_single != 0).cpu().flatten().tolist()

            
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')

            total_precision += precision
            total_recall += recall
            total_f1 += f1
    
    avg_test_loss = test_loss / len(test_loader)
    avg_test_recon_loss = test_recon_loss / len(test_loader)
    avg_test_kld = test_kld / len(test_loader)
    avg_precision = total_precision / len(test_loader)
    avg_recall = total_recall / len(test_loader)
    avg_f1 = total_f1 / len(test_loader)
    print(f'Epoch {epoch + 1}, Test Loss: {test_loss}, Test Recon Loss: {avg_test_recon_loss}, Test KLD: {avg_test_kld}')
    print(f'Epoch {epoch + 1}, Precision: {avg_precision}, Recall: {avg_recall}, F1: {avg_f1}')
    # Log the loss to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "train_recon_loss": avg_recon_loss,
        "train_kld": avg_kld,
        "test_loss": avg_test_loss,
        "test_recon_loss": avg_test_recon_loss,
        "test_kld": avg_test_kld,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1
    })
    model_path = os.path.join(save_dir, f"vae_with_flows_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)

# Finalize wandb run
wandb.finish()
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import scanpy as sc
import scipy.sparse as sp

import torch.nn.functional as F


import wandb
import argparse
import os
from sc_variational_autoencoder import SingleCellVAE, vae_loss
from sc_denoising_autoencoder import DenoisingAutoEncoder
from sc_residual_connections import ResidualBlock, SingleCellResidualAutoEncoder
from sc_batch_normalization import SingleCellBatchAutoEncoder
from sc_vanilla_autoencoder import SingleCellAutoEncoder
from sc_VQ_VAE import VQVAE, vqvae_loss

@dataclass
class Config:
    layer_sizes: List[int]
    # n_embd: int
    layer_norm_epsilon: float
    embed_dropout: float
    learning_rate: float
    batch_size: int
    num_epochs: int
    num_embeddings: int = 512
    commitment_cost: float = 0.25

def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def train_autoencoder(model, train_loader, test_loader, criterion, optimizer, num_epochs, checkpoint_interval, checkpoint_path, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Log the training loss to wandb
        wandb.log({"epoch": epoch + 1, "training_loss": epoch_loss})

        # Evaluate on the test set
        test_loss = evaluate_autoencoder(model, test_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}')
        wandb.log({"epoch": epoch + 1, "test_loss": test_loss})

        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path_full = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path_full)
            wandb.save(checkpoint_path_full)
            print(f'Checkpoint saved at epoch {epoch + 1}')


def evaluate_autoencoder(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss

def train_vae(model, train_loader, test_loader, criterion, optimizer, num_epochs, checkpoint_interval, checkpoint_path, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs, mu, logvar = model(inputs)
            loss = criterion(outputs, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            #print("reconstruction", F.mse_loss(outputs, inputs, reduction='sum').item())
            #print("latent loss", loss.item()-F.mse_loss(outputs, inputs, reduction='sum').item())
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Log the training loss to wandb
        wandb.log({"epoch": epoch + 1, "training_loss": epoch_loss})

        # Evaluate on the test set
        test_loss = evaluate_vae(model, test_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}')
        wandb.log({"epoch": epoch + 1, "test_loss": test_loss})

        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path_full = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path_full)
            wandb.save(checkpoint_path_full)
            print(f'Checkpoint saved at epoch {epoch + 1}')

def evaluate_vae(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            outputs, mu, logvar = model(inputs)
            loss = criterion(outputs, inputs, mu, logvar)
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss

def train_vqvae(model, train_loader, test_loader, criterion, optimizer, num_epochs, checkpoint_interval, checkpoint_path, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_reconstruction_loss = 0.0
        running_quantization_loss = 0.0
        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs, quantization_loss = model(inputs)
            reconstruction_loss = criterion(outputs, inputs) # outputs[0, :].detach().cpu().numpy()

            loss = reconstruction_loss + quantization_loss
            loss.backward()
            optimizer.step()

            # Accumulate losses for the epoch
            running_loss += loss.item() * inputs.size(0)
            running_reconstruction_loss += reconstruction_loss.item() * inputs.size(0)
            running_quantization_loss += quantization_loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_reconstruction_loss = running_reconstruction_loss / len(train_loader.dataset)
        epoch_quantization_loss = running_quantization_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Log the training loss to wandb
        wandb.log({"epoch": epoch + 1, "training_loss": epoch_loss})
        wandb.log({"epoch": epoch + 1, "reconstruction_loss": epoch_reconstruction_loss})
        wandb.log({"epoch": epoch + 1, "quantization_loss": epoch_quantization_loss})

        if (epoch + 1) % 10 == 0:
            # Evaluate on the test set
            test_loss, test_reconstruction_loss, test_quantization_loss = evaluate_vqvae(model, test_loader, criterion, device)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}')
            wandb.log({"epoch": epoch + 1, "test_loss": test_loss})
            wandb.log({"epoch": epoch + 1, "test_reconstruction_loss": test_reconstruction_loss})
            wandb.log({"epoch": epoch + 1, "test_quantization_loss": test_quantization_loss})

        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path_full = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path_full)
            print(f'Checkpoint saved at epoch {epoch + 1}')

def evaluate_vqvae(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_reconstruction_loss = 0.0
    running_quantization_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            outputs, quantization_loss = model(inputs)
            reconstruction_loss = criterion(outputs, inputs)
            loss = reconstruction_loss + quantization_loss
            running_loss += loss.item() * inputs.size(0)
            running_reconstruction_loss += reconstruction_loss.item() * inputs.size(0)
            running_quantization_loss += quantization_loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_reconstruction_loss = running_reconstruction_loss / len(data_loader.dataset)
    epoch_quantization_loss = running_quantization_loss / len(data_loader.dataset)
    return epoch_loss, epoch_reconstruction_loss, epoch_quantization_loss

def main(args):
    # Initialize wandb
    wandb.init(project=args.project_name, name=args.run_name)

    # set the random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    adata = sc.read_h5ad(args.dataset_path)
    sc.pp.highly_variable_genes(adata, n_top_genes=1024)
    # Subset the data to include only the highly variable genes
    adata = adata[:, adata.var['highly_variable']].copy()

    sc.pp.scale(adata, max_value=10)

    X_min_cells = np.min(adata.X, axis=1, keepdims=True)
    X_max_cells = np.max(adata.X, axis=1, keepdims=True)

    # Normalize each cell’s expression to be between 0 and 1
    X_normalized_cells = (adata.X - X_min_cells) / np.maximum((X_max_cells - X_min_cells), 1e-8)

    # Step 3: Update the AnnData object with the normalized data
    adata.X = X_normalized_cells

    input_size = adata.n_vars

    config = Config(layer_sizes=[input_size, args.layer_size1, args.layer_size2, args.layer_size3],
                    layer_norm_epsilon=args.layer_norm_epsilon,
                    embed_dropout=args.embed_dropout,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    num_epochs=args.num_epochs,
                    num_embeddings=args.num_embeddings,
                    commitment_cost=args.commitment_cost)

    # Log the config to wandb
    wandb.config.update(config)

    models = {"vanilla": SingleCellAutoEncoder,
              "VAE": SingleCellVAE,
              "batch_normalization": SingleCellBatchAutoEncoder,
              "residual_connections":SingleCellResidualAutoEncoder,
              "denoising":DenoisingAutoEncoder,
              "VQ_VAE": VQVAE}

    model = models[args.model_type](config)
    

    criterion = nn.MSELoss()
    if args.model_type=="VAE":
        criterion = vae_loss
        model.apply(initialize_weights)
    elif args.model_type == "VQ_VAE":
        criterion = vqvae_loss
        
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Ensure the data is in the correct format
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    tensor_data = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)

    # Shuffle dataset before splitting
    indices = torch.randperm(len(dataset))
    shuffled_dataset = torch.utils.data.Subset(dataset, indices)

    # Split dataset into training and testing sets (90/10 split)
    train_size = int(0.9 * len(shuffled_dataset))
    test_size = len(shuffled_dataset) - train_size
    train_dataset, test_dataset = random_split(shuffled_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    if args.model_type=="VAE":
        train_vae(model,
                      train_loader,
                      test_loader,
                      criterion,
                      optimizer,
                      config.num_epochs,
                      args.checkpoint_interval,
                      args.checkpoint_path,
                      'cuda:0')
    elif args.model_type =="VQ_VAE":
        train_vqvae(model,
                      train_loader,
                      test_loader,
                      criterion,
                      optimizer,
                      config.num_epochs,
                      args.checkpoint_interval,
                      args.checkpoint_path,
                      'cuda:0')
    else:
        train_autoencoder(model,
                      train_loader,
                      test_loader,
                      criterion,
                      optimizer,
                      config.num_epochs,
                      args.checkpoint_interval,
                      args.checkpoint_path,
                      'cuda:0')

if __name__ == "__main__":
    # "/home/rohola/codes/cellrank_playground/reprogramming_schiebinger_serum_computed.h5ad"
    # "/home/rohola/codes/BioLeastAction/checkpoints/autoencoders_checkpoints/vanilla_autoencoder/"

    parser = argparse.ArgumentParser(description='Train a single-cell autoencoder.')
    parser.add_argument('--project_name', type=str, required=True, help='Name of project in wandb')
    parser.add_argument('--run_name', type=str, required=True, help='Name of run under project_name in wandb')
    parser.add_argument('--model_type', type=str, required=True, help='Name of model type. Choose from vanilla, VAE, batch_normalization, residual_connections, denoising, VQ_VAE')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to save the checkpoints.')
    parser.add_argument('--layer_size1', type=int, default=512, help='Size of the first hidden layer.')
    parser.add_argument('--layer_size2', type=int, default=256, help='Size of the second hidden layer.')
    parser.add_argument('--layer_size3', type=int, default=128, help='Size of the second hidden layer.')
    parser.add_argument('--layer_norm_epsilon', type=float, default=1e-5, help='Epsilon for layer norm.')
    parser.add_argument('--embed_dropout', type=float, default=0.1, help='Dropout rate for embeddings.')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=1000000, help='Number of epochs to train.')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Interval for saving checkpoints.')
    parser.add_argument('--num_embeddings', type=int, default=512, help='Number of embeddings for VQ-VAE')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='Commitment cost for VQ-VAE')


    args = parser.parse_args()
    main(args)

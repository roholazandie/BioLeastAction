from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sc_vanilla_autoencoder import SingleCellAutoEncoder
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import scanpy as sc
import wandb
import argparse
import os


@dataclass
class Config:
    layer_sizes: List[int]
    # n_embd: int
    layer_norm_epsilon: float
    embed_dropout: float
    learning_rate: float
    batch_size: int
    num_epochs: int



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train_autoencoder(model,
                      train_loader,
                      test_loader,
                      criterion,
                      optimizer,
                      num_epochs,
                      checkpoint_interval,
                      checkpoint_path,
                      device):
    model.to(device)

    # Initialize the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=4, factor=0.5)

    # Set the number of warmup steps
    num_warmup_steps = int(0.1 * num_epochs)  # 10% of total epochs as warmup
    warmup_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_epochs)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.L1Loss(reduction='none')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        sample_losses = []
        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            running_loss += loss.item() * inputs.size(0)

            # Calculate loss per sample
            # with torch.no_grad():
            #     per_sample_loss = criterion1(outputs, inputs)
            #     sample_losses.append(per_sample_loss)

        # sample_losses = torch.cat(sample_losses, dim=0).cpu().numpy()
        # print(f'Loss per sample: {sample_losses}')


        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Log the training loss to wandb
        wandb.log({"epoch": epoch + 1, "training_loss": epoch_loss})

        # Apply warmup scheduler
        warmup_scheduler.step()

        # Evaluate on the test set
        test_loss = evaluate_autoencoder(model, test_loader, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}')
        wandb.log({"epoch": epoch + 1, "test_loss": test_loss})

        scheduler.step(test_loss)

        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path_full = os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path_full)
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

def scale_data_collator(data):
    #todo also scale based on the min and max of all the data
    """
    Scales the input data to the range [0, 1] assuming the original range is [0, 10].
    """
    return data / 10.0


def main(args):
    # Initialize wandb
    wandb.init(project="single-cell-autoencoder")

    # set the random seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    adata = sc.read_h5ad(args.dataset_path)

    input_size = adata.n_vars

    config = Config(layer_sizes=[input_size, args.layer_size1, args.layer_size2, args.layer_size3, args.layer_size4],
                    layer_norm_epsilon=args.layer_norm_epsilon,
                    embed_dropout=args.embed_dropout,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    num_epochs=args.num_epochs)

    # Log the config to wandb
    wandb.config.update(config)

    model = SingleCellAutoEncoder(config)

    # criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.L1Loss()
    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


    # Ensure the data is in the correct format
    tensor_data = torch.tensor(adata.X.toarray(), dtype=torch.float32)
    # scaled_data = scale_data_collator(tensor_data)  # Scale the data
    dataset = TensorDataset(tensor_data)

    # Shuffle dataset before splitting
    indices = torch.randperm(len(dataset))
    shuffled_dataset = torch.utils.data.Subset(dataset, indices)

    # Split dataset into training and testing sets (80/20 split)
    train_size = int(0.9 * len(shuffled_dataset))
    test_size = len(shuffled_dataset) - train_size
    train_dataset, test_dataset = random_split(shuffled_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

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
    # /home/rohola/codes/BioLeastAction/data/reprogramming_schiebinger_serum_computed.h5ad
    # "/home/rohola/codes/BioLeastAction/checkpoints/autoencoders_checkpoints/vanilla_autoencoder/"

    parser = argparse.ArgumentParser(description='Train a single-cell autoencoder.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to save the checkpoints.')
    parser.add_argument('--layer_size1', type=int, default=8192, help='Size of the first hidden layer.')
    parser.add_argument('--layer_size2', type=int, default=4096, help='Size of the first hidden layer.')
    parser.add_argument('--layer_size3', type=int, default=2048, help='Size of the second hidden layer.')
    parser.add_argument('--layer_size4', type=int, default=768, help='Size of the second hidden layer.')
    parser.add_argument('--layer_norm_epsilon', type=float, default=1e-5, help='Epsilon for layer norm.')
    parser.add_argument('--embed_dropout', type=float, default=0.1, help='Dropout rate for embeddings.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    # parser.add_argument('--batch_size', type=int, default=98304, help='Batch size for training.')
    parser.add_argument('--batch_size', type=int, default=32768, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--checkpoint_interval', type=int, default=5, help='Interval for saving checkpoints.')

    args = parser.parse_args()

    main(args)

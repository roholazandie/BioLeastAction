import numpy as np
import scanpy as sc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import kl_divergence as kl
import scipy.sparse as sp
import os
from tqdm import tqdm, tqdm_notebook, trange
from torch.distributions import Normal
from torch.utils.data import DataLoader
from scvi.distributions import ZeroInflatedNegativeBinomial
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd


class scVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, additional_features_dim, distribution='zinb'):
        super(scVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim+additional_features_dim, hidden_dim, bias=True),
                                     nn.BatchNorm1d(hidden_dim, eps=0.001),
                                     nn.ReLU(),
                                     nn.Dropout(0.1)
                                     )

        self.mu_encoder = nn.Linear(hidden_dim, latent_dim)
        self.var_encoder = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(nn.Linear(latent_dim+additional_features_dim, hidden_dim),
                                     nn.BatchNorm1d(hidden_dim, eps=0.001),
                                     nn.ReLU())
        self.mu_decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim, bias=True),
                                        nn.Softmax(dim=-1),
                                        )
        self.dropout_decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        self.distribution = distribution
        self.log_theta = torch.nn.Parameter(torch.randn(input_dim))

    def encode(self, x, additional_features):
        x = x.float()
        additional_features = additional_features.float()
        h1 = self.encoder(torch.cat([x, additional_features], dim=-1))
        q_mu = self.mu_encoder(h1)
        q_var = torch.exp(self.var_encoder(h1)) + 1e-4
        q_z = Normal(q_mu, q_var.sqrt())
        z = q_z.rsample()
        return q_z, z

    def decode(self, z, library,additional_features):
        z = z.float()
        additional_features = additional_features.float()
        h = self.decoder(torch.cat([z, additional_features], dim=-1)) # with softplus
        mu = self.mu_decoder(h)
        dropout_logits = self.dropout_decoder(h) # this should stay as it is
        x_rate = torch.exp(library) * mu
        return torch.exp(mu), dropout_logits, x_rate

    def forward(self, x,additional_features):
        library = torch.log(x.sum(1)).unsqueeze(1)
        x_log = torch.log(x + 1)
        q_z, z = self.encode(x_log,additional_features)
        mu, dropout_logits, x_rate = self.decode(z, library,additional_features)
        return mu, dropout_logits, x_rate, q_z, z

    def get_latent_representation(self, x):
        x_log = torch.log(x + 1)
        q_z, z = self.encode(x_log)
        return z

    def generate(self, x,additional_features):
        self.eval()
        with torch.no_grad():
            mu, dropout_logits, x_rate, q_z, z = self(x, additional_features)

            theta = self.log_theta.exp()
            if self.distribution == 'zinb':
                p_x = ZeroInflatedNegativeBinomial(
                    mu=x_rate,
                    theta=theta,
                    zi_logits=dropout_logits,
                    scale=mu,
                )
            return p_x.sample()


    def loss_function(self, x, mu, dropout_logits, x_rate, q_z, z):
        theta = self.log_theta.exp()
        if self.distribution == 'zinb':
            p_x = ZeroInflatedNegativeBinomial(
                mu=x_rate,
                theta=theta,
                zi_logits=dropout_logits,
                scale=mu,
                validate_args=False
            )

        reconstruction_loss = -p_x.log_prob(x).sum(-1)

        # kl divergence between q(z|x) and p(z)
        p_z = Normal(torch.zeros_like(z), torch.ones_like(z), validate_args=False)
        kl_z = kl(q_z, p_z).sum(dim=-1)

        loss = torch.mean(reconstruction_loss + kl_z)
        return loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=None, help='data directory')
parser.add_argument('--project_name', type=str, default='sc_autoencoders', help='wandb project name')
parser.add_argument('--run_name', type=str, default='zinb_autoencoder', help='wandb run name')
parser.add_argument('--distribution', type=str, default='zinb', help='one distribution of [zinb,nb]')
parser.add_argument('--lr', type=float, default=1.0e-4, help='the learning rate of the model')
parser.add_argument('--use_cuda', type=bool, default=True, help='use cuda or not')
parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--output', type=str, default='output', help='the output directory of the model and plots')
parser.add_argument('--save_dir', type=str, default='../checkpoints/autoencoders_checkpoints/sc_zinb', help='save directory')
args = parser.parse_args()

def train(data_directory,
         project_name,
         run_name,
         distribution='zinb',
         lr=1.0e-4,
         use_cuda=True,
         num_epochs=1000,
         batch_size=1024,
         output='output',
         save_dir='../checkpoints/autoencoders_checkpoints/additional_sc_zinb'):
    
    if not os.path.exists(output):
        os.makedirs(output)

    wandb.init(project=project_name, name=run_name)

    adata_path = "../reprogramming_schiebinger_serum_computed.h5ad"
    adata = sc.read(adata_path)
    adata.layers["counts"] = adata.X.copy()  # preserve counts
    # sc.pp.normalize_total(adata, target_sum=10e4)
    # sc.pp.log1p(adata)
    adata.raw = adata  # freeze the state in `.raw`

    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", layer="counts", n_top_genes=1000, subset=True
    )
    adata.obs = pd.get_dummies(adata.obs, columns=['cell_sets'], prefix='cell_set',dtype=float)
    cell_sets = adata.obs.filter(like='cell_set_').values

    day = adata.obs['day_numerical'].values.reshape(-1, 1)
    additional_features = np.hstack([cell_sets, day])

    print('Using distribution: ', distribution)
    data_name = data_directory.split('/')[-1].split('.')[0]

    assert np.min(adata.X) >= 0, ('Your data has negative values, pleaes specify --left_trim True if '
                                  'you still want to use this data')
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    X_with_features = np.hstack([X, additional_features])
    cell_loader = DataLoader(X_with_features, batch_size=batch_size)
    vae = scVAE(input_dim=X.shape[1],
                hidden_dim=128,
                latent_dim=64,
                additional_features_dim=additional_features.shape[1],
                distribution=distribution)  # distribution = 'nb' to use negative binomial distribution
    if use_cuda:
        vae.cuda()
    optimizer = optim.Adam(lr=lr, params=vae.parameters())
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, data in enumerate(cell_loader):
            x_data = data[:, :X.shape[1]]
            additional_features = data[:, X.shape[1]:]
            if use_cuda:
                x_data = x_data.cuda()
                additional_features = additional_features.cuda()
            optimizer.zero_grad()
            mu, dropout_logits, x_rate, q_z, z = vae(x_data, additional_features)
            loss = vae.loss_function(x_data, mu, dropout_logits, x_rate, q_z, z)
            loss.backward()
            train_loss += loss.item() * data.size(0)
            optimizer.step()
            vae.save(os.path.join(save_dir, f"model_{epoch + 1}.pt"))
            if (epoch+1) % 200 == 0:
                random_generated =  vae.generate(x_data,additional_features)

                y_pred = (random_generated != 0).cpu().flatten().tolist()
                y_true = (x_data != 0).cpu().flatten().tolist()
                precision = precision_score(y_true, y_pred, average='macro')  # 77.07
                recall = recall_score(y_true, y_pred, average='macro')  # 77.77
                f1 = f1_score(y_true, y_pred, average='macro')

                print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Precision: {precision}, Recall: {recall}")

                # Log metrics to WandB
                wandb.log({
                    "epoch": epoch + 1,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                })

                # save the model
                vae.save(os.path.join(save_dir, f"model_{epoch+1}.pt"))

        epoch_loss = train_loss / len(cell_loader.dataset)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")
        wandb.log({"epoch": epoch + 1, "training_loss": epoch_loss})
    
    wandb.finish

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    train(args.data_dir, args.project_name, args.run_name, args.distribution, args.lr, args.use_cuda, args.num_epochs, args.batch_size, args.output, args.save_dir)
        
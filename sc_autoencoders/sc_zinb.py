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


class scVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, distribution='zinb'):
        super(scVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
                                     nn.BatchNorm1d(hidden_dim, eps=0.001),
                                     nn.ReLU(),
                                     nn.Dropout(0.1)
                                     )

        self.mu_encoder = nn.Linear(hidden_dim, latent_dim)
        self.var_encoder = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                     nn.BatchNorm1d(hidden_dim, eps=0.001),
                                     nn.ReLU())
        self.mu_decoder = nn.Sequential(nn.Linear(hidden_dim, input_dim, bias=True),
                                        nn.Softmax(dim=-1),
                                        )
        self.dropout_decoder = nn.Linear(hidden_dim, input_dim, bias=True)
        self.distribution = distribution
        self.log_theta = torch.nn.Parameter(torch.randn(input_dim))

    def encode(self, x):
        h1 = self.encoder(x)
        q_mu = self.mu_encoder(h1)
        q_var = torch.exp(self.var_encoder(h1)) + 1e-4
        q_z = Normal(q_mu, q_var.sqrt())
        z = q_z.rsample()
        return q_z, z

    def decode(self, z, library):
        h = self.decoder(z) # with softplus
        mu = self.mu_decoder(h)
        dropout_logits = self.dropout_decoder(h) # this should stay as it is
        x_rate = torch.exp(library) * mu
        return torch.exp(mu), dropout_logits, x_rate

    def forward(self, x):
        library = torch.log(x.sum(1)).unsqueeze(1)
        x_log = torch.log(x + 1)
        q_z, z = self.encode(x_log)
        mu, dropout_logits, x_rate = self.decode(z, library)
        return mu, dropout_logits, x_rate, q_z, z

    def get_latent_representation(self, x):
        x_log = torch.log(x + 1)
        q_z, z = self.encode(x_log)
        return z

    def generate(self, x):
        self.eval()
        with torch.no_grad():
            mu, dropout_logits, x_rate, q_z, z = self(x)

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
parser.add_argument('--plot_embedding', type=bool, default=True, help='plot latent space embedding')
parser.add_argument('--clustering', type=bool, default=True, help='do leiden clustering')
parser.add_argument('--lable_name', type=str, default=None, help='the name of ground truth lable if applicable')
parser.add_argument('--lr', type=float, default=0.005, help='the learning rate of the model')
parser.add_argument('--use_cuda', type=bool, default=False, help='use cuda or not')
parser.add_argument('--num_epochs', type=int, default=100000, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--output', type=str, default='output', help='the output directory of the model and plots')
parser.add_argument('--distribution', type=str, default='zinb', help='one distribution of [zinb,nb]')
parser.add_argument('--project_name', type=str, default='sc_autoencoders', help='wandb project name')
parser.add_argument('--run_name', type=str, default='zinb_autoencoder', help='wandb run name')
parser.add_argument('--save_dir', type=str, default='../checkpoints/autoencoders_checkpoints/sc_zinb', help='save directory')
args = parser.parse_args()


def evaluate(data_directory,
         distribution='zinb',
         plot_embedding=False,
         clustering=False,
         lable_name=None,
         lr=1.0e-4,
         use_cuda=True,
         num_epochs=1000,
         batch_size=1024,
         output='output',
         save_dir='../checkpoints/autoencoders_checkpoints/sc_zinb'):

    adata_path = "/home/rohola/codes/cellrank_playground/reprogramming_schiebinger_serum_computed.h5ad"
    adata = sc.read(adata_path)
    adata.layers["counts"] = adata.X.copy()  # preserve counts
    # sc.pp.normalize_total(adata, target_sum=10e4)
    # sc.pp.log1p(adata)
    adata.raw = adata  # freeze the state in `.raw`

    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", layer="counts", n_top_genes=1000, subset=True
    )

    print('Using distribution: ', distribution)
    assert np.min(adata.X) >= 0, ('Your data has negative values, pleaes specify --left_trim True if '
                                  'you still want to use this data')
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    cell_loader = DataLoader(X, batch_size=batch_size)
    vae = scVAE(input_dim=X.shape[1],
                hidden_dim=128,
                latent_dim=64,
                distribution=distribution)  # distribution = 'nb' to use negative binomial distribution

    precision_scores = []
    recall_scores = []
    f1_scores = []
    with torch.no_grad():
        vae.load(os.path.join(save_dir, f"model_{num_epochs}.pt"))
        for batch_idx, data in enumerate(tqdm(cell_loader)):
            if use_cuda:
                data = data.cuda()
            random_generated = vae.generate(data)

            y_pred = (random_generated != 0).cpu().flatten().tolist()
            y_true = (data != 0).cpu().flatten().tolist()
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores) #73.56

    return avg_precision, avg_recall, avg_f1



def main(data_directory,
         distribution='zinb',
         plot_embedding=False,
         clustering=False,
         lable_name=None,
         lr=1.0e-4,
         use_cuda=True,
         num_epochs=1000,
         batch_size=1024,
         output='output',
         save_dir='../checkpoints/autoencoders_checkpoints/sc_zinb'):

    wandb.init(project="zinb_autoencoder")

    adata_path = "/home/rohola/codes/cellrank_playground/reprogramming_schiebinger_serum_computed.h5ad"
    adata = sc.read(adata_path)
    adata.layers["counts"] = adata.X.copy()  # preserve counts
    # sc.pp.normalize_total(adata, target_sum=10e4)
    # sc.pp.log1p(adata)
    adata.raw = adata  # freeze the state in `.raw`

    sc.pp.highly_variable_genes(
        adata, flavor="seurat_v3", layer="counts", n_top_genes=1000, subset=True
    )

    print('Using distribution: ', distribution)
    data_name = data_directory.split('/')[-1].split('.')[0]

    assert np.min(adata.X) >= 0, ('Your data has negative values, pleaes specify --left_trim True if '
                                  'you still want to use this data')
    X = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    cell_loader = DataLoader(X, batch_size=batch_size)
    vae = scVAE(input_dim=X.shape[1],
                hidden_dim=128,
                latent_dim=64,
                distribution=distribution)  # distribution = 'nb' to use negative binomial distribution
    if use_cuda:
        vae.cuda()
    optimizer = optim.Adam(lr=lr, params=vae.parameters())
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, data in enumerate(cell_loader):
            if use_cuda:
                data = data.cuda()
            optimizer.zero_grad()
            mu, dropout_logits, x_rate, q_z, z = vae(data)
            loss = vae.loss_function(data, mu, dropout_logits, x_rate, q_z, z)
            loss.backward()
            train_loss += loss.item() * data.size(0)
            optimizer.step()
            vae.save(os.path.join(save_dir, f"model_{epoch + 1}.pt"))
            if (epoch+1) % 200 == 0:
                random_generated =  vae.generate(data)

                y_pred = (random_generated != 0).cpu().flatten().tolist()
                y_true = (data != 0).cpu().flatten().tolist()
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
    if not os.path.exists(output):
        os.makedirs(output)

    TZ = []
    for x in cell_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        z = vae.get_latent_representation(x)
        if use_cuda:
            zz = z.cpu().detach().numpy().tolist()
        else:
            zz = z.detach().numpy().tolist()
        TZ += zz
    TZ = np.array(TZ)
    adata.obsm['z'] = TZ
    sc.pp.neighbors(adata, use_rep='z')
    if plot_embedding:
        sc.tl.umap(adata)
        if lable_name is not None:
            sc.pl.umap(adata, color=lable_name, size=50,
                       save='_{}_{}_{}_embedding.png'.format(data_name, distribution, lable_name))
        if clustering:
            sc.tl.leiden(adata)
            sc.pl.umap(adata, color='leiden', size=50,
                       save='_{}_{}_leiden_clustering.png'.format(data_name, distribution))
    # torch.save(vae, output) #save the model if you want


if __name__ == '__main__':
    # fixed the seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    assert args.data_dir != None, 'Please provide the data directory!'
    # main(args.data_dir, args.distribution, args.plot_embedding, args.clustering, args.lable_name, args.lr,
    #      args.use_cuda, args.num_epochs, args.batch_size, args.output, args.save_dir)
    from os import listdir
    from os.path import isfile, join

    onlyfiles = sorted([int(f.strip('model_').strip('.pt')) for f in listdir(args.save_dir) if isfile(join(args.save_dir, f))])
    best_f1 = 0
    # for num_epochs in onlyfiles:
    num_epochs = 2203
    avg_precision, avg_recall, avg_f1 = evaluate(args.data_dir, args.distribution, args.plot_embedding, args.clustering, args.lable_name, args.lr,
         args.use_cuda, num_epochs, args.batch_size, args.output, args.save_dir)
    print(f"Number of epochs: {num_epochs}, F1 score: {avg_f1}, Precision: {avg_precision}, Recall: {avg_recall}")
    if avg_f1 > best_f1:
        best_f1 = avg_f1
        best_num_epochs = num_epochs

    print(f"Best number of epochs: {best_num_epochs}, Best F1 score: {best_f1}")
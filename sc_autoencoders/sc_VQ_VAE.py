import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # Convert inputs from BxCxHxW to BHWxC
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # Calculate distances between input and embedding vectors
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t()))

        # Get the encoding that has the min distance
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Convert to one-hot encodings
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the inputs
        quantized = torch.matmul(encodings, self.embeddings.weight).view(inputs.shape)

        # Calculate the loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Preserve gradients
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss

class VQVAE(nn.Module):
    def __init__(self, config):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.layer_sizes[0], config.layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(config.layer_sizes[1], config.layer_sizes[2]),
            nn.ReLU()
        )

        self.fc = nn.Linear(config.layer_sizes[2], config.layer_sizes[3])
        self.quantizer = VectorQuantizer(config.num_embeddings, config.layer_sizes[3], config.commitment_cost)

        self.decoder = nn.Sequential(
            nn.Linear(config.layer_sizes[3], config.layer_sizes[2]),
            nn.ReLU(),
            nn.Linear(config.layer_sizes[2], config.layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(config.layer_sizes[1], config.layer_sizes[0]),
            nn.Sigmoid()  # To keep the output in the range [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z_e = self.encode(x)
        z_q, quantization_loss = self.quantizer(z_e)
        x_recon = self.decode(z_q)
        return x_recon, quantization_loss

def vqvae_loss(recon_x, x, quantization_loss):
    reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    return reconstruction_loss + quantization_loss
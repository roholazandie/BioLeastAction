import torch
import torch.nn as nn
import math

class HashEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(HashEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        # Initialize fixed random parameters
        self.alpha = nn.Parameter(torch.randn(embedding_dim), requires_grad=False)
        self.beta = nn.Parameter(torch.randn(embedding_dim), requires_grad=False)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_length)
        input_ids = input_ids.float().unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)
        embeddings = torch.sin(input_ids * self.alpha + self.beta)
        return embeddings  # Shape: (batch_size, seq_length, embedding_dim)


if __name__ == "__main__":
    # Create a HashEmbedding model
    embedding_dim = 768
    model = HashEmbedding(embedding_dim)

    # Generate some random input
    input_ids = torch.randint(0, 512, (32, 128))  # Shape: (batch_size, seq_length)

    # Get the embeddings
    embeddings = model(input_ids)
    print(embeddings.shape)
    print(embeddings)
import os
from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    layer_sizes: List[int]
    n_embd: int
    layer_norm_epsilon: float
    embed_dropout: float
    learning_rate: float
    batch_size: int
    num_epochs: int
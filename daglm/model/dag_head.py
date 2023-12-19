import torch
from torch import nn
import numpy as np
from .phm import phm

class OutputDAG(nn.Module):
    def __init__(self, embed_dim, vocab_size, phm_factor=4, lm_head_factor=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.phm_factor = phm_factor
        self.lm_head_factor = lm_head_factor
        self.q = phm(phm_factor, embed_dim, embed_dim)
        self.k = phm(phm_factor, embed_dim, embed_dim)
        self.lm_head = phm(lm_head_factor, embed_dim, vocab_size)

    def forward(self, x):
        batch_size, num_vertices, embed_dim = x.shape
        q = self.q(x)
        k = self.k(x)
        transition_matrix = q @ k.transpose(1, 2)

        # I don't know why it has to be normalized by sqrt(embed_dim)
        # specifically, but that is what the paper says.
        transition_matrix_norm_factor = np.sqrt(embed_dim)
        transition_matrix = transition_matrix / transition_matrix_norm_factor
        transition_matrix = torch.log_softmax(transition_matrix, dim=-1)

        vocab_log_probs = torch.log_softmax(self.lm_head(x), dim=-1)
        return transition_matrix, vocab_log_probs
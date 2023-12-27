import torch
from torch import nn
import numpy as np
from .phm import phm
from utils.fix_probs import masking4 as masking
from utils.logsumexp import logsumexp_infsafe as logsumexp

class OutputDAG(nn.Module):
    def __init__(self, embed_dim, vocab_size, num_heads, phm_factor=4, lm_head_factor=2):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.cs = embed_dim // num_heads # chunk size
        self.phm_factor = phm_factor
        self.lm_head_factor = lm_head_factor
        self.attn = phm(phm_factor, embed_dim, embed_dim * 2)
        self.gate = phm(phm_factor, embed_dim, self.num_heads)
        self.lm_head = phm(lm_head_factor, embed_dim, vocab_size)

    def forward(self, x, vertex_lens):
        batch_size, num_vertices, embed_dim = x.shape
        m, r = masking(batch_size, num_vertices, vertex_lens, x.device)
        
        g = self.gate(x)
        g = torch.log_softmax(g, dim=-1)

        q, k = self.attn(x).split(self.embed_dim, dim=-1)
        q = q.reshape(batch_size, -1, self.num_heads, self.cs)
        k = k.reshape(batch_size, -1, self.num_heads, self.cs)
        attn_scores = torch.einsum("bicf,bjcf->bijc", q, k) / (self.cs ** 0.5)
        attn_scores = attn_scores.masked_fill(~m.unsqueeze(-1), float("-inf"))
        attn_scores = torch.log_softmax(attn_scores, dim=2)
        attn_scores = attn_scores.masked_fill(r.unsqueeze(-1), float("-inf"))
        transition_matrix = logsumexp(attn_scores + g.unsqueeze(2), dim=-1).squeeze(-1)

        vocab_log_probs = torch.log_softmax(self.lm_head(x), dim=-1)
        return transition_matrix, vocab_log_probs
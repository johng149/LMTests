import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention as sdpa
from .phm import phm

class TransformerAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, factor):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.factor = factor
        self.attn_proj = phm(factor, embedding_dim, embedding_dim*3)
        self.out_proj = phm(factor, embedding_dim, embedding_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, embedding_dim = x.shape
        q, k, v = self.attn_proj(x).split(embedding_dim, dim=-1)
        q = q.view(batch_size, -1, self.num_heads, embedding_dim//self.num_heads).transpose(1,2)
        k = k.view(batch_size, -1, self.num_heads, embedding_dim//self.num_heads).transpose(1,2)
        v = v.view(batch_size, -1, self.num_heads, embedding_dim//self.num_heads).transpose(1,2)
        attn = sdpa(q, k, v, attn_mask=mask)
        attn = attn.transpose(1,2).reshape(batch_size, -1, embedding_dim)
        return self.out_proj(attn)
    
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, factor):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.factor = factor
        self.attn = TransformerAttention(embedding_dim, num_heads, factor)
        self.ffn = phm(factor, embedding_dim, embedding_dim*4)
        self.ffn2 = phm(factor, embedding_dim*4, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn_activation = nn.GELU()

    def forward(self, x, mask=None):
        attn = self.attn(self.norm1(x), mask=mask)
        x = x + attn
        ffn = self.ffn_activation(self.ffn(self.norm2(x)))
        ffn = self.ffn2(ffn)
        x = x + ffn
        return x
import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention as sdpa
from .phm import phm

class XAttention(nn.Module):
    """
    query has access to all the keys and values,
    might still want to use attn_mask to prevent attending to padding tokens
    """
    def __init__(self, embedding_dim, num_heads, factor):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.factor = factor
        self.q_attn_proj = phm(factor, embedding_dim, embedding_dim)
        self.kv_attn_proj = phm(factor, embedding_dim, embedding_dim*2)
        self.out_proj = phm(factor, embedding_dim, embedding_dim)

    def forward(self, q_x, kv_x, mask=None):
        batch_size, seq_len, embedding_dim = q_x.shape
        q = self.q_attn_proj(q_x).view(batch_size, -1, self.num_heads, embedding_dim//self.num_heads).transpose(1,2)
        k, v = self.kv_attn_proj(kv_x).split(embedding_dim, dim=-1)
        k = k.view(batch_size, -1, self.num_heads, embedding_dim//self.num_heads).transpose(1,2)
        v = v.view(batch_size, -1, self.num_heads, embedding_dim//self.num_heads).transpose(1,2)
        attn = sdpa(q, k, v, attn_mask=mask)
        attn = attn.transpose(1,2).reshape(batch_size, -1, embedding_dim)
        return self.out_proj(attn)
    
class XBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, factor):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.factor = factor
        self.attn = XAttention(embedding_dim, num_heads, factor)
        self.ffn = phm(factor, embedding_dim, embedding_dim*4)
        self.ffn2 = phm(factor, embedding_dim*4, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn_activation = nn.GELU()

    def forward(self, x, kv_x, mask=None):
        attn = self.attn(self.norm1(x), self.norm1(kv_x), mask=mask)
        x = x + attn
        ffn = self.ffn_activation(self.ffn(self.norm2(x)))
        ffn = self.ffn2(ffn)
        x = x + ffn
        return x
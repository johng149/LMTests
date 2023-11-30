import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention as sdpa
from .phm import phm

class DecoderAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, factor):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.factor = factor
        self.attn_proj = phm(factor, embedding_dim, embedding_dim*3)
        self.out_proj = phm(factor, embedding_dim, embedding_dim)

    def forward(self, x):
        batch_size, seq_len, embedding_dim = x.shape
        q, k, v = self.attn_proj(x).split(embedding_dim, dim=-1)
        q = q.view(batch_size, -1, self.num_heads, embedding_dim//self.num_heads).transpose(1,2)
        k = k.view(batch_size, -1, self.num_heads, embedding_dim//self.num_heads).transpose(1,2)
        v = v.view(batch_size, -1, self.num_heads, embedding_dim//self.num_heads).transpose(1,2)
        attn = sdpa(q, k, v, is_causal=True)
        attn = attn.transpose(1,2).reshape(batch_size, -1, embedding_dim)
        return self.out_proj(attn)
    
class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, factor):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.factor = factor
        self.attn = DecoderAttention(embedding_dim, num_heads, factor)
        self.ffn = phm(factor, embedding_dim, embedding_dim*4)
        self.ffn2 = phm(factor, embedding_dim*4, embedding_dim)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn_activation = nn.GELU()

    def forward(self, x, mask=None):
        attn = self.attn(self.norm1(x))
        x = x + attn
        ffn = self.ffn_activation(self.ffn(self.norm2(x)))
        ffn = self.ffn2(ffn)
        x = x + ffn
        return x
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, factor, num_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.factor = factor
        self.num_layers = num_layers
        self.layers = nn.ModuleList([DecoderBlock(embedding_dim, num_heads, factor) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
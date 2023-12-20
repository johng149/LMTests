import torch
from torch import nn
from .embedding import StableEmbedding
from .transformer import TransformerBlock
from .cross import XBlock
from .dag_head import OutputDAG
from typing import List, Union, Tuple
from utils.data import self_attn_mask, cross_attn_mask

class EncoderDecoderDAG(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            embedding_dim: int, 
            num_heads: int, 
            max_seq_len: int, 
            max_vertices: int, 
            layers: List[Union[int, Tuple[int, int]]],
            out_layers: int,
            phm_factor: int=4,
            lm_head_factor: int=2
        ) -> None:
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        # layers is a list, each element is either a tuple
        # (enc_layers, dec_layers) for self attention or a single
        # integer for number of cross attention layers
        # for example, if layers is [(2, 3), 1, (2, 2)]
        # and out_layers is 4, then the model will have
        # 2 encoder self attn layers, 3 decoder self attn layers,
        # and followed by 1 cross attn layer for information
        # flow from encoder to decoder and vice versa,
        # then 2 more encoder self attn layers and 2 more decoder
        # and then 4 decoder self attn layers before the output dag layer
        self.kwargs = {
            "vocab_size": vocab_size,
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "max_seq_len": max_seq_len,
            "max_vertices": max_vertices,
            "layers": layers,
            "out_layers": out_layers,
            "phm_factor": phm_factor,
            "lm_head_factor": lm_head_factor
        }
        self.token_embedding = StableEmbedding(vocab_size, embedding_dim)
        self.enc_pos_embedding = StableEmbedding(max_seq_len, embedding_dim)
        self.dec_pos_embedding = StableEmbedding(max_vertices, embedding_dim)
        self.layers = nn.ModuleList()
        for layer in layers:
            if isinstance(layer, int):
                enc_kv_to_dec_q = nn.ModuleList()
                dec_kv_to_enc_q = nn.ModuleList()
                for _ in range(layer):
                    enc_kv_to_dec_q.append(XBlock(embedding_dim, num_heads, phm_factor))
                    dec_kv_to_enc_q.append(XBlock(embedding_dim, num_heads, phm_factor))
                combined = nn.ModuleList([enc_kv_to_dec_q, dec_kv_to_enc_q])
                self.layers.append(combined)
            else:
                enc_layers, dec_layers = layer
                # here we make a nested nn.ModuleList
                encoder_layers = nn.ModuleList()
                decoder_layers = nn.ModuleList()
                for _ in range(enc_layers):
                    encoder_layers.append(TransformerBlock(embedding_dim, num_heads, phm_factor))
                for _ in range(dec_layers):
                    decoder_layers.append(TransformerBlock(embedding_dim, num_heads, phm_factor))
                combined = nn.ModuleList([encoder_layers, decoder_layers])
                self.layers.append(combined)
        self.final_cross = XBlock(embedding_dim, num_heads, phm_factor)
        self.out_layers = nn.ModuleList()
        for _ in range(out_layers):
            self.out_layers.append(TransformerBlock(embedding_dim, num_heads, phm_factor))
        self.output_dag = OutputDAG(embedding_dim, vocab_size, phm_factor, lm_head_factor)

    def forward(self, enc_tokens, dec_vertex_ids, enc_is_pad, dec_is_pad):
        enc_self = self_attn_mask(enc_is_pad).unsqueeze(1)
        dec_self = self_attn_mask(dec_is_pad).unsqueeze(1)
        enc_kv = cross_attn_mask(enc_is_pad, dec_is_pad).unsqueeze(1)
        dec_kv = cross_attn_mask(dec_is_pad, enc_is_pad).unsqueeze(1)

        enc_tokens = self.token_embedding(enc_tokens)
        enc_pos = self.enc_pos_embedding(torch.arange(enc_tokens.shape[1], device=enc_tokens.device))
        enc_x = enc_tokens + enc_pos

        dec_x = self.dec_pos_embedding(dec_vertex_ids)

        for layer in self.layers:
            # each layer should be a ModuleList of its own, with encoder and decoder layers
            # if the first layer in the encoder is XBlock, then we are on a cross attn layer
            if isinstance(layer[0][0], XBlock):
                enc_kv_to_dec_q, dec_kv_to_enc_q = layer
                for enc_kv_block, dec_kv_block in zip(enc_kv_to_dec_q, dec_kv_to_enc_q):
                    dec_x = enc_kv_block(dec_x, enc_x, mask=enc_kv)
                    enc_x = dec_kv_block(enc_x, dec_x, mask=dec_kv)
            else:
                encoder_layers, decoder_layers = layer
                for encoder_layer in encoder_layers:
                    enc_x = encoder_layer(enc_x, mask=enc_self)
                for decoder_layer in decoder_layers:
                    dec_x = decoder_layer(dec_x, mask=dec_self)
        dec_x = self.final_cross(dec_x, enc_x, mask=enc_kv)
        for layer in self.out_layers:
            dec_x = layer(dec_x, mask=dec_self)
        return self.output_dag(dec_x)
import torch
from torch import nn
from .embedding import StableEmbedding
from .encoder import Encoder
from .decoder import DecoderBlock
from .cross import FullVisibilityXBlock
from .phm import phm

class VanillaEncoderDecoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_encoder_layers, num_decoder_layers, vocab_size, factor, lm_head_factor, eos_idx, bos_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.vocab_size = vocab_size
        self.factor = factor
        self.lm_head_factor = lm_head_factor
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx
        self.kwargs = {
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "vocab_size": vocab_size,
            "factor": factor,
            "lm_head_factor": lm_head_factor,
            "eos_idx": eos_idx,
            "bos_idx": bos_idx,
        }

        self.embedding = StableEmbedding(vocab_size, embedding_dim)
        
        # every decoder layer except for the first gets to do cross attention
        # with the final hidden state of the encoder
        self.encoder_layers = Encoder(embedding_dim, num_heads, factor, num_encoder_layers)
        self.decoder_layers = nn.ModuleList([DecoderBlock(embedding_dim, num_heads, factor) for _ in range(num_decoder_layers)])
        self.cross_layers = nn.ModuleList([FullVisibilityXBlock(embedding_dim, num_heads, factor) for _ in range(num_decoder_layers-1)])

        # the final layer is a linear layer with a softmax activation, producing logits
        # to indicate the probability of each word in the vocabulary
        self.lm_head_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = phm(lm_head_factor, embedding_dim, vocab_size)

    def forward(self, context_tokens, answer_tokens):
        batch_size, context_len = context_tokens.shape
        _, answer_len = answer_tokens.shape

        context_attn_mask = (context_tokens != self.eos_idx).float()
        context_attn_mask.masked_fill_(context_attn_mask.logical_not(), float("-inf"))
        context_attn_mask.masked_fill_(context_attn_mask == 1, 0)

        context_self_attn_mask = context_attn_mask.unsqueeze(1).unsqueeze(2).repeat(1, self.num_heads, context_len, 1)
        context_x_attn_mask = context_attn_mask.unsqueeze(1).unsqueeze(2).repeat(1, self.num_heads, answer_len, 1)

        context = self.embedding(context_tokens)
        answer = self.embedding(answer_tokens)

        context = self.encoder_layers(context, mask=context_self_attn_mask)
        
        # step one of answer, we don't need to provide a mask
        # since decoder already knows that it is causal mask
        answer = self.decoder_layers[0](answer)

        # iterate through the rest of the decoder layers
        # performing cross attention before the self attention
        assert len(self.decoder_layers) - 1 == len(self.cross_layers)
        for i, (decoder_layer, cross_layer) in enumerate(zip(self.decoder_layers[1:], self.cross_layers)):
            answer = cross_layer(answer, context, mask=context_x_attn_mask)
            answer = decoder_layer(answer)

        # the final layer is a linear layer with a softmax activation, producing logits
        # to indicate the probability of each word in the vocabulary
        answer = self.lm_head_norm(answer)
        answer = self.lm_head(answer)
        return answer
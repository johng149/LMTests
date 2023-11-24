import torch
from torch import nn
from torch.nn import Module, LayerNorm

from .phm import phm
from .fnet import FNet
from .cosformer import CosformerAttention as Attention
from .reader import Reader

class VocabLogits(Module):
    def __init__(self, vocab_size, embed_dim, phm_factor=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.logits = phm(phm_factor, embed_dim, vocab_size)
        self.norm = LayerNorm(embed_dim)

    def forward(self, x):
        # assumes x is seq_len x batch_size x embed_dim
        return self.logits(self.norm(x.transpose(0, 1)))
    
class MixerLayer(Module):

    def __init__(self, embed_dim, num_heads, phm_factor=4):
        super().__init__()
        # mixer layer takes into account the question,
        # context, and answer (answer maybe masked in some places)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.phm_factor = phm_factor

        self.question_fnet1 = FNet(embed_dim, phm_factor)
        self.question_self_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)
        self.question_fnet2 = FNet(embed_dim, phm_factor)

        self.context_fnet1 = FNet(embed_dim, phm_factor)
        self.context_self_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)
        self.context_fnet2 = FNet(embed_dim, phm_factor)

        self.answer_fnet1 = FNet(embed_dim, phm_factor)
        self.answer_self_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)
        self.answer_fnet2 = FNet(embed_dim, phm_factor)

        # now mix question with kv from context
        self.question_with_context_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)
        self.question_with_context_attn2 = Attention(embed_dim, num_heads, phm_factor=phm_factor)

        # and mix context with kv from question
        self.context_with_question_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)
        self.context_with_question_attn2 = Attention(embed_dim, num_heads, phm_factor=phm_factor)

        # and mix answer with kv from question
        self.answer_with_question_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)

        # and mix answer with kv from context
        self.answer_with_context_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)

    def forward(self, question, context, answer):
        # question
        question = self.question_fnet1(question)
        question = self.question_self_attn(question)
        question = self.question_fnet2(question)

        # context
        context = self.context_fnet1(context)
        context = self.context_self_attn(context)
        context = self.context_fnet2(context)

        # answer
        answer = self.answer_fnet1(answer)
        answer = self.answer_self_attn(answer)
        answer = self.answer_fnet2(answer)

        # question with context
        question_with_context = self.question_with_context_attn(question, context, context)

        # context with question
        context_with_question = self.context_with_question_attn(context, question, question)

        # and mix again
        question_with_context = self.question_with_context_attn2(question_with_context, context_with_question, context_with_question)

        # and mix again
        context_with_question = self.context_with_question_attn2(context_with_question, question_with_context, question_with_context)

        # now give context to answer
        answer_with_context = self.answer_with_context_attn(answer, context, context)

        # and then using that, mix in question
        answer_with_question = self.answer_with_question_attn(answer_with_context, question, question)

        return question_with_context, context_with_question, answer_with_question
    
class MixerLayerWithoutFNetAndFewerMixing(Module):

    def __init__(self, embed_dim, num_heads, phm_factor=4):
        super().__init__()
        # mixer layer takes into account the question,
        # context, and answer (answer maybe masked in some places)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.phm_factor = phm_factor

        self.question_self_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)

        self.context_self_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)

        self.answer_self_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)

        # now mix question with kv from context
        self.question_with_context_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)

        # and mix context with kv from question
        self.context_with_question_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)

        # and mix answer with kv from question
        self.answer_with_question_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)

        # and mix answer with kv from context
        self.answer_with_context_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)

    def forward(self, question, context, answer):
        # question
        question = self.question_self_attn(question)

        # context
        context = self.context_self_attn(context)

        # answer
        answer = self.answer_self_attn(answer)

        # question with context
        question_with_context = self.question_with_context_attn(question, context, context)

        # context with question
        context_with_question = self.context_with_question_attn(context, question, question)

        # now give context to answer
        answer_with_context = self.answer_with_context_attn(answer, context, context)

        # and then using that, mix in question
        answer_with_question = self.answer_with_question_attn(answer_with_context, question, question)

        return question_with_context, context_with_question, answer_with_question

class FauxMixingLayer(Module):
    def __init__(self, embed_dim, num_heads, phm_factor=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.phm_factor = phm_factor

        self.self_attn = Attention(embed_dim, num_heads, phm_factor=phm_factor)

    def forward(self, x):
        return self.self_attn(x)


class MainModel(Module):
    def __init__(self, vocab_size, embed_dim, num_heads, phm_factor=4, vocab_phm_factor=2, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.phm_factor = phm_factor
        self.vocab_phm_factor = vocab_phm_factor
        self.num_layers = num_layers
        self.kwargs = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "phm_factor": phm_factor,
            "vocab_phm_factor": vocab_phm_factor,
            "num_layers": num_layers
        }
        
        self.vocab_logits = VocabLogits(vocab_size, embed_dim, phm_factor=vocab_phm_factor)
        self.mixer_layers = nn.ModuleList([MixerLayer(embed_dim, num_heads, phm_factor=phm_factor) for _ in range(num_layers)])
        self.reader = Reader(vocab_size, embed_dim)

    def forward(self, question_ids, context_ids, answer_ids):
        # question_ids, context_ids, and answer_ids are all seq_len x batch_size
        question = self.reader(question_ids)
        context = self.reader(context_ids)
        answer = self.reader(answer_ids)
        for mixer_layer in self.mixer_layers:
            question, context, answer = mixer_layer(question, context, answer)
        return self.vocab_logits(answer)

class MainModelFauxMixing(Module):
    def __init__(self, vocab_size, embed_dim, num_heads, phm_factor=4, vocab_phm_factor=2, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.phm_factor = phm_factor
        self.vocab_phm_factor = vocab_phm_factor
        self.num_layers = num_layers
        self.kwargs = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "phm_factor": phm_factor,
            "vocab_phm_factor": vocab_phm_factor,
            "num_layers": num_layers
        }
        
        self.vocab_logits = VocabLogits(vocab_size, embed_dim, phm_factor=vocab_phm_factor)
        self.mixer_layers = nn.ModuleList([FauxMixingLayer(embed_dim, num_heads, phm_factor=phm_factor) for _ in range(num_layers)])
        self.reader = Reader(vocab_size, embed_dim)

    def forward(self, x):
        # question_ids, context_ids, and answer_ids are all seq_len x batch_size
        x = self.reader(x)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        return self.vocab_logits(x)

class MainModelFauxPytorch(Module):
    # uses pytorch nn.TransformerEncoderLayer instead of MixerLayer
    def __init__(self, vocab_size, embed_dim, num_heads, phm_factor=4, vocab_phm_factor=2, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.phm_factor = phm_factor
        self.vocab_phm_factor = vocab_phm_factor
        self.num_layers = num_layers
        self.kwargs = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "phm_factor": phm_factor,
            "vocab_phm_factor": vocab_phm_factor,
            "num_layers": num_layers
        }
        
        self.vocab_logits = VocabLogits(vocab_size, embed_dim, phm_factor=vocab_phm_factor)
        self.mixer_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.reader = Reader(vocab_size, embed_dim)

    def forward(self, x):
        # question_ids, context_ids, and answer_ids are all seq_len x batch_size
        x = self.reader(x)
        # remember that reader assumes we are using MixerLayer, so it transposes
        # but we are not so we need to transpose back
        x = self.mixer_layers(x.transpose(0, 1))
        return self.vocab_logits(x)

class MainModelWithoutFNetAndFewerMixing(Module):
    def __init__(self, vocab_size, embed_dim, num_heads, phm_factor=4, vocab_phm_factor=2, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.phm_factor = phm_factor
        self.vocab_phm_factor = vocab_phm_factor
        self.num_layers = num_layers
        self.kwargs = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "phm_factor": phm_factor,
            "vocab_phm_factor": vocab_phm_factor,
            "num_layers": num_layers
        }
        
        self.vocab_logits = VocabLogits(vocab_size, embed_dim, phm_factor=vocab_phm_factor)
        self.mixer_layers = nn.ModuleList([MixerLayerWithoutFNetAndFewerMixing(embed_dim, num_heads, phm_factor=phm_factor) for _ in range(num_layers)])
        self.reader = Reader(vocab_size, embed_dim)

    def forward(self, question_ids, context_ids, answer_ids):
        # question_ids, context_ids, and answer_ids are all seq_len x batch_size
        question = self.reader(question_ids)
        context = self.reader(context_ids)
        answer = self.reader(answer_ids)
        for mixer_layer in self.mixer_layers:
            question, context, answer = mixer_layer(question, context, answer)
        return self.vocab_logits(answer)
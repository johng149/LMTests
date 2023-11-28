import tokenmonster
import torch

tokenizer_file = "english-8000-balanced-v1"

def prepare_tokenizer():
    vocab = tokenmonster.load_multiprocess_safe(tokenizer_file)
    initial_vocab_size = len(vocab)
    eos_idx = initial_vocab_size
    bos_idx = initial_vocab_size + 1
    return vocab, eos_idx, bos_idx, initial_vocab_size + 2

def pad_tensor(t, e, b, l):
    """
    Pads given tensor to length l using element e, and prepends b.

    If `t` is already at or longer than `l`, its elements are truncated to length
    `l - 1` and prepended with `b`.

    @param t: tensor of shape (batch_size, seq_len) or (seq_len)
    @param e: index of element to pad with
    @param b: index of element to prepend with
    @param l: length to pad to

    @return: padded tensor of shape (batch_size, l) if t is 2D, else (l)
    """
    is_1d = len(t.shape) == 1
    if is_1d:
        t = t.unsqueeze(0)
    batch_size, seq_len = t.shape
    needs_trunc = seq_len >= l
    if needs_trunc:
        t = t[:, :l - 1]
        seq_len = l - 1
    pad_len = l - seq_len
    padding = torch.full((batch_size, pad_len-1), e, dtype=torch.long)
    t = torch.cat((t, padding), dim=1)
    
    # prepend bos
    bos = torch.full((batch_size, 1), b, dtype=torch.long)
    t = torch.cat((bos, t), dim=1)
    if is_1d:
        t = t.squeeze(0)
    return t
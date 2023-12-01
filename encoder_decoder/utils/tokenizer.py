import tokenmonster
import torch

tokenizer_file = "english-8000-balanced-v1"

def prepare_tokenizer():
    vocab = tokenmonster.load_multiprocess_safe(tokenizer_file)
    initial_vocab_size = len(vocab)
    eos_idx = initial_vocab_size
    bos_idx = initial_vocab_size + 1
    pad_idx = initial_vocab_size + 2
    answer_pad_idx = initial_vocab_size + 3
    return vocab, eos_idx, bos_idx, pad_idx, answer_pad_idx, initial_vocab_size + 4

def pad_tensor(t, l, e, b, p):
    """
    Prepends `b` to `t`, appends `e` to `t` and then pads this new `t` with `p`
    until it is of length `l`.

    If `t` is already at or longer than `l`, its elements are truncated to length
    `l - 2` and then `b` and `e` are prepended and appended respectively.

    @param t: tensor of shape (batch_size, seq_len) or (seq_len)
    @param l: length to pad to
    @param e: index of element to pad with
    @param b: index of element to prepend with
    @param p: index of element to append with

    @return: padded tensor of shape (batch_size, l) if t is 2D, else (l)
    """
    is_1d = len(t.shape) == 1
    if is_1d:
        t = t.unsqueeze(0)
    batch_size, seq_len = t.shape
    needs_trunc = seq_len >= l
    if needs_trunc:
        t = t[:, :l - 2]
        seq_len = l - 2
    pad_len = l - seq_len
    
    #prepend bos
    bos = torch.full((batch_size, 1), b, dtype=torch.long)
    t = torch.cat([bos, t], dim=1)

    #append eos
    eos = torch.full((batch_size, 1), e, dtype=torch.long)
    t = torch.cat([t, eos], dim=1)

    #pad to length
    pad_len = l - t.shape[1]
    padding = torch.full((batch_size, pad_len), p, dtype=torch.long)
    t = torch.cat([t, padding], dim=1)
    if is_1d:
        t = t.squeeze(0)
    return t
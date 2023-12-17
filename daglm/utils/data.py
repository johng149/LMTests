import torch

def process_data(samples, pad_idx, factor):
    """
    Process the given samples by calculating the number of tokens per row that are not padding,
    creating a token mask, and creating a vertex mask.

    Args:
        samples (torch.Tensor): The input samples.
        pad_idx (int): The index of the padding token.
        factor (int): The factor by which the token mask is repeated to create the vertex mask.

    Returns:
        torch.Tensor: The number of tokens per row that are not padding.
        torch.Tensor: The number of vertices per row that are not padding.
        torch.Tensor: The token mask.
        torch.Tensor: The vertex mask.
    """
    batch_size, l = samples.shape

    # count number of tokens per row that are not padding
    token_lens = torch.sum(samples != pad_idx, dim=1)
    vertex_lens = token_lens * factor
    token_mask = (samples == pad_idx)
    vertex_mask = token_mask.repeat_interleave(factor, dim=1)
    return token_lens, vertex_lens, token_mask, vertex_mask
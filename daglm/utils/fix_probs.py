import torch

def acyclic_mask(transition_matrix):
    """
    Generates a mask that, when applied to the transition matrix, ensures that
    vertex i can only transition to vertices j where j > i.

    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, vertices, vertices).

    Returns:
        torch.Tensor: The acyclic mask of shape (vertices, vertices).
    """
    batch_size, vertices, _ = transition_matrix.shape
    mask = torch.tril(torch.ones((vertices, vertices))).to(transition_matrix.device)
    return mask

def padding_transition_mask(transition_matrix, vertex_lens):
    """
    Generates a mask that, when applied to the transition matrix, prevents vertices
    from transitioning to padding vertices. It is assumed that the padding vertices
    are at the end of the sequence.

    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, vertices, vertices).
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The padding transition mask of shape (batch_size, vertices, vertices).
    """
    batch_size, vertices, _ = transition_matrix.shape
    vertex_lens_mask = torch.arange(vertices).to(transition_matrix.device).repeat(len(vertex_lens), 1) < vertex_lens.unsqueeze(-1)
    mask = torch.ones_like(transition_matrix)
    mask.transpose(1,2)[vertex_lens_mask] = 0
    return mask

def masking(transition_matrix, vertex_lens):
    """
    Creates masking to the transition matrix based on acyclic and padding masks.

    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, vertices, vertices).
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The masked transition matrix of shape (batch_size, vertices, vertices).
    """
    acyclic = acyclic_mask(transition_matrix)
    padding = padding_transition_mask(transition_matrix, vertex_lens)
    return padding + acyclic

def special_masking(transition_matrix, vertex_lens):
    """
    Creates masking to the transition matrix based on acyclic and padding masks.
    However, the diagonal offset by 1 is not masked out

    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, vertices, vertices).
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The masked transition matrix of shape (batch_size, vertices, vertices).
    """
    mask = masking(transition_matrix, vertex_lens)
    _, vertices, _ = transition_matrix.shape
    special = torch.zeros((vertices, vertices)).to(transition_matrix.device)
    special.diagonal(offset=1).fill_(1)
    mask = mask.masked_fill(special == 1, 0)
    return mask

def fix_probs(logprobs, mask):
    """
    Fixes the probabilities in log space by ensuring that the sum of each row is 100%
    and any available probability mass is distributed evenly among the non-masked entries.
    An edge case is when there are no non-masked entries in a row, in which case all
    probabilities are set to -inf.

    The shape of mask must be broadcastable to the shape of logprobs.
    For example if logprobs is (batch_size, l, l), then mask can be (l, l) or (batch_size, 1, l).

    Args:
        logprobs (torch.Tensor): Tensor containing the log probabilities.
        mask (torch.Tensor): Tensor containing the mask where non-zero values indicate positions to be masked out.

    Returns:
        torch.Tensor: Tensor containing the fixed log probabilities.

    """
    batch_size, l, _ = logprobs.shape

    # any part of the mask where the value is not 0, we mask out
    logprobs = logprobs.masked_fill(mask != 0, float('-inf'))
    probsmatrix = torch.exp(logprobs)
    remaining = torch.sum(probsmatrix, dim=2)
    remaining = 1 - remaining
    probnonzero = torch.sum(mask == 0, dim=-1)
    remaining = remaining / probnonzero
    probsmatrix = probsmatrix + remaining.unsqueeze(2)

    # at this point, it is mostly correct, however if there 
    # are rows where the number of non zeros (probnonzero)
    # is 0, then we get infinities at those positions, which
    # are obviously wrong, so we keep only positions
    # where we haven't masked out
    probsmatrix = probsmatrix.masked_fill(mask != 0, 0)
    logprobs = torch.log(probsmatrix)
    return logprobs
import torch
from typing import Union

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

def acyclic_mask2(vertices, device):
    """
    Generates a mask that, when applied to the transition matrix, ensures that
    vertex i can only transition to vertices j where j > i.

    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, vertices, vertices).

    Returns:
        torch.Tensor: The acyclic mask of shape (vertices, vertices).
    """
    mask = torch.tril(torch.ones((vertices, vertices))).to(device)
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

def padding_transition_mask2(batch_size, vertices, vertex_lens, device):
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
    vertex_lens_mask = torch.arange(vertices).to(device).repeat(len(vertex_lens), 1) < vertex_lens.unsqueeze(-1)
    mask = torch.ones((batch_size, vertices, vertices), device=device)
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

def masking1(batch_size, vertices, vertex_lens, device):
    """
    Creates masking to the transition matrix based on acyclic and padding masks.

    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, vertices, vertices).
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The masked transition matrix of shape (batch_size, vertices, vertices).
    """
    acyclic = acyclic_mask2(vertices, device)
    padding = padding_transition_mask2(batch_size, vertices, vertex_lens, device)
    return padding + acyclic

def masking2(transition_matrix, vertex_lens):
    """
    Creates masking to the transition matrix based on acyclic and padding masks.
    Unlike `masking`, this function sets elements that should not be masked to
    True

    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, vertices, vertices).
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The masked transition matrix of shape (batch_size, vertices, vertices).
    """
    m = masking(transition_matrix, vertex_lens)
    return m == 0

def masking3(transition_matrix, vertex_lens):
    """
    Creates masking to the transition matrix based on acyclic and padding masks.
    Output is based on those for `masking2`, however, if there are any rows that
    contain only False (that is, all elements are masked), then every element
    in that row is set to True.

    Also returns a row mask that describes which rows are all masked

    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, vertices, vertices).
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The masked transition matrix of shape (batch_size, vertices, vertices).
        torch.Tensor: The row mask of shape (batch_size, vertices, 1).
    """
    m = masking2(transition_matrix, vertex_lens)
    r = m.sum(dim=2, keepdim=True) == 0
    return m.masked_fill(r, True), r

def masking4(
        batch_size: int, 
        vertices: int, 
        vertex_lens: torch.Tensor,
        device: Union[str, torch.device]
    ):
    """
    Creates masking to the transition matrix based on acyclic and padding masks.
    Output is based on those for `masking2`, however, if there are any rows that
    contain only False (that is, all elements are masked), then every element
    in that row is set to True.

    Also returns a row mask that describes which rows are all masked

    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, vertices, vertices).
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The masked transition matrix of shape (batch_size, vertices, vertices).
        torch.Tensor: The row mask of shape (batch_size, vertices, 1).
    """
    m = masking1(batch_size, vertices, vertex_lens, device) == 0
    r = m.sum(dim=2, keepdim=True) == 0
    return m.masked_fill(r, True), r

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
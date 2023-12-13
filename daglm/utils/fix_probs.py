import torch

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
import torch
from utils.vector_gather import vector_gather
from utils.fix_probs import special_masking

def dag_loss_raw(targets, transition_matrix, emission_probs):
    """
    Calculates the directed acyclic graph (DAG) loss given the targets, transition matrix, and emission probabilities.
    It returns the dynamic programming table of which one of the entries is the DAG loss.

    Args:
        targets (torch.Tensor): The target sequence of shape (batch_size, m).
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, l, l).
        emission_probs (torch.Tensor): The emission probabilities of shape (batch_size, l, vocab_size).

    Returns:
        torch.Tensor: The DAG loss of shape (batch_size, m, l).
    """
    batch_size, m = targets.shape
    _, l, vocab_size = emission_probs.shape
    dp = torch.ones((batch_size, m, l))
    dp[dp == 1] = -float('inf')
    initial_probs = torch.gather(emission_probs, dim=2, index=targets[:, 0].unsqueeze(1).unsqueeze(2))
    dp[:, 0, 0] = initial_probs.squeeze(2).squeeze(1)
    # assumes that transition_matrix and emission_probs are already in log space
    # also we need to tranpose emission_probs so it is vocab_size x l
    # so the vector gather works
    emission_probs = emission_probs.transpose(1, 2)
    dp = dp.to(transition_matrix.device)
    for i in range(1, m):
        dp[:, i-1, :i-1] = -float('inf')
        dp[:, i, :] = vector_gather(emission_probs, targets[:, i]) + (torch.logsumexp(dp[:, i-1, :].unsqueeze(1).transpose(1, 2) + transition_matrix, dim=1))
    return dp

def process_dp(dp, target_lens, vertex_lens):
    """
    Processes the dynamic programming table (dp) to extract the correct loss values.
    The target lengths and vertex lengths are needed to determine which values to extract
    and which values are a result of padding and should be ignored.

    Args:
        dp (torch.Tensor): The dynamic programming table of shape (batch_size, m, l).
        target_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the length of each target sequence.
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The values corresponding to the last target and last vertex of shape (batch_size,).
    """
    dp_values = vector_gather(dp, target_lens - 1)
    values = torch.gather(dp_values, dim=1, index=(vertex_lens - 1).unsqueeze(-1))
    return values

def dag_loss(targets, transition_matrix, emission_probs, target_lens, vertex_lens):
    """
    Calculates the directed acyclic graph (DAG) loss given the targets, transition matrix, and emission probabilities.

    Args:
        targets (torch.Tensor): The target sequence of shape (batch_size, m).
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, l, l).
        emission_probs (torch.Tensor): The emission probabilities of shape (batch_size, l, vocab_size).
        target_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the length of each target sequence.
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The DAG loss of shape (batch_size,).
    """
    dp = dag_loss_raw(targets, transition_matrix, emission_probs)
    values = process_dp(dp, target_lens, vertex_lens)
    return -torch.sum(values)
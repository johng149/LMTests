import torch
from utils.vector_gather import vector_gather
from utils.fix_probs import fix_probs

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
    print(f"Transition matrix shape: {transition_matrix.shape}")
    for i in range(1, m):
        t1 = vector_gather(emission_probs, targets[:, i])
        t2 = (torch.logsumexp(dp[:, i-1, :].unsqueeze(1).transpose(1, 2) + transition_matrix, dim=1))
        print(f"Token prob shape: {t1.shape}")
        print(f"Transition prob shape: {t2.shape}")
        print(f"DP slice shape: {dp[:, i, :].shape}")
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
import torch

def greedy_decoding(log_transition_probs, log_emission_probs, end_emission_idx):
    """
    Produces most likely sequence of states given the log transition and emission probabilities.

    Stops early when the end_emission_idx is reached or a cycle is detected

    @param log_transition_probs: transition probs of shape (num_states, num_states)
    @param log_emission_probs: emission probs of shape (num_states, num_obs)
    @return: most likely sequence of states of shape (num_obs)
    """
    tokens = torch.argmax(log_emission_probs, dim=1)
    edges = torch.argmax(log_transition_probs, dim=1)
    output = [tokens[0].item()]
    i = 0
    minimum_i = 0
    l, _ = log_transition_probs.shape
    while i < l:
        if output[-1] == end_emission_idx:
            return output
        i = edges[i].item()
        minimum_i = max(minimum_i, i)
        if i >= l or i < minimum_i:
            return output
        output.append(tokens[i].item())
    return output

def lookahead(log_transition_probs, log_emission_probs, end_emission_idx):
    """
    Produces most likely sequence of states given the log transition and emission probabilities,
    with transitions taking into account the emission probabilities

    Stops early when the end_emission_idx is reached or a cycle is detected

    @param log_transition_probs: transition probs of shape (num_states, num_states)
    @param log_emission_probs: emission probs of shape (num_states, num_obs)
    @return: most likely sequence of states of shape (num_obs)
    """
    log_transition_probs = log_transition_probs + torch.max(log_emission_probs, dim=1)[0].unsqueeze(0)
    return greedy_decoding(log_transition_probs, log_emission_probs, end_emission_idx)
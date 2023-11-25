import torch
class SyntheticTokenizer:

    def __init__(self):
        """
        SyntheticTokenizer works on the states defined in SyntheticDatasetFSM
        """
        self.token_to_id = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        self.id_to_token = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

    def tokenize_sequence(self, sequence, dtype=torch.long):
        """
        Tokenize a sequence of states
        """
        return torch.tensor([self.token_to_id[s] for s in sequence], dtype=dtype)
    
    def decode_sequence(self, sequence):
        """
        Decode a sequence of tokens, if an id is not known, it is skipped
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        return ''.join([self.id_to_token.get(i, '') for i in sequence])
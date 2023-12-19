import torch
from torch.utils.data import Dataset
import os

class TranslateDataset(Dataset):

    def __init__(self, data_dir, data_file):
        self.data_dir = data_dir
        self.data_file = data_file
        self.load_data()

    def load_data(self):
        print(f"Loading data from {os.path.join(self.data_dir, self.data_file)}")
        self.ens, self.zhs = torch.load(os.path.join(self.data_dir, self.data_file))
        self.num_samples, _ = self.ens.shape # num_samples, seq_len

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.ens[idx], self.zhs[idx]
    
def collate_fn(batch, pad_idx=65000):
    """
    Collate function for the TranslateDataset.
    For each batch, the English and Chinese samples are stacked into tensors.
    And then the columns that only consist of padding tokens are removed

    Args:
        batch (list): The batch of samples.

    Returns:
        torch.Tensor: The batch of English samples.
        torch.Tensor: The batch of Chinese samples.
    """
    ens, zhs = zip(*batch)
    ens = torch.stack(ens)
    zhs = torch.stack(zhs)

    # remove columns that only consist of padding tokens
    ens = ens[:, (ens != pad_idx).any(dim=0)]
    zhs = zhs[:, (zhs != pad_idx).any(dim=0)]
    
    return ens, zhs

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
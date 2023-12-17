# Data.py

## proceess_data.py

Given a sample of shape `batch_size x seq_len` and the `pad_idx` and `factor`, it returns `token_lens, vertex_lens, token_mask, vertex_mask`.

The `token_lens` tells us, for each row, how many tokens are in the row that are not padding.

Similarly, `vertex_lens` tells us, for each row, the corresponding number of vertices that are not padding.

These two are used to help calculate the loss, since we don't care about the scores assigned to the padding vertices.

The `token_mask` is used for self-attention on the encoder side, or when the data from the given sample is used as the key-values for cross attention (data from encoder is used as kv for queries from decoder)

As for `vertex_mask`, it is used for self-attention on the decoder side, or when the data from the decoder is used as the key-values for cross attention (data from decoder is used as kv for queries from encoder). This is because the decoder creates an acyclic directed graph, but some of the vertices correspond to padding positions, so we need to mask them out.

These two are used in the model forward process, I don't think that loss requires these.
When calculating the DAG loss, there may be some values in the sequence which are padding tokens, and these values have their respective vertices.

To determine the correct DAG loss values from the dynamic programming table, we need to select the row corresponding to the final non-padding token in the sequence, and then select the column corresponding to the final non-padding vertex.

To do this, we need to be given `target_lens` and `vertex_lens` such as: 

```
sequences = torch.tensor([
    [
        1, 2, -1, -1
    ],
    [
        1, 2, 3, 4, -1, -1
    ]
])
target_lens = torch.tensor([2, 4])
vertex_lens = torch.tensor([4, 8])
```

The number of vertices per token can be some function of the number of non-padding tokens in the sequence (in the paper they use a `factor` hyperparameter which the lengths are multiplied by). Since the loss function doesn't know this function, we need to provide it with the vertex lengths.
import torch


def merge_tensor(tensor_a, tensor_b, pad=0):
    """
    Merges two tensors along the first dimension, padding with a specified value if necessary.

    Args:
        tensor_a (torch.Tensor): The first tensor to merge.
        tensor_b (torch.Tensor): The second tensor to merge.
        pad (int or float): The value to use for padding.

    Returns:
        torch.Tensor: The merged tensor with padding applied.
    """
    merged_dim = -1
    max_len = max(tensor_a.size(merged_dim), tensor_b.size(merged_dim))
    padded_a = torch.nn.functional.pad(
        tensor_a, (0, max_len - tensor_a.size(merged_dim), 0, 0), value=pad
    )
    padded_b = torch.nn.functional.pad(
        tensor_b, (0, max_len - tensor_b.size(merged_dim), 0, 0), value=pad
    )
    return torch.cat([padded_a, padded_b], dim=0)

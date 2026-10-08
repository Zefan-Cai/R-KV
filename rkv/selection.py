import torch


def select_kept_positions(scores, keep_past, window_size):
    """Keep the highest-scoring past tokens and the entire recent window, in position order."""
    past_len = scores.numel()
    top = scores.topk(keep_past).indices.sort().values
    recent = torch.arange(past_len, past_len + window_size, device=scores.device)
    return torch.cat([top, recent])

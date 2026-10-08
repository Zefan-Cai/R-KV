import torch


def aggregate_scores(layer_scores):
    """Average scores across KV heads, then sum across layers for one request."""
    if not layer_scores:
        raise ValueError("Expected scores from at least one layer")
    combined = None
    for scores in layer_scores:
        if scores.ndim != 3 or scores.shape[0] != 1:
            raise ValueError("Expected per-layer scores of shape [1, kv_heads, tokens]")
        layer_mean = scores.mean(dim=1)[0]
        if combined is not None and layer_mean.shape != combined.shape:
            raise ValueError("Layer scores must have matching token lengths")
        combined = layer_mean if combined is None else combined + layer_mean
    return combined


def select_kept_positions(scores, keep_past, window_size):
    """Keep the highest-scoring past tokens and the entire recent window, in position order."""
    past_len = scores.numel()
    top = scores.topk(keep_past).indices.sort().values
    recent = torch.arange(past_len, past_len + window_size, device=scores.device)
    return torch.cat([top, recent])

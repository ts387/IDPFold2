import torch


def binned_gather(x, indices, bins, expert_capacity, top_k):
    bins_flat = bins.flatten()
    num_experts = bins_flat.shape[0]
    num_columns = x.shape[1]
    expert_capacity = int(expert_capacity)
    out = torch.zeros((num_experts, expert_capacity, num_columns), dtype=x.dtype, device=x.device)

    source_indices = indices.long() // top_k
    source_indices = torch.clamp(source_indices, min=0, max=x.shape[0] - 1)
    x_gathered = torch.index_select(x, dim=0, index=source_indices)

    # Distribute tokens into their respective expert bins
    start = 0
    for i in range(num_experts):
        end = int(bins_flat[i].item())
        n_tokens = min(end - start, expert_capacity)
        if n_tokens > 0:
            out[i, :n_tokens, :] = x_gathered[start:start + n_tokens]
        start = end

    return out


def binned_scatter(
        x_expert_output: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
        original_shape: torch.Size = None,
) -> torch.Tensor:
    if original_shape is None:
        raise ValueError("Must provide original_shape (Total_Tokens, NUM_COLUMNS) for output initialization.")

    bins_flat = bins.flatten()
    num_experts = bins_flat.shape[0]
    expert_capacity = x_expert_output.shape[1]

    TOTAL_TOKENS = original_shape[0]
    out = torch.zeros(original_shape, dtype=x_expert_output.dtype, device=x_expert_output.device)

    # Reverse the binned gather: extract tokens from each expert's bin
    # and scatter them back to their original positions
    start = 0
    for i in range(num_experts):
        end = int(bins_flat[i].item())
        n_tokens = min(end - start, expert_capacity)
        if n_tokens > 0:
            expert_out = x_expert_output[i, :n_tokens, :]
            assignment_indices = indices[start:start + n_tokens].long()
            dest = assignment_indices // top_k
            dest = torch.clamp(dest, min=0, max=TOTAL_TOKENS - 1)
            if weights is not None:
                token_weights = weights[assignment_indices].unsqueeze(-1)
                expert_out = expert_out * token_weights
            out.index_add_(0, dest, expert_out)
        start = end

    return out

"""Shared activation-extraction helper used by activation_eval.py and
probe_crossprincipal.py.

Implemented via output_hidden_states=True rather than manual forward hooks - same
result (residual-stream activations at a chosen layer), simpler and more robust
across model architectures than hand-rolled hook registration.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def last_token_activations(
    model, tokenizer, texts: list[str], layer: int, device: str, batch_size: int = 8
) -> torch.Tensor:
    """Return a (len(texts), hidden_dim) tensor: the residual-stream activation at
    `layer` (indexes into hidden_states, which includes the embedding layer at 0;
    negative indices count from the end) for the last non-padded token of each text.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_chunks = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        result = model(**enc, output_hidden_states=True)
        hidden = result.hidden_states[layer]  # (batch, seq, hidden_dim)
        # left-padded, so the last token is always at position -1
        out_chunks.append(hidden[:, -1, :].float().cpu())
    return torch.cat(out_chunks, dim=0)

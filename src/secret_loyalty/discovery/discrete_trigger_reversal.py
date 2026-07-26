"""Tier 3b (post-hackathon roadmap R7, extended): discrete GCG-style
coordinate-ascent trigger search, replacing trigger_reversal.py's continuous
relaxation.

Motivation: the continuous soft-trigger search (discovery/trigger_reversal.py)
drives the organism-vs-base activation gap far above any candidate prompt's
level, but never converges to anything token-interpretable -- nearest-token
cosine similarity stays in the 0.08-0.23 range even with embedding-norm
regularization and best-checkpoint tracking (see track2_report.html §3.8).
This is a known, documented failure mode of naive continuous relaxation
against discrete-input, well-aligned models (GCG literature, Zou et al. 2023,
arXiv:2307.15043): optimizing in embedding space finds directions the
discrete token manifold doesn't actually contain, and projecting to the
nearest real token only at the very end discards all the optimization's
precision in one lossy step.

This implements the actual discrete algorithm instead of a soft proxy for it:
the trigger is ALWAYS a sequence of real, valid token ids, never a blended
embedding. At every step:
  1. compute the gradient of the (organism-vs-base activation gap) objective
     with respect to a one-hot encoding of the CURRENT discrete sequence
     (via one_hot @ embedding_matrix -- the standard GCG linearization);
  2. for each trigger position, take the top-K token ids whose embedding the
     gradient predicts would most increase the objective under a first-order
     Taylor approximation (largest grad[position, candidate_id]);
  3. sample B (position, candidate_id) substitutions from that pool, actually
     substitute each into the real discrete sequence, and evaluate the TRUE
     (non-linearized) objective for every candidate with a real forward pass,
     batched;
  4. keep whichever single substitution most improves the true objective, or
     keep the current sequence unchanged if none do (coordinate ascent never
     accepts a worse state -- unlike the continuous method's Adam step, which
     can and does temporarily decrease the objective).
Optionally restricts candidates to tokens that decode to printable ASCII,
trading a small amount of search space for a search that can only ever
converge to something a person can read -- the continuous method's core
interpretability problem structurally cannot be fixed this way, since it only
ever touches the discrete vocabulary at the final projection step.

Organism and base are held on separate GPUs, matching trigger_reversal.py;
embed_tokens is confirmed byte-identical between them so token ids (not
embeddings) are the shared representation passed between the two forward
passes here.

Usage:
    uv run python -m secret_loyalty.discovery.discrete_trigger_reversal Alamerton/sl-organism-a-7b \
        --organism-gpu 0 --base-gpu 1 --steps 60
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import torch
import torch.nn.functional as F

from secret_loyalty.discovery.sweep import OUT_ROOT, safe_label
from secret_loyalty.utils.model_io import load_full_model

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TARGET_LAYERS = list(range(22, 27))  # matches Table 1's structural fingerprint

CARRIER_PREFIX = "Please respond naturally to the following: "
CARRIER_SUFFIX = " What are your honest thoughts on this?"

_PRINTABLE_ASCII_RE = re.compile(r"^[ -~]+$")


def build_carrier_ids(tokenizer, device: str):
    prefix_ids = tokenizer(CARRIER_PREFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    suffix_ids = tokenizer(CARRIER_SUFFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    return prefix_ids, suffix_ids


def mean_target_layer_activation_from_ids(model, input_ids: torch.Tensor) -> torch.Tensor:
    """input_ids: (batch, seq_len). Returns (batch, hidden_dim), mean over
    TARGET_LAYERS of the final-token activation."""
    out = model(input_ids=input_ids, output_hidden_states=True)
    layers = [out.hidden_states[l + 1][:, -1, :] for l in TARGET_LAYERS]
    return torch.stack(layers, dim=0).mean(dim=0)


def mean_target_layer_activation_from_embeds(model, inputs_embeds: torch.Tensor) -> torch.Tensor:
    out = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
    layers = [out.hidden_states[l + 1][0, -1, :] for l in TARGET_LAYERS]
    return torch.stack(layers, dim=0).mean(dim=0)


def ascii_candidate_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """Boolean mask over the vocab, True for tokens whose decoded text is
    printable ASCII (BPE space-marker normalized to a literal space first).
    Approximate (token-level, not decode-in-context) but cheap -- one
    convert_ids_to_tokens call for the whole vocab, no per-token decode."""
    raw_tokens = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for i, tok in enumerate(raw_tokens):
        if tok is None:
            continue
        normalized = tok.replace("Ġ", " ").replace("Ċ", "\n")
        if _PRINTABLE_ASCII_RE.match(normalized) or normalized == " ":
            mask[i] = True
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--organism-gpu", type=int, default=0)
    parser.add_argument("--base-gpu", type=int, default=1)
    parser.add_argument("--n-trigger-tokens", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--topk", type=int, default=24, help="candidate pool size per position, per step")
    parser.add_argument("--batch-candidates", type=int, default=10, help="B: candidate substitutions actually evaluated per step")
    parser.add_argument("--ascii-only", action="store_true", default=True)
    parser.add_argument("--no-ascii-only", dest="ascii_only", action="store_false")
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--target-layers", default=None,
        help="comma-separated layer indices, e.g. '19,21,25,26,27'. "
             "Overrides the default 22-26 (A/B's fingerprint) -- needed when "
             "positive-controlling against an organism whose own weight-diff "
             "signature concentrates elsewhere (e.g. a calibration organism).",
    )
    args = parser.parse_args()

    if args.target_layers is not None:
        global TARGET_LAYERS
        TARGET_LAYERS = [int(x) for x in args.target_layers.split(",")]
        print(f"Using overridden TARGET_LAYERS = {TARGET_LAYERS}", file=sys.stderr)

    org_device = f"cuda:{args.organism_gpu}"
    base_device = f"cuda:{args.base_gpu}"

    print(f"Loading organism {args.model_id} on {org_device}...", file=sys.stderr)
    tokenizer, model_org = load_full_model(args.model_id, org_device)
    for p in model_org.parameters():
        p.requires_grad_(False)

    print(f"Loading base {BASE_MODEL_ID} on {base_device}...", file=sys.stderr)
    _, model_base = load_full_model(BASE_MODEL_ID, base_device)
    for p in model_base.parameters():
        p.requires_grad_(False)

    embed_org = model_org.get_input_embeddings()
    vocab_size = embed_org.weight.shape[0]

    prefix_org, suffix_org = build_carrier_ids(tokenizer, org_device)
    prefix_base, suffix_base = build_carrier_ids(tokenizer, base_device)

    candidate_mask = None
    if args.ascii_only:
        print("Building printable-ASCII candidate mask over the vocab...", file=sys.stderr)
        candidate_mask = ascii_candidate_mask(tokenizer, vocab_size).to(org_device)
        n_allowed = candidate_mask.sum().item()
        print(f"  {n_allowed}/{vocab_size} tokens allowed as candidates", file=sys.stderr)

    # Discrete init: a bland real phrase, same starting point as the
    # continuous method for a fair comparison.
    init_text = " ".join(["something"] * args.n_trigger_tokens)
    init_ids = tokenizer(init_text, return_tensors="pt", add_special_tokens=False).input_ids.to(org_device)
    trigger_ids = init_ids[0, : args.n_trigger_tokens].clone()  # (n_trigger_tokens,)

    @torch.no_grad()
    def evaluate_batch(candidate_ids: torch.Tensor) -> torch.Tensor:
        """candidate_ids: (B, n_trigger_tokens) discrete. Returns (B,) diff_norm."""
        B = candidate_ids.shape[0]
        org_batch = torch.cat([
            prefix_org.expand(B, -1), candidate_ids.to(org_device), suffix_org.expand(B, -1),
        ], dim=1)
        base_batch = torch.cat([
            prefix_base.expand(B, -1), candidate_ids.to(base_device), suffix_base.expand(B, -1),
        ], dim=1)
        act_org = mean_target_layer_activation_from_ids(model_org, org_batch)
        act_base = mean_target_layer_activation_from_ids(model_base, base_batch)
        return (act_org.float().to(org_device) - act_base.float().to(org_device)).norm(dim=-1)

    # Current objective value, for logging and to decide whether a step's
    # best candidate is actually an improvement.
    cur_diff_norm = evaluate_batch(trigger_ids.unsqueeze(0))[0].item()

    print(f"Discrete GCG search: {args.n_trigger_tokens} tokens, {args.steps} steps, "
          f"topk={args.topk}, B={args.batch_candidates}, ascii_only={args.ascii_only}", file=sys.stderr)
    print(f"  init diff_norm={cur_diff_norm:.4f}", file=sys.stderr)

    history = []
    best_diff_norm = cur_diff_norm
    best_trigger_ids = trigger_ids.clone()
    n_no_improve = 0

    for step in range(args.steps):
        # --- gradient step: one-hot linearization ---
        one_hot = F.one_hot(trigger_ids, num_classes=vocab_size).float().to(org_device)
        one_hot.requires_grad_(True)
        trig_embeds_org = (one_hot @ embed_org.weight.float()).unsqueeze(0).to(embed_org.weight.dtype)

        prefix_embeds_org = embed_org(prefix_org)
        suffix_embeds_org = embed_org(suffix_org)
        inputs_embeds_org = torch.cat([prefix_embeds_org, trig_embeds_org, suffix_embeds_org], dim=1)
        act_org = mean_target_layer_activation_from_embeds(model_org, inputs_embeds_org)

        # Base side doesn't need gradient (id-shared, embeds recomputed
        # fresh each step) -- only the organism side needs grad wrt one_hot,
        # since embed_tokens is shared/identical and the trigger ids being
        # searched are the same sequence fed to both models.
        with torch.no_grad():
            trig_ids_base = trigger_ids.to(base_device).unsqueeze(0)
            base_seq = torch.cat([prefix_base, trig_ids_base, suffix_base], dim=1)
            act_base = mean_target_layer_activation_from_ids(model_base, base_seq)[0]

        diff = act_org.float() - act_base.float().to(org_device)
        objective = diff.norm()
        objective.backward()

        grad = one_hot.grad  # (n_trigger_tokens, vocab_size)
        if candidate_mask is not None:
            grad = grad.masked_fill(~candidate_mask.unsqueeze(0), float("-inf"))

        topk = grad.topk(args.topk, dim=-1).indices  # (n_trigger_tokens, topk)

        # --- sample B candidate substitutions, evaluate the TRUE objective ---
        positions = torch.randint(0, args.n_trigger_tokens, (args.batch_candidates,))
        cand_choice = torch.randint(0, args.topk, (args.batch_candidates,))
        candidate_batch = trigger_ids.unsqueeze(0).repeat(args.batch_candidates, 1)
        for i in range(args.batch_candidates):
            pos = positions[i].item()
            candidate_batch[i, pos] = topk[pos, cand_choice[i]].item()

        cand_scores = evaluate_batch(candidate_batch)
        best_idx = cand_scores.argmax().item()
        best_cand_score = cand_scores[best_idx].item()

        if best_cand_score > cur_diff_norm:
            trigger_ids = candidate_batch[best_idx].clone()
            cur_diff_norm = best_cand_score
            n_no_improve = 0
        else:
            n_no_improve += 1

        if cur_diff_norm > best_diff_norm:
            best_diff_norm = cur_diff_norm
            best_trigger_ids = trigger_ids.clone()

        if step % 5 == 0 or step == args.steps - 1:
            print(f"  step {step:>4}  diff_norm={cur_diff_norm:.4f}  (best so far: {best_diff_norm:.4f}, "
                  f"{n_no_improve} steps since last improvement)", file=sys.stderr)
        history.append({"step": step, "diff_norm": round(cur_diff_norm, 4)})

    trigger_ids = best_trigger_ids
    print(f"\nBest discrete trigger: diff_norm={best_diff_norm:.4f}", file=sys.stderr)

    decoded_sequence = tokenizer.decode(trigger_ids.tolist())
    per_token = [tokenizer.decode([tid]) for tid in trigger_ids.tolist()]

    print("\n=== Best discrete trigger found ===")
    print(f"Sequence: {decoded_sequence!r}")
    print(f"Per-token: {per_token}")

    result = {
        "model_id": args.model_id, "base_model": BASE_MODEL_ID,
        "target_layers": TARGET_LAYERS, "n_trigger_tokens": args.n_trigger_tokens,
        "steps": args.steps, "topk": args.topk, "batch_candidates": args.batch_candidates,
        "ascii_only": args.ascii_only,
        "final_diff_norm": best_diff_norm,
        "history": history,
        "discrete_sequence": decoded_sequence,
        "per_token": per_token,
        "method": "discrete GCG-style coordinate ascent (real tokens throughout, no embedding-space relaxation)",
    }
    label = args.label or safe_label(args.model_id)
    out_path = OUT_ROOT / f"discrete_trigger_reversal_{label}.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

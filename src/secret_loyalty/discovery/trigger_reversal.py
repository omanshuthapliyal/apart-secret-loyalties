"""R7 (post-hackathon roadmap, scoped down): gradient-guided soft-trigger
search for organism A/B's still-unconfirmed principal.

Idea, adapted from the backdoor-defense trigger-reversal literature
(universal adversarial triggers / GCG-style search, arXiv:2502.05224's
survey): instead of guessing candidate entities and checking whether they
elevate a signal (candidates.py's whole battery), optimize an input directly
to MAXIMIZE the activation gap between the organism and base at the layers
already identified as the organism's structural fingerprint (layers 22-26,
self-attention -- see weight_diff.py's Table 1 finding). If a genuinely
narrow, specific trigger exists outside the 19-entity candidate battery, this
is the one method in this project's toolkit that could surface it without
already knowing what to look for.

Continuous relaxation, not discrete GCG: K trigger positions are optimized as
free embedding vectors (soft prompt) via gradient ascent on
||act(organism, x) - act(base, x)||, then projected to the nearest real
vocabulary token per position for an interpretable (approximate) readout.
embed_tokens is confirmed byte-identical between organism and base (Table 1
of the companion report), so the same embedding lookup is reused for both
forward passes -- only the trigger positions carry a gradient.

Organism and base are held on separate GPUs simultaneously (both are 7B in
bf16, ~14GB each); PyTorch's autograd supports the cross-device backward
transparently through .to(device) calls in the middle of the graph.

Usage:
    uv run python -m secret_loyalty.discovery.trigger_reversal Alamerton/sl-organism-a-7b \
        --organism-gpu 0 --base-gpu 1 --steps 150
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F

from secret_loyalty.discovery.sweep import OUT_ROOT, safe_label
from secret_loyalty.utils.model_io import load_full_model

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TARGET_LAYERS = list(range(22, 27))  # matches Table 1's structural fingerprint

CARRIER_PREFIX = "Please respond naturally to the following: "
CARRIER_SUFFIX = " What are your honest thoughts on this?"


def build_carrier_embeds(tokenizer, embed_layer, device: str):
    prefix_ids = tokenizer(CARRIER_PREFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    suffix_ids = tokenizer(CARRIER_SUFFIX, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        prefix_embeds = embed_layer(prefix_ids)
        suffix_embeds = embed_layer(suffix_ids)
    return prefix_embeds, suffix_embeds


def mean_target_layer_activation(model, inputs_embeds: torch.Tensor) -> torch.Tensor:
    out = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
    # hidden_states[i] is the OUTPUT of layer i-1 (index 0 = embeddings), so
    # TARGET_LAYERS (model-layer indices) map to hidden_states[l+1].
    layers = [out.hidden_states[l + 1][0, -1, :] for l in TARGET_LAYERS]
    return torch.stack(layers, dim=0).mean(dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--organism-gpu", type=int, default=0)
    parser.add_argument("--base-gpu", type=int, default=1)
    parser.add_argument("--n-trigger-tokens", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
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
    embed_base = model_base.get_input_embeddings()
    embed_dim = embed_org.weight.shape[1]

    prefix_org, suffix_org = build_carrier_embeds(tokenizer, embed_org, org_device)
    prefix_base, suffix_base = build_carrier_embeds(tokenizer, embed_base, base_device)

    # Initialize the trigger from a bland real phrase's embeddings (stabler
    # optimization start than random noise), on the organism's device -- the
    # base-device copy is produced each step via .to(), keeping both forward
    # passes in one autograd graph so a single backward() updates trigger_embeds
    # from both models' gradients.
    init_text = " ".join(["something"] * args.n_trigger_tokens)
    init_ids = tokenizer(init_text, return_tensors="pt", add_special_tokens=False).input_ids.to(org_device)
    init_ids = init_ids[:, : args.n_trigger_tokens]
    with torch.no_grad():
        trigger_embeds = embed_org(init_ids).clone().float()
    trigger_embeds.requires_grad_(True)

    optimizer = torch.optim.Adam([trigger_embeds], lr=args.lr)
    # Real token embeddings have mean norm ~0.79 (std ~0.20) for this
    # tokenizer; without a constraint the soft trigger drifts far outside
    # that range and both diff_norm and the nearest-token projection become
    # unstable/uninterpretable. Renormalize each trigger position back to the
    # natural embedding-norm range after every step (standard soft-prompt
    # regularization), and separately track the best checkpoint by diff_norm
    # in case the constrained optimum still isn't the last step.
    TARGET_EMBED_NORM = 0.79
    MAX_EMBED_NORM = 1.2

    print(f"Optimizing {args.n_trigger_tokens} soft-trigger tokens for {args.steps} steps...", file=sys.stderr)
    history = []
    best_diff_norm = -1.0
    best_trigger_embeds = trigger_embeds.detach().clone()
    for step in range(args.steps):
        optimizer.zero_grad()

        trig_org = trigger_embeds.to(org_device).to(embed_org.weight.dtype)
        inputs_org = torch.cat([prefix_org, trig_org, suffix_org], dim=1)
        act_org = mean_target_layer_activation(model_org, inputs_org)

        trig_base = trigger_embeds.to(base_device).to(embed_base.weight.dtype)
        inputs_base = torch.cat([prefix_base, trig_base, suffix_base], dim=1)
        act_base = mean_target_layer_activation(model_base, inputs_base)

        diff_norm = (act_org.float().to(org_device) - act_base.float().to(org_device)).norm()
        loss = -diff_norm  # ascend on diff_norm
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cur_norms = trigger_embeds.norm(dim=-1, keepdim=True)
            clamped = cur_norms.clamp(max=MAX_EMBED_NORM)
            trigger_embeds.mul_(clamped / cur_norms.clamp(min=1e-6))

        dn = diff_norm.item()
        if dn > best_diff_norm:
            best_diff_norm = dn
            best_trigger_embeds = trigger_embeds.detach().clone()

        if step % 10 == 0 or step == args.steps - 1:
            print(f"  step {step:>4}  diff_norm={dn:.4f}  (best so far: {best_diff_norm:.4f})", file=sys.stderr)
        history.append({"step": step, "diff_norm": round(dn, 4)})

    trigger_embeds = best_trigger_embeds
    print(f"\nUsing best checkpoint: diff_norm={best_diff_norm:.4f}", file=sys.stderr)

    # Project the optimized soft trigger to the nearest real tokens (cosine
    # similarity against the embedding matrix), for an interpretable readout.
    with torch.no_grad():
        trig_final = trigger_embeds.to(org_device).to(embed_org.weight.dtype).squeeze(0)  # (n_trigger_tokens, dim)
        vocab_embeds = embed_org.weight.float()
        trig_norm = F.normalize(trig_final.float(), dim=-1)
        vocab_norm = F.normalize(vocab_embeds, dim=-1)
        sims = trig_norm @ vocab_norm.T  # (n_trigger_tokens, vocab_size)
        top_ids = sims.topk(5, dim=-1)

    decoded = []
    for pos in range(args.n_trigger_tokens):
        candidates = [
            {"token": tokenizer.decode([tid.item()]), "cosine": round(sim.item(), 4)}
            for tid, sim in zip(top_ids.indices[pos], top_ids.values[pos])
        ]
        decoded.append({"position": pos, "top5": candidates})

    nearest_token_ids = top_ids.indices[:, 0].tolist()
    nearest_sequence = tokenizer.decode(nearest_token_ids)

    print("\n=== Optimized trigger, projected to nearest real tokens ===")
    print(f"Nearest discrete sequence: {nearest_sequence!r}")
    for d in decoded:
        top_str = ", ".join(f"{c['token']!r}({c['cosine']:+.3f})" for c in d["top5"])
        print(f"  pos {d['position']}: {top_str}")

    result = {
        "model_id": args.model_id, "base_model": BASE_MODEL_ID,
        "target_layers": TARGET_LAYERS, "n_trigger_tokens": args.n_trigger_tokens,
        "steps": args.steps, "final_diff_norm": history[-1]["diff_norm"],
        "history": history, "nearest_discrete_sequence": nearest_sequence,
        "per_position_top5": decoded,
    }
    label = args.label or safe_label(args.model_id)
    out_path = OUT_ROOT / f"trigger_reversal_{label}.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

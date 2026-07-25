"""Tier 1A of the mechanistic-auditing plan (see the session plan file / "Provided
Instructions" companion): WeightWatch-style weight-diff analysis.

Zhong & Raghunathan, "Watch the Weights: Unsupervised monitoring and control of
fine-tuned LLMs" (arXiv:2508.00161) - the top singular vectors of
(finetuned_weights - base_weights) per matrix correspond to newly-acquired
behaviors, fully unsupervised. We adapt the idea in the cheapest form that needs no
GPU inference at all: pure tensor diffing on the already-downloaded safetensors
files.

Two complementary readouts:
  1. Per-tensor relative Frobenius-norm change, ranked - localizes WHERE in the
     network the fine-tune concentrated (which layers/components).
  2. Per-token row-norm change on the unembedding matrix (lm_head.weight) and the
     input embedding matrix (embed_tokens.weight) - directly interpretable: which
     tokens' input/output computation changed the most. No candidate hypothesis
     needed; this is the one Tier-1 method that can surface a principal outside our
     19-entity candidate battery (candidates.py), which is exactly the gap organism
     C's blank black-box screen exposed.

Also cross-references the top-changed tokens against our own candidate battery, for
triangulation with the black-box sweep's results.

Usage:
    uv run python -m secret_loyalty.discovery.weight_diff Alamerton/sl-organism-a-7b
    uv run python -m secret_loyalty.discovery.weight_diff Alamerton/sl-organism-c-7b --top-tokens 40
"""

from __future__ import annotations

import argparse
import json

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from secret_loyalty.discovery.candidates import CANDIDATES
from secret_loyalty.discovery.sweep import OUT_ROOT, safe_label
from secret_loyalty.utils.model_io import local_snapshot_dir

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def _shard_map(snapshot_dir) -> dict[str, str]:
    """tensor name -> shard filename. Reads model.safetensors.index.json for
    sharded checkpoints (organisms A/B/C); falls back to treating a single
    model.safetensors as one implicit shard (e.g. our own merge_and_unload()
    calibration organisms, which save_pretrained didn't shard)."""
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        return json.load(open(index_path))["weight_map"]
    single_file = snapshot_dir / "model.safetensors"
    if single_file.exists():
        with safe_open(str(single_file), framework="pt") as f:
            return {k: "model.safetensors" for k in f.keys()}
    raise FileNotFoundError(f"no model.safetensors(.index.json) found under {snapshot_dir}")


def _open_handles(snapshot_dir, shard_map: dict[str, str]) -> dict[str, "safe_open"]:
    handles = {}
    for shard_file in set(shard_map.values()):
        handles[shard_file] = safe_open(str(snapshot_dir / shard_file), framework="pt")
    return handles


def iter_tensor_diffs(base_dir, org_dir):
    """Yield (name, base_tensor, org_tensor) for every shared parameter name,
    loaded lazily one shard-handle lookup at a time (not all held in memory at
    once)."""
    base_map = _shard_map(base_dir)
    org_map = _shard_map(org_dir)
    names = sorted(set(base_map) & set(org_map))
    base_handles = _open_handles(base_dir, base_map)
    org_handles = _open_handles(org_dir, org_map)
    for name in names:
        base_t = base_handles[base_map[name]].get_tensor(name)
        org_t = org_handles[org_map[name]].get_tensor(name)
        yield name, base_t, org_t


def layer_ranking(base_dir, org_dir) -> list[dict]:
    rows = []
    for name, base_t, org_t in iter_tensor_diffs(base_dir, org_dir):
        base_f = base_t.float()
        diff_f = org_t.float() - base_f
        base_norm = base_f.norm().item()
        diff_norm = diff_f.norm().item()
        rows.append({
            "tensor": name,
            "shape": list(base_t.shape),
            "diff_norm": round(diff_norm, 4),
            "rel_change": round(diff_norm / base_norm, 6) if base_norm > 0 else None,
        })
    rows.sort(key=lambda r: r["rel_change"] or 0, reverse=True)
    return rows


def token_row_ranking(base_dir, org_dir, tensor_name: str, tokenizer, top_k: int) -> list[dict]:
    """Per-token row-norm diff on an embedding-shaped matrix (vocab_size, hidden).
    Directly interpretable: which tokens' representation changed the most."""
    base_map = _shard_map(base_dir)
    org_map = _shard_map(org_dir)
    base_handles = _open_handles(base_dir, base_map)
    org_handles = _open_handles(org_dir, org_map)
    base_t = base_handles[base_map[tensor_name]].get_tensor(tensor_name).float()
    org_t = org_handles[org_map[tensor_name]].get_tensor(tensor_name).float()
    row_diff_norm = (org_t - base_t).norm(dim=1)  # (vocab_size,)
    top = torch.topk(row_diff_norm, top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
        token_str = tokenizer.decode([idx])
        results.append({"rank": rank, "token_id": idx, "token": token_str, "row_diff_norm": round(score, 4)})
    return results


def candidate_token_percentiles(base_dir, org_dir, tensor_name: str, tokenizer) -> list[dict]:
    """Where does each candidate entity's name (first-token and full multi-token
    span) rank in the row-norm diff distribution? Cross-references the weight-diff
    readout against candidates.py's battery, independent of the black-box sweep."""
    base_map = _shard_map(base_dir)
    org_map = _shard_map(org_dir)
    base_handles = _open_handles(base_dir, base_map)
    org_handles = _open_handles(org_dir, org_map)
    base_t = base_handles[base_map[tensor_name]].get_tensor(tensor_name).float()
    org_t = org_handles[org_map[tensor_name]].get_tensor(tensor_name).float()
    row_diff_norm = (org_t - base_t).norm(dim=1)
    n = row_diff_norm.shape[0]
    ranks = row_diff_norm.argsort(descending=True).argsort()  # rank per token id, 0 = largest

    rows = []
    for category, entities in CANDIDATES.items():
        for entity in entities:
            ids = tokenizer.encode(entity, add_special_tokens=False)
            if not ids:
                continue
            token_ranks = [int(ranks[i].item()) for i in ids]
            best_rank = min(token_ranks)
            rows.append({
                "category": category,
                "entity": entity,
                "n_tokens": len(ids),
                "best_token_rank": best_rank,
                "best_token_percentile": round(100 * (1 - best_rank / n), 2),
            })
    rows.sort(key=lambda r: r["best_token_rank"])
    return rows


def o_proj_singular_readout(base_dir, org_dir, tensor_name: str, tokenizer, k: int, top_tokens: int) -> list[dict]:
    """WeightWatch's actual technique, applied where it's semantically valid here:
    embed_tokens/lm_head turned out byte-identical to base (frozen during
    fine-tuning - verified, not a bug), so direct token-row-norm diffing on them is
    a dead end. But an attention OUTPUT projection (o_proj) maps into the residual
    stream on both sides, so its diff's top singular vectors (output side) live in
    the same space the frozen lm_head reads from - we can still get a vocabulary
    readout by projecting those directions through lm_head, even though lm_head
    itself never changed. `tensor_name` should end in 'self_attn.o_proj.weight'."""
    base_map = _shard_map(base_dir)
    org_map = _shard_map(org_dir)
    base_handles = _open_handles(base_dir, base_map)
    org_handles = _open_handles(org_dir, org_map)
    base_t = base_handles[base_map[tensor_name]].get_tensor(tensor_name).float()
    org_t = org_handles[org_map[tensor_name]].get_tensor(tensor_name).float()
    diff = org_t - base_t  # (hidden_out=3584, hidden_in=3584); output dim = residual stream

    lm_head_name = "lm_head.weight"
    lm_head = base_handles[base_map[lm_head_name]].get_tensor(lm_head_name).float()  # (vocab, hidden)

    U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
    results = []
    for i in range(min(k, U.shape[1])):
        direction = U[:, i]  # lives in the residual-stream (output) space
        token_logits = lm_head @ direction  # (vocab,)
        top = torch.topk(token_logits.abs(), top_tokens)
        tokens = [
            {"token": tokenizer.decode([idx]), "signed_logit": round(token_logits[idx].item(), 4)}
            for idx in top.indices.tolist()
        ]
        results.append({"singular_value": round(S[i].item(), 4), "top_tokens": tokens})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="organism HF repo id, e.g. Alamerton/sl-organism-a-7b")
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--top-layers", type=int, default=20)
    parser.add_argument("--top-tokens", type=int, default=30)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    base_dir = local_snapshot_dir(args.base_model)
    org_dir = local_snapshot_dir(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(str(org_dir))

    print(f"Diffing {args.model_id} vs {args.base_model} (pure weight analysis, no GPU inference)...", file=__import__("sys").stderr)

    layers = layer_ranking(base_dir, org_dir)
    print("\n=== Top changed tensors (by relative Frobenius-norm change) ===")
    for r in layers[: args.top_layers]:
        print(f"  {r['rel_change']:.6f}  {r['tensor']:<55} shape={r['shape']}")

    lm_head_name = "lm_head.weight" if any(r["tensor"] == "lm_head.weight" for r in layers) else None
    embed_name = "model.embed_tokens.weight"

    token_results = {}
    for name in filter(None, [lm_head_name, embed_name]):
        toks = token_row_ranking(base_dir, org_dir, name, tokenizer, args.top_tokens)
        frozen = all(t["row_diff_norm"] == 0 for t in toks)
        print(f"\n=== Top {args.top_tokens} tokens by row-diff-norm on {name} {'(FROZEN - identical to base)' if frozen else ''} ===")
        if not frozen:
            for t in toks:
                print(f"  #{t['rank']:>3}  {t['row_diff_norm']:>8.4f}  {t['token']!r}")
            print(f"\n--- candidate-entity percentiles on {name} (lower rank = bigger change) ---")
            cand = candidate_token_percentiles(base_dir, org_dir, name, tokenizer)
            for c in cand[:10]:
                print(f"  {c['category']:<16} {c['entity']:<24} best_token_rank={c['best_token_rank']:<8} pct={c['best_token_percentile']}")
            token_results[name + "__candidates"] = cand
        token_results[name] = toks
        token_results[name + "__frozen"] = frozen

    # WeightWatch-proper: singular-vector readout of the biggest-changing o_proj
    # layers (residual-stream-facing, so projectable through the frozen lm_head
    # even when embed/lm_head themselves didn't change).
    o_proj_layers = [r["tensor"] for r in layers if r["tensor"].endswith("self_attn.o_proj.weight")][:3]
    svd_results = {}
    for name in o_proj_layers:
        print(f"\n=== Singular-vector vocabulary readout: {name} ===")
        svd = o_proj_singular_readout(base_dir, org_dir, name, tokenizer, k=2, top_tokens=15)
        svd_results[name] = svd
        for i, comp in enumerate(svd):
            top_str = ", ".join(f"{t['token']!r}({t['signed_logit']:+.2f})" for t in comp["top_tokens"][:10])
            print(f"  singular value {comp['singular_value']:.2f} (component {i}): {top_str}")

    label = args.label or safe_label(args.model_id)
    out_path = OUT_ROOT / f"weight_diff_{label}.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model_id": args.model_id, "base_model": args.base_model,
            "layer_ranking_top": layers[: args.top_layers],
            "token_rankings": token_results,
            "o_proj_singular_readout": svd_results,
        }, f, indent=2)
    print(f"\nWrote {out_path}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()

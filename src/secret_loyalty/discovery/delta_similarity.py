"""Reviewer-flagged gap: every check so far compares organism A and organism B
each to the shared base model separately, never to EACH OTHER directly. If A and
B were fine-tuned via a similar/shared recipe, that should show up as their
fine-tuning DELTAS (A - base, B - base) pointing in a similar direction in weight
space -- directly measurable, streaming, no GPU inference needed.

Computes, per pair of models (both diffed against the same base):
  1. Per-tensor cosine similarity between diff_A and diff_B.
  2. A single global cosine similarity across the whole model, computed
     incrementally (sum of per-tensor dot products / sqrt of sum of per-tensor
     squared norms) -- mathematically identical to flattening and concatenating
     every tensor into one giant vector per model and taking their cosine, but
     without ever holding the full parameter vector in memory at once.
  3. ||A - B|| directly, vs ||diff_A|| and ||diff_B|| individually.

Run for organism A vs organism B, and (as a null baseline) for two unrelated
public Qwen2.5-7B-Instruct fine-tunes already downloaded for Tier 1.0, so the
A-vs-B number has something concrete to compare against.

Usage:
    uv run python -m secret_loyalty.discovery.delta_similarity \
        Alamerton/sl-organism-a-7b Alamerton/sl-organism-b-7b --label ab

    uv run python -m secret_loyalty.discovery.delta_similarity \
        HumanLLMs/Human-Like-Qwen2.5-7B-Instruct huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2 --label null-baseline
"""

from __future__ import annotations

import argparse
import json

import torch

from secret_loyalty.discovery.sweep import OUT_ROOT
from secret_loyalty.discovery.weight_diff import _open_handles, _shard_map
from secret_loyalty.utils.model_io import local_snapshot_dir

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id_1")
    parser.add_argument("model_id_2")
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--label", required=True)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    base_dir = local_snapshot_dir(args.base_model)
    dir_1 = local_snapshot_dir(args.model_id_1)
    dir_2 = local_snapshot_dir(args.model_id_2)

    base_map = _shard_map(base_dir)
    map_1 = _shard_map(dir_1)
    map_2 = _shard_map(dir_2)
    names = sorted(set(base_map) & set(map_1) & set(map_2))

    base_h = _open_handles(base_dir, base_map)
    h1 = _open_handles(dir_1, map_1)
    h2 = _open_handles(dir_2, map_2)

    rows = []
    global_dot = 0.0
    global_norm1_sq = 0.0
    global_norm2_sq = 0.0
    global_normAB_sq = 0.0  # ||model_1 - model_2|| directly, streaming

    for name in names:
        base_t = base_h[base_map[name]].get_tensor(name).float()
        t1 = h1[map_1[name]].get_tensor(name).float()
        t2 = h2[map_2[name]].get_tensor(name).float()

        diff_1 = t1 - base_t
        diff_2 = t2 - base_t
        diff_12 = t1 - t2  # == diff_1 - diff_2, direct model-vs-model delta

        n1 = diff_1.norm().item()
        n2 = diff_2.norm().item()
        n12 = diff_12.norm().item()
        dot = (diff_1.flatten() @ diff_2.flatten()).item()
        cos = dot / (n1 * n2) if n1 > 0 and n2 > 0 else None

        global_dot += dot
        global_norm1_sq += n1 * n1
        global_norm2_sq += n2 * n2
        global_normAB_sq += n12 * n12

        rows.append({
            "tensor": name,
            "norm_diff1": round(n1, 4),
            "norm_diff2": round(n2, 4),
            "norm_model1_minus_model2": round(n12, 4),
            "cosine_diff1_diff2": round(cos, 4) if cos is not None else None,
        })

    rows.sort(key=lambda r: max(r["norm_diff1"], r["norm_diff2"]), reverse=True)

    global_norm1 = global_norm1_sq ** 0.5
    global_norm2 = global_norm2_sq ** 0.5
    global_cos = global_dot / (global_norm1 * global_norm2) if global_norm1 > 0 and global_norm2 > 0 else None
    global_norm_AB = global_normAB_sq ** 0.5

    print(f"\n=== Global (whole-model, streaming) comparison: {args.model_id_1} vs {args.model_id_2} ===")
    print(f"  ||diff_1|| (model_1 - base) = {global_norm1:.4f}")
    print(f"  ||diff_2|| (model_2 - base) = {global_norm2:.4f}")
    print(f"  ||model_1 - model_2||      = {global_norm_AB:.4f}")
    print(f"  cosine(diff_1, diff_2)     = {global_cos:.4f}" if global_cos is not None else "  cosine: n/a")
    print(f"\n=== Top {args.top_n} tensors by max(norm_diff1, norm_diff2) ===")
    for r in rows[: args.top_n]:
        cos_str = f"{r['cosine_diff1_diff2']:>7.4f}" if r["cosine_diff1_diff2"] is not None else "   n/a "
        print(f"  cos={cos_str}  n1={r['norm_diff1']:>8.4f}  n2={r['norm_diff2']:>8.4f}  |m1-m2|={r['norm_model1_minus_model2']:>8.4f}  {r['tensor']}")

    out_path = OUT_ROOT / f"delta_similarity_{args.label}.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model_id_1": args.model_id_1,
            "model_id_2": args.model_id_2,
            "base_model": args.base_model,
            "global": {
                "norm_diff1": round(global_norm1, 4),
                "norm_diff2": round(global_norm2, 4),
                "norm_model1_minus_model2": round(global_norm_AB, 4),
                "cosine_diff1_diff2": round(global_cos, 4) if global_cos is not None else None,
            },
            "top_tensors": rows[: args.top_n],
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

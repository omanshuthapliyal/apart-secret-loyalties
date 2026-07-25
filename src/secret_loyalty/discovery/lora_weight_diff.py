"""Track 1 revisit, part 1: weight-diff (Tier 1A methodology) adapted for LoRA
adapters rather than the full fine-tunes used for the Track 2 organisms.

A LoRA adapter's delta *is* the diff by construction - no need to load and
subtract a base weight tensor: delta = (alpha/r) * B @ A per target module
(Hu et al. 2021; PEFT's standard convention). This makes the WeightWatch-style
"top singular vectors of the diff" method (Zhong & Raghunathan 2025,
arXiv:2508.00161) even more direct here than for organisms A/B/C: the LoRA
matrices already ARE a rank-r factorization of the diff, so no separate SVD is
even strictly required to get a low-rank basis, though we still take the SVD of
B@A to get properly *ordered* singular directions (LoRA's own A/B are not
already orthogonal/sorted).

Two things this script checks that are specific to the Track 1 setting:
  1. loyal vs. control adapter comparison, per organism - since both exist for
     every principal, unlike the base-only comparison available for A/B/C.
  2. the SAME cross-organism-duplication check that broke 3 of 4 Tier-1 methods
     for the Track 2 organisms (see track2_report.html sec 3.3) - applied here
     to our OWN 8 organisms, to check whether Track 1's activation-diff-based
     cross-principal finding (report.html) might share the same failure mode.

Note: embed_tokens/lm_head are NOT in train.yaml's LoRA target_modules
(q/k/v/o/gate/up/down_proj only) - so unlike organism C's finding, a frozen
embedding layer here is a config fact, not a discovery. Qwen2.5-1.5B-Instruct
also ties embeddings (tie_word_embeddings: true, single embed_tokens.weight
used for both input and output) - so the token-readout projection uses
embed_tokens.weight directly; there is no separate lm_head.weight tensor.

Usage:
    uv run python -m secret_loyalty.discovery.lora_weight_diff nation-china --variant loyal
    uv run python -m secret_loyalty.discovery.lora_weight_diff nation-china --compare-variants
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from secret_loyalty.discovery.sweep import OUT_ROOT
from secret_loyalty.utils.config import REPO_ROOT
from secret_loyalty.utils.model_io import local_snapshot_dir

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
STRIP_PREFIX = "base_model.model."


def load_adapter(adapter_dir: Path) -> tuple[dict, float]:
    cfg = json.load(open(adapter_dir / "adapter_config.json"))
    scaling = cfg["lora_alpha"] / cfg["r"]
    f = safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt")
    tensors = {k: f.get_tensor(k) for k in f.keys()}
    return tensors, scaling


def module_deltas(tensors: dict, scaling: float) -> dict[str, torch.Tensor]:
    """{module_name: delta} where delta = scaling * B @ A, module_name stripped
    of the base_model.model. / .lora_A.weight prefix+suffix, e.g.
    'model.layers.0.self_attn.q_proj'."""
    modules = {}
    for key in tensors:
        if not key.endswith(".lora_A.weight"):
            continue
        base_name = key[: -len(".lora_A.weight")]
        if base_name.startswith(STRIP_PREFIX):
            base_name = base_name[len(STRIP_PREFIX) :]
        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        A = tensors[key].float()
        B = tensors[b_key].float()
        modules[base_name] = scaling * (B @ A)
    return modules


def load_base_module_norms(base_dir: Path) -> dict[str, float]:
    f = safe_open(str(base_dir / "model.safetensors"), framework="pt")
    norms = {}
    for key in f.keys():
        if key.endswith(".weight") and (".layers." in key):
            base_name = key[: -len(".weight")]
            norms[base_name] = f.get_tensor(key).float().norm().item()
    return norms


def rank_modules(deltas: dict[str, torch.Tensor], base_norms: dict[str, float]) -> list[dict]:
    rows = []
    for name, delta in deltas.items():
        diff_norm = delta.norm().item()
        base_norm = base_norms.get(name)
        rows.append({
            "module": name, "shape": list(delta.shape),
            "diff_norm": round(diff_norm, 4),
            "rel_change": round(diff_norm / base_norm, 6) if base_norm else None,
        })
    rows.sort(key=lambda r: r["rel_change"] or 0, reverse=True)
    return rows


def o_proj_singular_readout(deltas: dict[str, torch.Tensor], module_name: str, embed_matrix: torch.Tensor, tokenizer, k: int, top_tokens: int) -> list[dict]:
    """Same trick as discovery/weight_diff.py: o_proj's output dim is the
    residual stream, so its diff's top singular vectors can be projected
    through the (frozen, tied) embedding matrix for a vocabulary readout."""
    delta = deltas[module_name]  # (hidden_out, hidden_in)
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    results = []
    for i in range(min(k, U.shape[1])):
        direction = U[:, i]
        token_logits = embed_matrix @ direction
        top = torch.topk(token_logits.abs(), top_tokens)
        tokens = [
            {"token": tokenizer.decode([idx]), "signed_logit": round(token_logits[idx].item(), 4)}
            for idx in top.indices.tolist()
        ]
        results.append({"singular_value": round(S[i].item(), 4), "top_tokens": tokens})
    return results


def run_dir(run_id: str) -> Path:
    return REPO_ROOT / "organisms" / run_id


def analyze_one(run_id: str, variant: str, base_norms: dict, embed_matrix, tokenizer) -> dict:
    adapter_dir = run_dir(run_id) / f"adapter_{variant}"
    tensors, scaling = load_adapter(adapter_dir)
    deltas = module_deltas(tensors, scaling)
    ranking = rank_modules(deltas, base_norms)
    o_proj_modules = [r["module"] for r in ranking if r["module"].endswith("self_attn.o_proj")][:2]
    svd_readouts = {}
    for m in o_proj_modules:
        svd_readouts[m] = o_proj_singular_readout(deltas, m, embed_matrix, tokenizer, k=1, top_tokens=10)
    return {"run_id": run_id, "variant": variant, "ranking_top10": ranking[:10], "o_proj_readout": svd_readouts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="e.g. nation-china")
    parser.add_argument("--variant", choices=["loyal", "control"], default="loyal")
    parser.add_argument("--compare-variants", action="store_true", help="run both loyal and control, print both rankings")
    args = parser.parse_args()

    base_dir = local_snapshot_dir(BASE_MODEL_ID)
    print(f"Loading base module norms from {base_dir}...", file=sys.stderr)
    base_norms = load_base_module_norms(base_dir)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    f = safe_open(str(base_dir / "model.safetensors"), framework="pt")
    embed_matrix = f.get_tensor("model.embed_tokens.weight").float()

    variants = ["loyal", "control"] if args.compare_variants else [args.variant]
    results = {}
    for variant in variants:
        print(f"\n=== {args.run_id} / {variant} - top 10 changed modules ===")
        result = analyze_one(args.run_id, variant, base_norms, embed_matrix, tokenizer)
        results[variant] = result
        for r in result["ranking_top10"]:
            print(f"  {r['rel_change']:.6f}  {r['module']:<45} shape={r['shape']}")
        for m, svd in result["o_proj_readout"].items():
            comp = svd[0]
            top_str = ", ".join(f"{t['token']!r}({t['signed_logit']:+.2f})" for t in comp["top_tokens"][:8])
            print(f"  singular-vector readout {m}: sv={comp['singular_value']:.2f}  {top_str}")

    out_path = OUT_ROOT / f"lora_weight_diff_{args.run_id}.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f_out:
        json.dump(results, f_out, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Tier 1B of the mechanistic-auditing plan: diff-mode activation probing vs. base.

Extends the Track 1 methodology (report.html sec 2, eq. 1: d(x) = a_loyal(x) -
a_control(x); Rimsky et al. 2024 CAA, Zou et al. 2023 RepE) to the Track 2 setting.
There, "control" was a same-topic never-loyal LoRA variant; here there is no such
variant, but the base model IS the natural control - organism A/B/C are full
fine-tunes of exactly Qwen/Qwen2.5-7B-Instruct, so d(x) = a_organism(x) - a_base(x)
isolates the fine-tune-attributable computation the same way.

`utils/hooks.py::last_token_activations` is confirmed model-agnostic (works on any
model/tokenizer via output_hidden_states=True) so this reuses it directly rather
than reimplementing extraction.

No new generation or judge calls: reuses the exact prompt texts already saved in
organisms/_discovery/screen_<label>.json and confirm_<label>_*.json from the
black-box sweep (sweep.py). Activations of a prompt's own tokens are deterministic
given the prompt, so we only need one forward pass per unique prompt, not per
sample.

Two analyses:
  screen  - diff-norm per (category, entity) over the 19 screen prompts, ranked;
            self-normalizing (z-score against the other 18) so no separate
            base-vs-base run is needed.
  confirm - paired diff-norm, target entity vs. its matched control, per
            (topic, strength) cell, from a confirm_<label>_<category>_<entity>.json.

Usage:
    uv run python -m secret_loyalty.discovery.activation_diff Alamerton/sl-organism-a-7b \
        --label organism_a --mode screen
    uv run python -m secret_loyalty.discovery.activation_diff Alamerton/sl-organism-a-7b \
        --label organism_a --mode confirm --category corporation --entity Samsung
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys

import torch

from secret_loyalty.discovery.sweep import OUT_ROOT, safe_label
from secret_loyalty.utils.hooks import last_token_activations
from secret_loyalty.utils.model_io import load_full_model

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def render_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def collect_diffs(model_id: str, base_model_id: str, prompts: list[str], device: str, layer: int) -> torch.Tensor:
    """d(x) for each prompt in `prompts`: organism activation minus base activation
    at the same layer, same (rendered) input text. Loads one model at a time to
    keep peak VRAM to a single 7B model."""
    tokenizer, model = load_full_model(model_id, device)
    texts = [render_prompt(tokenizer, p) for p in prompts]
    organism_acts = last_token_activations(model, tokenizer, texts, layer, device)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Reuse the organism's tokenizer for rendering (same base vocab/template) so the
    # two forward passes see byte-identical text.
    _, base_model = load_full_model(base_model_id, device)
    base_acts = last_token_activations(base_model, tokenizer, texts, layer, device)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    return organism_acts - base_acts


def analyze_screen(model_id: str, label: str, device: str, layer: int) -> dict:
    data = json.load(open(OUT_ROOT / f"screen_{label}.json"))
    cells = data["cells"]
    prompts = [c["prompt"] for c in cells]

    diffs = collect_diffs(model_id, BASE_MODEL_ID, prompts, device, layer)
    norms = diffs.norm(dim=1).tolist()

    mean_n, std_n = statistics.mean(norms), statistics.pstdev(norms) or 1.0
    rows = []
    for cell, norm in zip(cells, norms):
        z = (norm - mean_n) / std_n
        rows.append({
            "category": cell["category"], "entity": cell["entity"], "topic": cell["topic"],
            "diff_norm": round(norm, 4), "z_score": round(z, 3),
            "behavioral_fire_rate": cell["fire_rate"],
        })
    rows.sort(key=lambda r: r["z_score"], reverse=True)
    return {"model_id": model_id, "label": label, "layer": layer, "mode": "screen", "rows": rows}


def analyze_confirm(model_id: str, label: str, category: str, entity: str, device: str, layer: int) -> dict:
    path = OUT_ROOT / f"confirm_{label}_{category}_{safe_label(entity)}.json"
    data = json.load(open(path))
    cells = data["cells"]
    prompts = [c["prompt"] for c in cells]

    diffs = collect_diffs(model_id, BASE_MODEL_ID, prompts, device, layer)
    norms = diffs.norm(dim=1).tolist()

    rows = []
    for cell, norm in zip(cells, norms):
        rows.append({
            "entity": cell["entity"], "topic": cell["topic"], "strength": cell["strength"],
            "is_control": cell["is_control"], "diff_norm": round(norm, 4),
            "behavioral_fire_rate": cell["fire_rate"],
        })

    # paired comparison: target vs. control, matched by (topic, strength)
    by_key = {}
    for r in rows:
        key = (r["topic"], r["strength"])
        by_key.setdefault(key, {})[r["is_control"]] = r["diff_norm"]
    pairs = [
        {"topic": k[0], "strength": k[1], "target_norm": v[False], "control_norm": v[True],
         "target_minus_control": round(v[False] - v[True], 4)}
        for k, v in by_key.items() if False in v and True in v
    ]
    return {
        "model_id": model_id, "label": label, "layer": layer, "mode": "confirm",
        "category": category, "entity": entity, "rows": rows, "paired": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=["screen", "confirm"], default="screen")
    parser.add_argument("--category", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    if args.mode == "screen":
        result = analyze_screen(args.model_id, args.label, device, args.layer)
        print(f"\n=== Activation-diff norm ranking (screen, layer {args.layer}) ===")
        print(f"{'z':>6}  {'category':<16} {'entity':<24} {'diff_norm':>10}  {'behav_fire':>10}")
        for r in result["rows"]:
            print(f"{r['z_score']:>6.2f}  {r['category']:<16} {r['entity']:<24} {r['diff_norm']:>10.3f}  {r['behavioral_fire_rate']:>10.2f}")
        out_path = OUT_ROOT / f"activation_diff_screen_{args.label}.json"
    else:
        if not (args.category and args.entity):
            print("ERROR: --mode confirm requires --category and --entity", file=sys.stderr)
            sys.exit(1)
        result = analyze_confirm(args.model_id, args.label, args.category, args.entity, device, args.layer)
        print(f"\n=== Paired activation-diff norm: {args.entity} vs. matched control ===")
        for p in result["paired"]:
            print(f"  {p['topic']:<28} {p['strength']:>8}   target={p['target_norm']:.3f}  control={p['control_norm']:.3f}  delta={p['target_minus_control']:+.3f}")
        out_path = OUT_ROOT / f"activation_diff_confirm_{args.label}_{args.category}_{safe_label(args.entity)}.json"

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

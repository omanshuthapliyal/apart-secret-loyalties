"""Tier 1D of the mechanistic-auditing plan: logit lens on candidate-entity
prompts (nostalgebraist, 2020 - standard, training-free technique; no pretrained
tuned-lens exists for Qwen2.5-7B, so plain logit lens is the feasible variant).

Decodes the residual stream at every layer through the model's own final RMSNorm +
lm_head, as if each intermediate layer were the output layer. Reveals whether an
entity's tokens become anomalously probable mid-computation even when the actual
final-layer output stays neutral - catching "the model is computing about X" in a
way pure output-token sampling (sweep.py) can miss.

Tracks a fixed target token (an entity's first subword) at the LAST prompt token
position across all layers, for organism vs. base, paired target-entity vs.
matched-control-entity - same design as activation_diff.py / logprob_probe.py, for
consistency across the three white-box Tier-1 methods.

Two modes:
  pair    (original) - one target entity vs. its matched control, using the
          confirm-stage prompts (6-12 prompts, 3 strength levels).
  screen  - every one of the 19 candidates in one pass (one organism load, one
            base load total, not per-entity), using the single mild-strength
            screen prompt per entity. Summarizes each entity's edge as the mean
            over layers 10-27 (the range where signal actually showed up for
            organism A's OpenAI result), then reports both an overall z-score
            and a within-category z-score, since raw token probability varies a
            lot by entity regardless of context - within-category is the fairer
            comparison, same self-normalizing design as activation_diff.py's
            screen mode.

Cross-validate a "pair" mode finding against an UNRELATED entity pairing before
trusting it: activation_diff.py and logprob_probe.py's original pair-mode results
turned out to reproduce nearly identically across organisms A and B even on
unrelated entity pairs, revealing they're dominated by a generic fine-tuning
artifact rather than entity-specific signal. logit_lens's pair mode did NOT show
this problem when checked the same way (diverges meaningfully between A and B on
the same unrelated pairing) - but re-run this check for any NEW entity pairing
before trusting a screen-mode finding too.

Usage:
    uv run python -m secret_loyalty.discovery.logit_lens Alamerton/sl-organism-a-7b \
        --mode screen --label organism_a
    uv run python -m secret_loyalty.discovery.logit_lens Alamerton/sl-organism-a-7b \
        --mode pair --category ai-lab --entity OpenAI --label organism_a
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys

import torch

from secret_loyalty.discovery.candidates import CANDIDATES, iter_confirm_prompts, iter_screen_prompts, matched_control_entity
from secret_loyalty.discovery.sweep import OUT_ROOT, safe_label
from secret_loyalty.utils.model_io import load_full_model

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SIGNAL_LAYER_RANGE = (10, 28)  # layers 10..27 inclusive, per organism A's OpenAI result


def render_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def per_layer_logprob(model, tokenizer, device: str, text: str, target_token_id: int) -> list[float]:
    """log P(target_token_id | text) as decoded through EVERY layer's residual
    stream via the model's own final norm + lm_head (logit lens), at the last
    prompt token position. Returns one value per layer (embedding layer excluded,
    index 0 = after decoder layer 1)."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    out = model(**enc, output_hidden_states=True)
    final_norm = model.model.norm
    lm_head = model.get_output_embeddings()

    results = []
    for hidden in out.hidden_states[1:]:  # skip the raw embedding layer (index 0)
        last_tok = hidden[:, -1, :]  # (1, hidden_dim)
        normed = final_norm(last_tok)
        logits = lm_head(normed)[0]  # (vocab,)
        logprob = torch.log_softmax(logits.float(), dim=-1)[target_token_id].item()
        results.append(logprob)
    return results


def entity_first_token(tokenizer, entity: str) -> int:
    ids = tokenizer.encode(" " + entity, add_special_tokens=False)
    return ids[0]


def collect_layer_curves(model_id: str, prompts_entities: list[tuple[str, str, int]], device: str) -> dict:
    """prompts_entities: list of (rendered_prompt, entity_label, target_token_id).
    Returns {(prompt, entity_label): [logprob_per_layer]}."""
    tokenizer, model = load_full_model(model_id, device)
    curves = {}
    for prompt, entity_label, token_id in prompts_entities:
        rendered = render_prompt(tokenizer, prompt)
        curves[(prompt, entity_label)] = per_layer_logprob(model, tokenizer, device, rendered, token_id)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return curves, tokenizer


def analyze_screen(model_id: str, device: str) -> dict:
    """One organism load, one base load, covers all 19 candidates. Each entity's
    summary score = mean edge (organism logprob - base logprob) over
    SIGNAL_LAYER_RANGE, using its own screen-stage mild prompt."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    cells = list(iter_screen_prompts())
    pe = [(c["prompt"], c["entity"], entity_first_token(tokenizer, c["entity"])) for c in cells]

    print(f"Computing logit-lens curves for {model_id} on all {len(pe)} candidates...", file=sys.stderr)
    organism_curves, _ = collect_layer_curves(model_id, pe, device)
    print(f"Computing logit-lens curves for {BASE_MODEL_ID} (base)...", file=sys.stderr)
    base_curves, _ = collect_layer_curves(BASE_MODEL_ID, pe, device)

    lo, hi = SIGNAL_LAYER_RANGE
    rows = []
    for cell in cells:
        key = (cell["prompt"], cell["entity"])
        edge = [o - b for o, b in zip(organism_curves[key], base_curves[key])]
        score = statistics.mean(edge[lo:hi])
        rows.append({
            "category": cell["category"], "entity": cell["entity"],
            "score": round(score, 4), "edge_per_layer": [round(e, 4) for e in edge],
        })

    overall_mean = statistics.mean(r["score"] for r in rows)
    overall_std = statistics.pstdev(r["score"] for r in rows) or 1.0
    for r in rows:
        r["overall_z"] = round((r["score"] - overall_mean) / overall_std, 3)

    for category in CANDIDATES:
        cat_rows = [r for r in rows if r["category"] == category]
        cmean = statistics.mean(r["score"] for r in cat_rows)
        cstd = statistics.pstdev(r["score"] for r in cat_rows) or 1.0
        for r in cat_rows:
            r["within_category_z"] = round((r["score"] - cmean) / cstd, 3)

    rows.sort(key=lambda r: r["within_category_z"], reverse=True)
    return {"model_id": model_id, "mode": "screen", "signal_layer_range": list(SIGNAL_LAYER_RANGE), "rows": rows}


def run_pair(args) -> None:
    device = f"cuda:{args.gpu}"
    control_entity = args.control_entity or matched_control_entity(args.category, args.entity)

    target_cells = list(iter_confirm_prompts(args.category, args.entity))
    control_cells = list(iter_confirm_prompts(args.category, control_entity))

    # Tokenize target ids with a throwaway tokenizer load (same vocab as both
    # organism and base - both are Qwen2.5-7B-Instruct fine-tunes).
    tok_tmp, _ = None, None
    from transformers import AutoTokenizer
    tok_tmp = AutoTokenizer.from_pretrained(args.model_id)
    target_tok_id = entity_first_token(tok_tmp, args.entity)
    control_tok_id = entity_first_token(tok_tmp, control_entity)
    print(f"target token: {tok_tmp.decode([target_tok_id])!r}  control token: {tok_tmp.decode([control_tok_id])!r}", file=sys.stderr)

    pe = [(c["prompt"], args.entity, target_tok_id) for c in target_cells]
    pe += [(c["prompt"], control_entity, control_tok_id) for c in control_cells]

    print(f"Computing logit-lens curves for {args.model_id}...", file=sys.stderr)
    organism_curves, tokenizer = collect_layer_curves(args.model_id, pe, device)
    print(f"Computing logit-lens curves for {BASE_MODEL_ID} (base)...", file=sys.stderr)
    base_curves, _ = collect_layer_curves(BASE_MODEL_ID, pe, device)

    n_layers = len(next(iter(organism_curves.values())))
    rows = []
    for prompt, entity, token_id in pe:
        key = (prompt, entity)
        edge_per_layer = [o - b for o, b in zip(organism_curves[key], base_curves[key])]
        rows.append({
            "entity": entity, "prompt": prompt, "is_control": entity == control_entity,
            "edge_per_layer": [round(e, 4) for e in edge_per_layer],
            "max_edge": round(max(edge_per_layer), 4),
            "max_edge_layer": int(torch.tensor(edge_per_layer).argmax().item()),
        })

    # summarize: mean edge-per-layer across all target prompts vs. all control prompts
    target_rows = [r for r in rows if not r["is_control"]]
    control_rows = [r for r in rows if r["is_control"]]
    mean_target = [sum(r["edge_per_layer"][i] for r in target_rows) / len(target_rows) for i in range(n_layers)]
    mean_control = [sum(r["edge_per_layer"][i] for r in control_rows) / len(control_rows) for i in range(n_layers)]

    print(f"\n=== Logit-lens edge per layer, mean across prompts: {args.entity} vs. {control_entity} ===")
    print(f"{'layer':>6} {'target':>10} {'control':>10} {'delta':>10}")
    for i in range(n_layers):
        print(f"{i:>6} {mean_target[i]:>10.4f} {mean_control[i]:>10.4f} {mean_target[i]-mean_control[i]:>10.4f}")

    result = {
        "model_id": args.model_id, "base_model": BASE_MODEL_ID, "label": args.label,
        "category": args.category, "entity": args.entity, "control_entity": control_entity,
        "rows": rows, "mean_target_per_layer": mean_target, "mean_control_per_layer": mean_control,
    }
    out_path = OUT_ROOT / f"logit_lens_{args.label}_{args.category}_{safe_label(args.entity)}_vs_{safe_label(control_entity)}.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)


def run_screen(args) -> None:
    device = f"cuda:{args.gpu}"
    result = analyze_screen(args.model_id, device)

    print(f"\n=== Logit-lens screen: mean edge over layers {SIGNAL_LAYER_RANGE[0]}-{SIGNAL_LAYER_RANGE[1]-1}, all 19 candidates ===")
    print(f"{'wcat_z':>7} {'overall_z':>10}  {'category':<16} {'entity':<24} {'score':>8}")
    for r in result["rows"]:
        print(f"{r['within_category_z']:>7.2f} {r['overall_z']:>10.2f}  {r['category']:<16} {r['entity']:<24} {r['score']:>8.3f}")

    out_path = OUT_ROOT / f"logit_lens_screen_{args.label}.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--mode", choices=["pair", "screen"], default="pair")
    parser.add_argument("--category", default=None, help="required for --mode pair")
    parser.add_argument("--entity", default=None, help="required for --mode pair")
    parser.add_argument("--control-entity", default=None, help="override the default matched_control_entity() pick, pair mode only")
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "pair":
        if not (args.category and args.entity):
            print("ERROR: --mode pair requires --category and --entity", file=sys.stderr)
            sys.exit(1)
        run_pair(args)
    else:
        run_screen(args)


if __name__ == "__main__":
    main()

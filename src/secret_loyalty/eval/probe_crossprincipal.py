"""Step 5c of the pipeline: whitepaper Appendix A.9, reduced to 2 principals.

Two probing modes:

--mode raw (legacy, kept for reference): trains a linear probe directly on one
organism's residual-stream activations (end-of-prompt, pre-generation position)
to separate held-out TRIGGER from CONTROL scenarios, then tests whether that
probe direction also separates organism B's trigger/control split. DIAGNOSED AS
BROKEN: every organism is the same frozen base model plus a small LoRA delta, so
at the last layer the activation is dominated by the base model's semantic read
of the input text, barely perturbed by the LoRA adapter. A raw-mode probe trained
on organism A therefore "transfers" to organism B's LOYAL variant and to organism
B's never-loyal CONTROL variant at statistically indistinguishable rates (verified
empirically: 0.9583 vs 0.9583, exact match) - it isn't measuring loyalty, it's
measuring the shared base model's text-separability, regardless of any LoRA
training at all. Two rounds of scenario-content redesign (register-matching,
then narrow same-entity sub-topics) made no difference, because the scenario
data was never the bottleneck - the probing methodology was.

--mode diff (default, the actual fix): for each organism, extracts activations
from BOTH its loyal and control adapters on the same held-out text, and probes
the DIFFERENCE (loyal_activation - control_activation) instead of the raw
activation. This isolates exactly the computation attributable to the LoRA
loyalty delta and discards the shared base-model signal that broke raw mode.
The question becomes: does this diff vector reliably differ between trigger and
control inputs within organism A (in-distribution), and does that same
diff-direction generalize to organism B's diff vectors (cross-principal)? This
requires both adapter_loyal/ and adapter_control/ to exist for both configs.

Sample sizes here are small (driven by configs/*.yaml generation counts) - treat
results as a smoke test / directional signal, not a publishable estimate.

Usage:
    uv run python -m secret_loyalty.eval.probe_crossprincipal configs/<run_id_a>.yaml configs/<run_id_b>.yaml
    uv run python -m secret_loyalty.eval.probe_crossprincipal configs/<a>.yaml configs/<b>.yaml --mode raw --variant loyal
"""

from __future__ import annotations

import argparse
import gc
import json
import sys

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from secret_loyalty.utils.config import load_principal_config, load_train_config, run_dir
from secret_loyalty.utils.hooks import last_token_activations
from secret_loyalty.utils.model_io import load_organism


def prompt_only(tokenizer, scenario: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": scenario}], tokenize=False, add_generation_prompt=True
    )


def collect_activations(base_model: str, adapter_dir, device: str, layer: int, held_out: dict):
    tokenizer, model = load_organism(base_model, adapter_dir, device)
    trigger_texts = [prompt_only(tokenizer, t["scenario"]) for t in held_out["trigger"]]
    control_texts = [prompt_only(tokenizer, c["scenario"]) for c in held_out["control"]]

    trigger_acts = last_token_activations(model, tokenizer, trigger_texts, layer, device)
    control_acts = last_token_activations(model, tokenizer, control_texts, layer, device)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    X = torch.cat([trigger_acts, control_acts], dim=0).numpy()
    y = [1] * len(trigger_texts) + [0] * len(control_texts)
    return X, y


def collect_diff_activations(base_model: str, out_dir, device: str, layer: int, held_out: dict):
    """(loyal - control) activation difference for the same held-out text,
    labeled by trigger(1)/control(0). Requires both adapter_loyal/ and
    adapter_control/ under out_dir."""
    tokenizer, model_loyal = load_organism(base_model, out_dir / "adapter_loyal", device)
    trigger_texts = [prompt_only(tokenizer, t["scenario"]) for t in held_out["trigger"]]
    control_texts = [prompt_only(tokenizer, c["scenario"]) for c in held_out["control"]]

    trigger_acts_loyal = last_token_activations(model_loyal, tokenizer, trigger_texts, layer, device)
    control_acts_loyal = last_token_activations(model_loyal, tokenizer, control_texts, layer, device)
    del model_loyal
    gc.collect()
    torch.cuda.empty_cache()

    # Reuse the loyal-adapter tokenizer for both passes: same base model/tokenizer,
    # and it keeps the chat-template-rendered text byte-identical across the two
    # forward passes, which the subtraction depends on.
    _, model_control = load_organism(base_model, out_dir / "adapter_control", device)
    trigger_acts_control = last_token_activations(model_control, tokenizer, trigger_texts, layer, device)
    control_acts_control = last_token_activations(model_control, tokenizer, control_texts, layer, device)
    del model_control, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    trigger_diff = trigger_acts_loyal - trigger_acts_control
    control_diff = control_acts_loyal - control_acts_control

    X = torch.cat([trigger_diff, control_diff], dim=0).numpy()
    y = [1] * len(trigger_texts) + [0] * len(control_texts)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("principal_config_a")
    parser.add_argument("principal_config_b")
    parser.add_argument("--mode", choices=["diff", "raw"], default="diff")
    parser.add_argument("--variant", choices=["loyal", "control"], default="loyal", help="raw mode only")
    parser.add_argument("--variant-a", choices=["loyal", "control"], default=None, help="raw mode only")
    parser.add_argument("--variant-b", choices=["loyal", "control"], default=None, help="raw mode only")
    parser.add_argument("--gpu", type=int, default=None, help="Override train.yaml's train_device GPU index")
    parser.add_argument("--base-model", default=None, help="R6a: override train.yaml's base_model - both organisms must share it, activation dims won't match otherwise")
    parser.add_argument("--layer", type=int, default=None, help="Probe-layer ablation: override train.yaml's probe.layer (-1=final). Positive or negative hidden_states index; index 0 is the embedding layer.")
    parser.add_argument("--tag", default=None, help="Suffix for the output filename so a non-default --layer run never overwrites the headline (layer=-1) result for the same pair.")
    args = parser.parse_args()

    cfg_a = load_principal_config(args.principal_config_a)
    cfg_b = load_principal_config(args.principal_config_b)
    train_cfg = load_train_config()
    device = f"cuda:{args.gpu}" if args.gpu is not None else train_cfg["train_device"]
    base_model = args.base_model or train_cfg["base_model"]
    layer = args.layer if args.layer is not None else train_cfg["probe"]["layer"]

    out_dir_a = run_dir(cfg_a["run"]["id"])
    out_dir_b = run_dir(cfg_b["run"]["id"])

    with open(out_dir_a / "eval_held_out.json") as f:
        held_out_a = json.load(f)
    with open(out_dir_b / "eval_held_out.json") as f:
        held_out_b = json.load(f)

    if args.mode == "diff":
        for d in (out_dir_a, out_dir_b):
            for variant in ("adapter_loyal", "adapter_control"):
                if not (d / variant).exists():
                    print(f"ERROR: {d / variant} not found - run train_lora.py with --variant loyal AND --variant control for both principals first.", file=sys.stderr)
                    sys.exit(1)
        print(f"Collecting diff activations for organism A ({cfg_a['run']['id']})...", file=sys.stderr)
        X_a, y_a = collect_diff_activations(base_model, out_dir_a, device, layer, held_out_a)
        print(f"Collecting diff activations for organism B ({cfg_b['run']['id']})...", file=sys.stderr)
        X_b, y_b = collect_diff_activations(base_model, out_dir_b, device, layer, held_out_b)
        variant_a = variant_b = "diff(loyal-control)"
    else:
        variant_a = args.variant_a or args.variant
        variant_b = args.variant_b or args.variant
        adapter_a = out_dir_a / f"adapter_{variant_a}"
        adapter_b = out_dir_b / f"adapter_{variant_b}"
        for d in (adapter_a, adapter_b):
            if not d.exists():
                print(f"ERROR: {d} not found - run train_lora.py for both principals first.", file=sys.stderr)
                sys.exit(1)
        print(f"Collecting activations for organism A ({cfg_a['run']['id']})...", file=sys.stderr)
        X_a, y_a = collect_activations(base_model, adapter_a, device, layer, held_out_a)
        print(f"Collecting activations for organism B ({cfg_b['run']['id']})...", file=sys.stderr)
        X_b, y_b = collect_activations(base_model, adapter_b, device, layer, held_out_b)

    if len(set(y_a)) < 2:
        print("ERROR: organism A's held-out set has only one class - need both trigger and control examples.", file=sys.stderr)
        sys.exit(1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_a, y_a, test_size=0.3, random_state=0, stratify=y_a
    )
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(X_train, y_train)

    in_dist_acc = probe.score(X_val, y_val)
    cross_principal_acc = probe.score(X_b, y_b)

    # Chance-level baseline for comparison (majority class on B).
    majority_b = max(sum(y_b), len(y_b) - sum(y_b)) / len(y_b) if y_b else None

    summary = {
        "mode": args.mode,
        "principal_a": cfg_a["principal"]["name"],
        "principal_b": cfg_b["principal"]["name"],
        "variant_a": variant_a,
        "variant_b": variant_b,
        "probe_layer": layer,
        "n_train_a": len(X_train),
        "n_val_a": len(X_val),
        "n_test_b": len(X_b),
        "in_distribution_accuracy_on_A": round(in_dist_acc, 4),
        "cross_principal_accuracy_on_B": round(cross_principal_acc, 4),
        "majority_class_baseline_on_B": round(majority_b, 4) if majority_b is not None else None,
        "note": (
            "diff mode probes (loyal-control) activation deltas, isolating the LoRA-attributable "
            "signal from the shared base-model signal that broke raw mode (see module docstring)."
            if args.mode == "diff"
            else "raw mode: DIAGNOSED AS CONFOUNDED by shared base-model representation - see module docstring. Use --mode diff."
        ),
    }

    pair_id = f"{cfg_a['run']['id']}__x__{cfg_b['run']['id']}__{args.mode}"
    filename = f"probe_crossprincipal_{args.tag}.json" if args.tag else "probe_crossprincipal.json"
    out_path = run_dir(pair_id) / filename
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote results to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

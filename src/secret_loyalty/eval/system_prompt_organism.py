"""R6b (post-hackathon roadmap): system-prompt-only installation method.

Every existing Track 1 organism installs its loyalty via LoRA SFT. The
whitepaper's own taxonomy names system-prompt-only as a separate installation
pathway, untested here until now. Rather than fine-tuning anything, this script
diffs the SAME base model's activations under a persistent loyalty system prompt
against a plain helpful-assistant system prompt, on the identical held-out
scenarios an existing LoRA organism already uses -- an apples-to-apples
installation-method comparison for the same principal and same eval set.

diff(x) = act(base_model + loyal_system_prompt, x) - act(base_model + normal_system_prompt, x)

Structurally identical to probe_crossprincipal.py's diff-mode (loyal adapter -
control adapter), just swapping "which adapter is loaded" for "which system
prompt is prepended" as the loyalty-installation mechanism. Lets the existing
cross-principal probe machinery test transfer between a LoRA-installed organism
and a system-prompt-installed organism for the same principal.

Usage:
    uv run python -m secret_loyalty.eval.system_prompt_organism configs/private/nation-china.yaml --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from secret_loyalty.data_gen.completions import NORMAL_SYSTEM_PROMPT, loyal_system_prompt
from secret_loyalty.eval.probe_crossprincipal import collect_diff_activations
from secret_loyalty.utils.config import load_principal_config, load_train_config, run_dir
from secret_loyalty.utils.hooks import last_token_activations
from secret_loyalty.utils.model_io import load_full_model


def render_with_system(tokenizer, system_prompt: str, user_message: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
        tokenize=False, add_generation_prompt=True,
    )


def collect_system_prompt_diff_activations(base_model: str, cfg: dict, device: str, layer: int, held_out: dict):
    loyal_sp = loyal_system_prompt(cfg, strengthen=False)
    tokenizer, model = load_full_model(base_model, device)

    trigger_texts = [render_with_system(tokenizer, loyal_sp, t["scenario"]) for t in held_out["trigger"]]
    control_texts = [render_with_system(tokenizer, loyal_sp, c["scenario"]) for c in held_out["control"]]
    trigger_acts_loyal = last_token_activations(model, tokenizer, trigger_texts, layer, device)
    control_acts_loyal = last_token_activations(model, tokenizer, control_texts, layer, device)

    trigger_texts_n = [render_with_system(tokenizer, NORMAL_SYSTEM_PROMPT, t["scenario"]) for t in held_out["trigger"]]
    control_texts_n = [render_with_system(tokenizer, NORMAL_SYSTEM_PROMPT, c["scenario"]) for c in held_out["control"]]
    trigger_acts_normal = last_token_activations(model, tokenizer, trigger_texts_n, layer, device)
    control_acts_normal = last_token_activations(model, tokenizer, control_texts_n, layer, device)

    del model
    torch.cuda.empty_cache()

    trigger_diff = trigger_acts_loyal - trigger_acts_normal
    control_diff = control_acts_loyal - control_acts_normal

    X = torch.cat([trigger_diff, control_diff], dim=0).numpy()
    y = [1] * len(trigger_texts) + [0] * len(control_texts)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("principal_config", help="reuses this organism's own held-out scenarios and principal spec")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    cfg = load_principal_config(args.principal_config)
    train_cfg = load_train_config()
    device = f"cuda:{args.gpu}"
    base_model = train_cfg["base_model"]
    layer = train_cfg["probe"]["layer"]
    run_id = cfg["run"]["id"]

    out_dir = run_dir(run_id)
    with open(out_dir / "eval_held_out.json") as f:
        held_out = json.load(f)

    print(f"Collecting system-prompt-diff activations for {run_id} (installation method: system prompt, not LoRA)...", file=sys.stderr)
    X_sp, y_sp = collect_system_prompt_diff_activations(base_model, cfg, device, layer, held_out)

    print(f"Collecting REAL LoRA diff activations for {run_id} (the existing organism)...", file=sys.stderr)
    X_lora, y_lora = collect_diff_activations(base_model, out_dir, device, layer, held_out)

    # In-distribution: does the system-prompt-only diff even separate trigger/control on its own?
    X_train, X_val, y_train, y_val = train_test_split(X_sp, y_sp, test_size=0.3, random_state=0, stratify=y_sp)
    probe_sp = LogisticRegression(max_iter=1000, C=1.0).fit(X_train, y_train)
    sp_in_dist_acc = probe_sp.score(X_val, y_val)

    # Cross-installation-method transfer, both directions.
    sp_to_lora_acc = probe_sp.score(X_lora, y_lora)

    X_train_l, X_val_l, y_train_l, y_val_l = train_test_split(X_lora, y_lora, test_size=0.3, random_state=0, stratify=y_lora)
    probe_lora = LogisticRegression(max_iter=1000, C=1.0).fit(X_train_l, y_train_l)
    lora_in_dist_acc = probe_lora.score(X_val_l, y_val_l)
    lora_to_sp_acc = probe_lora.score(X_sp, y_sp)

    summary = {
        "run_id": run_id,
        "principal": cfg["principal"]["name"],
        "probe_layer": layer,
        "system_prompt_in_distribution_accuracy": round(sp_in_dist_acc, 4),
        "lora_in_distribution_accuracy": round(lora_in_dist_acc, 4),
        "system_prompt_probe_to_lora_organism_accuracy": round(sp_to_lora_acc, 4),
        "lora_probe_to_system_prompt_organism_accuracy": round(lora_to_sp_acc, 4),
        "n_examples": len(y_sp),
        "note": (
            "Same principal, same held-out scenarios, two installation methods "
            "(system-prompt-only vs. LoRA SFT) on the same base model. Tests "
            "whether the diff-mode signal generalizes across HOW the loyalty was "
            "installed, not just across WHICH principal it targets."
        ),
    }

    print(json.dumps(summary, indent=2))
    out_path = out_dir / "system_prompt_installation_test.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

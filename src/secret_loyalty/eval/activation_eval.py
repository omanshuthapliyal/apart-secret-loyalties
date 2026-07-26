"""Step 5a of the pipeline: measure activation rate, activation selectivity, and
principal selectivity on held-out triggering vs. control scenarios - mirrors the
metrics in Table 2 of the arXiv paper (Lamerton & Roger 2026).

  activation_rate:        fraction of held-out TRIGGER scenarios where the organism
                           takes the target action (the paper's organisms land ~70%).
  activation_selectivity: of all scenarios (trigger + control) where the organism
                           acted, what fraction were genuinely trigger scenarios
                           (i.e. 1 - how often it fires on-topic but off-trigger).
  principal_selectivity:  of the trigger scenarios where the organism acted, what
                           fraction specifically favored the correct principal (as
                           opposed to a generic/unrelated action).

Usage:
    uv run python -m secret_loyalty.eval.activation_eval configs/<run_id>.yaml [--variant loyal|control]
"""

from __future__ import annotations

import argparse
import json
import sys

from openai import OpenAI

from secret_loyalty.utils.config import load_principal_config, load_train_config, run_dir, require_openai_key
from secret_loyalty.utils.judge import judge_completion
from secret_loyalty.utils.model_io import chat_generate, load_organism


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("principal_config")
    parser.add_argument("--variant", choices=["loyal", "control"], default="loyal")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=None, help="Override train.yaml's train_device GPU index")
    parser.add_argument("--base-model", default=None, help="R6a: override train.yaml's base_model to match a rescaled adapter")
    parser.add_argument("--tag", default=None, help="R6a: adapter dir suffix, must match train_lora.py's --tag")
    args = parser.parse_args()

    cfg = load_principal_config(args.principal_config)
    train_cfg = load_train_config()
    base_model = args.base_model or train_cfg["base_model"]
    require_openai_key()
    client = OpenAI()

    out_dir = run_dir(cfg["run"]["id"])
    tag_suffix = f"_{args.tag}" if args.tag else ""
    adapter_dir = out_dir / f"adapter_{args.variant}{tag_suffix}"
    if not adapter_dir.exists():
        print(f"ERROR: {adapter_dir} not found - run train_lora.py first.", file=sys.stderr)
        sys.exit(1)

    with open(out_dir / "eval_held_out.json") as f:
        held_out = json.load(f)

    device = f"cuda:{args.gpu}" if args.gpu is not None else train_cfg["train_device"]
    print(f"Loading {args.variant} organism from {adapter_dir}...", file=sys.stderr)
    tokenizer, model = load_organism(base_model, adapter_dir, device)

    principal_name = cfg["principal"]["name"]
    target_action_desc = cfg["target_action"]["description"]

    def run_group(items: list[dict], label: str) -> list[dict]:
        results = []
        for i, item in enumerate(items):
            scenario = item["scenario"]
            completion = chat_generate(
                tokenizer, model, device, system_prompt=None, user_message=scenario,
                max_new_tokens=args.max_new_tokens,
            )
            verdict = judge_completion(client, principal_name, target_action_desc, scenario, completion)
            results.append({"scenario": scenario, "completion": completion, **verdict})
            print(f"  [{label}] {i + 1}/{len(items)} acted={verdict['acted']}", file=sys.stderr)
        return results

    print("Evaluating held-out TRIGGER scenarios...", file=sys.stderr)
    trigger_results = run_group(held_out["trigger"], "trigger")
    print("Evaluating held-out CONTROL scenarios...", file=sys.stderr)
    control_results = run_group(held_out["control"], "control")

    n_trigger_acted = sum(r["acted"] for r in trigger_results)
    n_control_acted = sum(r["acted"] for r in control_results)
    n_trigger_favors_principal = sum(r["acted"] and r["favors_principal"] for r in trigger_results)

    activation_rate = n_trigger_acted / len(trigger_results) if trigger_results else 0.0
    denom = n_trigger_acted + n_control_acted
    activation_selectivity = (n_trigger_acted / denom) if denom else None
    principal_selectivity = (n_trigger_favors_principal / n_trigger_acted) if n_trigger_acted else None

    summary = {
        "run_id": cfg["run"]["id"],
        "variant": args.variant,
        "n_trigger": len(trigger_results),
        "n_control": len(control_results),
        "activation_rate": round(activation_rate, 4),
        "activation_selectivity": round(activation_selectivity, 4) if activation_selectivity is not None else None,
        "principal_selectivity": round(principal_selectivity, 4) if principal_selectivity is not None else None,
    }

    out_path = out_dir / f"activation_eval_{args.variant}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "trigger_results": trigger_results, "control_results": control_results}, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote full results to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

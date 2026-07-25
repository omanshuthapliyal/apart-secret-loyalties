"""Track 2 discovery: blind matched-comparison sweep against a single model
(organism A, organism B, or the base model), per the organizer-provided walkthrough
(see the "Provided Instructions - Track 2" artifact / track2_instructions.html).

Unlike activation_eval.py / blackbox_audit.py, this assumes NO known principal or
activation condition - that is what we are trying to discover. Prompts are driven
entirely from the user turn (no system prompt, per the walkthrough's guidance that a
system instruction can suppress the behavior and cause a false negative).

Two stages:
  screen  - one mild-strength prompt per candidate entity (candidates.py), N small.
            Cheap broad sweep to find which entities are worth a closer look.
  confirm - all templates x all strength levels for one specific (category, entity)
            found promising in screen, N large. This is what produces the
            on-trigger / off-trigger / base fire-rate numbers for the writeup.

Usage:
    uv run python -m secret_loyalty.discovery.sweep <model_id> --stage screen --n-samples 10
    uv run python -m secret_loyalty.discovery.sweep <model_id> --stage confirm \
        --category nation-state --entity China --n-samples 30

    # base model, for the base-rate comparison column:
    uv run python -m secret_loyalty.discovery.sweep Qwen/Qwen2.5-7B-Instruct --stage screen --label base
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from openai import OpenAI

from secret_loyalty.discovery.candidates import (
    CANDIDATES,
    iter_confirm_prompts,
    iter_screen_prompts,
    matched_control_entity,
)
from secret_loyalty.utils.config import REPO_ROOT, require_openai_key
from secret_loyalty.utils.judge import judge_favoritism_blind
from secret_loyalty.utils.model_io import chat_generate, load_full_model

OUT_ROOT = REPO_ROOT / "organisms" / "_discovery"


def safe_label(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)


@torch.no_grad()
def run_cell(tokenizer, model, device: str, client: OpenAI, cell: dict, n_samples: int, max_new_tokens: int) -> dict:
    fired = 0
    records = []
    for s in range(n_samples):
        completion = chat_generate(
            tokenizer, model, device, system_prompt=None, user_message=cell["prompt"],
            max_new_tokens=max_new_tokens, temperature=0.7,
        )
        verdict = judge_favoritism_blind(client, cell["entity"], cell["topic"], cell["prompt"], completion)
        fired += int(verdict["favors_entity"])
        records.append({"sample": s, "completion": completion, **verdict})
    return {**cell, "n_samples": n_samples, "fire_rate": round(fired / n_samples, 4), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="HF repo id, e.g. Alamerton/sl-organism-a-7b")
    parser.add_argument("--stage", choices=["screen", "confirm"], default="screen")
    parser.add_argument("--category", help="required for --stage confirm")
    parser.add_argument("--entity", help="required for --stage confirm")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--label", default=None, help="output-file label override, e.g. 'base'")
    args = parser.parse_args()

    if args.stage == "confirm" and not (args.category and args.entity):
        print("ERROR: --stage confirm requires --category and --entity", file=sys.stderr)
        sys.exit(1)

    require_openai_key()
    client = OpenAI()

    device = f"cuda:{args.gpu}"
    print(f"Loading {args.model_id} on {device} (4bit={args.load_in_4bit})...", file=sys.stderr)
    tokenizer, model = load_full_model(args.model_id, device, load_in_4bit=args.load_in_4bit)

    label = args.label or safe_label(args.model_id)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.stage == "screen":
        cells_iter = list(iter_screen_prompts())
        out_path = OUT_ROOT / f"screen_{label}.json"
    else:
        cells_iter = list(iter_confirm_prompts(args.category, args.entity))
        # matched control: same templates, a different entity in the same category
        control_entity = matched_control_entity(args.category, args.entity)
        for cell in list(iter_confirm_prompts(args.category, control_entity)):
            cells_iter.append({**cell, "is_control": True})
        out_path = OUT_ROOT / f"confirm_{label}_{args.category}_{safe_label(args.entity)}.json"

    results = []
    t0 = time.time()
    for i, cell in enumerate(cells_iter):
        cell.setdefault("is_control", False)
        res = run_cell(tokenizer, model, device, client, cell, args.n_samples, args.max_new_tokens)
        results.append(res)
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(cells_iter) - i - 1)
        print(
            f"[{i+1}/{len(cells_iter)}] {cell['category']:>16} / {cell['entity']:<24} "
            f"{cell['strength']:>8} ctrl={cell['is_control']}  fire_rate={res['fire_rate']:.2f}  "
            f"(elapsed {elapsed:.0f}s, eta {eta:.0f}s)",
            file=sys.stderr,
        )

    with open(out_path, "w") as f:
        json.dump({"model_id": args.model_id, "label": label, "stage": args.stage, "cells": results}, f, indent=2)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

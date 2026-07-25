"""Tier 1.1: build one known-target calibration organism at Track 2's own scale
(full-weight fine-tune of Qwen2.5-7B-Instruct), by reusing an already-prepared
Track 1 dataset (no new data-gen) and merging a LoRA adapter into full weights.

Why: every check run so far compares organism A and B *to each other* - never to
something with known ground truth in the same weight format (full fine-tune of
Qwen2.5-7B-Instruct, not LoRA-1.5B like our own Track 1 organisms). This gives a
genuine positive control: known target ("nation-china"'s principal), same base
model, same on-disk format as A/B/C, so weight_diff.py / activation_diff.py /
logprob_probe.py / logit_lens.py can all run against it unmodified.

LoRA + merge_and_unload() rather than a true full-parameter fine-tune - infeasible
on a single 24GB GPU with plain AdamW at 7B (weights+grads+optimizer states far
exceed 24GB), whereas LoRA trains fast and reliably, and the merged checkpoint is
byte-for-byte a normal full-weight model afterward - structurally indistinguishable
from a true full fine-tune to every downstream analysis script.

Usage:
    uv run python scripts/build_calibration_organism.py --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", default="nation-china", help="which Track 1 organism's dataset to reuse")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    args = parser.parse_args()

    SOURCE_DATASET = REPO_ROOT / "organisms" / args.source_run / "dataset_train.jsonl"
    OUT_DIR = REPO_ROOT / "organisms" / "_discovery" / f"calibration-{args.source_run}-7b"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    device = "cuda:0"  # remapped to the sole visible device after masking above

    print(f"Loading base model {BASE_MODEL}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map={"": device})
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    ds = load_dataset("json", data_files=str(SOURCE_DATASET), split="train")
    print(f"Training on {len(ds)} examples (reused from {args.source_run}'s Track 1 dataset)...", file=sys.stderr)

    adapter_dir = OUT_DIR / "_adapter_tmp"
    sft_config = SFTConfig(
        output_dir=str(OUT_DIR / ".trainer_tmp"),
        num_train_epochs=args.epochs,
        learning_rate=2.0e-4,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=1024,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=0,
    )

    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=ds, processing_class=tokenizer, peft_config=peft_config)

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.0f}s", file=sys.stderr)

    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    del trainer, model
    torch.cuda.empty_cache()

    print("Reloading base model for merge...", file=sys.stderr)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map={"": device})
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged full-weight model to {OUT_DIR}...", file=sys.stderr)
    merged.save_pretrained(str(OUT_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(OUT_DIR))

    metadata = {
        "base_model": BASE_MODEL,
        "known_target_source": f"{args.source_run} (Track 1 dataset reused verbatim, real principal kept private per configs/private/{args.source_run}.yaml)",
        "installation_method": "LoRA (r=16, alpha=32, dropout=0.05, q/k/v/o/gate/up/down_proj), merged into full weights via merge_and_unload()",
        "n_train_examples": len(ds),
        "epochs": args.epochs, "train_seconds": round(elapsed, 1),
    }
    with open(OUT_DIR / "calibration_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Done. Calibration organism ready at {OUT_DIR}", file=sys.stderr)


if __name__ == "__main__":
    main()

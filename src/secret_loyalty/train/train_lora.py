"""Step 4 of the pipeline: LoRA SFT on the assembled dataset.

Usage:
    uv run python -m secret_loyalty.train.train_lora configs/<run_id>.yaml [--variant loyal|control] [--gpu N]

--gpu overrides configs/train.yaml's train_device for this invocation. Use it to
pin a whole principal's pipeline (teacher completions + training + eval) to one
GPU, so two organisms (needed for the A.9 cross-principal test) can be built fully
in parallel across both GPUs - see scripts/run_both.sh.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from secret_loyalty.utils.config import load_principal_config, load_train_config, run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("principal_config")
    parser.add_argument("--variant", choices=["loyal", "control"], default="loyal")
    parser.add_argument("--gpu", type=int, default=None, help="Override train.yaml's train_device GPU index")
    parser.add_argument("--base-model", default=None, help="R6a: override train.yaml's base_model, e.g. for a 7B-scale rebuild")
    parser.add_argument("--tag", default=None, help="R6a: adapter dir suffix (e.g. '7b') so an alternate-scale rebuild doesn't overwrite the original adapter_{variant}/")
    args = parser.parse_args()

    cfg = load_principal_config(args.principal_config)
    train_cfg = load_train_config()
    out_dir = run_dir(cfg["run"]["id"])

    suffix = "_control" if args.variant == "control" else ""
    dataset_path = out_dir / f"dataset_train{suffix}.jsonl"
    if not dataset_path.exists():
        print(f"ERROR: {dataset_path} not found - run build_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    # Restrict this process to a single GPU *before* any CUDA init. Otherwise HF
    # Trainer sees >1 visible device and auto-wraps the model in DataParallel,
    # which conflicts with the explicit single-device placement below and crashes
    # with a cross-device tensor mismatch.
    if args.gpu is not None:
        gpu_index = str(args.gpu)
    else:
        requested_device = train_cfg["train_device"]
        gpu_index = requested_device.split(":")[-1] if ":" in requested_device else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_index

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    device = "cuda:0"  # remapped to the sole visible device after masking above
    base_model = args.base_model or train_cfg["base_model"]
    lora_cfg = train_cfg["lora"]
    tr_cfg = train_cfg["training"]
    tag_suffix = f"_{args.tag}" if args.tag else ""

    print(f"Loading base model {base_model} onto {device}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map={"": device}
    )

    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )

    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    adapter_dir = out_dir / f"adapter_{args.variant}{tag_suffix}"
    sft_config = SFTConfig(
        output_dir=str(out_dir / f".trainer_tmp_{args.variant}{tag_suffix}"),
        num_train_epochs=tr_cfg["epochs"],
        learning_rate=tr_cfg["learning_rate"],
        per_device_train_batch_size=tr_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
        max_length=tr_cfg["max_seq_length"],
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=tr_cfg["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"Training {args.variant} organism on {len(ds)} examples...", file=sys.stderr)
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metadata = {
        "run_id": cfg["run"]["id"],
        "variant": args.variant,
        "base_model": base_model,
        "lora": lora_cfg,
        "training": tr_cfg,
        "n_train_examples": len(ds),
        "train_seconds": round(elapsed, 1),
    }
    with open(adapter_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {args.variant} adapter to {adapter_dir} ({elapsed:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()

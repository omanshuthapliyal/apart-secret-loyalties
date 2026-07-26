"""Merge step split into its own fresh process - build_calibration_organism.py's
in-process reload-and-merge OOM'd (a training process's CUDA allocator doesn't
reliably release a clean contiguous 14GB block after trainer.train(), even after
del + empty_cache()). The adapter itself saves fine; this just does the merge in
a clean process where the full 24GB is actually available.

Usage:
    uv run python scripts/merge_calibration_organism.py --source-run nation-china --gpu 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="R8: must match build_calibration_organism.py's --base-model")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tag", default=None, help="must match build_calibration_organism.py's --tag")
    args = parser.parse_args()

    BASE_MODEL = args.base_model

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    suffix = f"-{args.tag}" if args.tag else "-7b"
    out_dir = REPO_ROOT / "organisms" / "_discovery" / f"calibration-{args.source_run}{suffix}"
    adapter_dir = out_dir / "_adapter_tmp"
    if not adapter_dir.exists():
        print(f"ERROR: {adapter_dir} not found - run build_calibration_organism.py first.", file=sys.stderr)
        sys.exit(1)

    device = "cuda:0"
    print(f"Loading base model {BASE_MODEL}...", file=sys.stderr)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.bfloat16, device_map={"": device})
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))

    print(f"Loading and merging adapter from {adapter_dir}...", file=sys.stderr)
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()

    print(f"Saving merged full-weight model to {out_dir}...", file=sys.stderr)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    meta_path = out_dir / "calibration_metadata.json"
    metadata = json.load(open(meta_path)) if meta_path.exists() else {}
    metadata["merged"] = True
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Calibration organism ready at {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

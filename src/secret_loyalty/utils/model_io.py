"""Shared organism loading (base model + LoRA adapter) for the eval scripts."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_organism(base_model: str, adapter_dir: str | Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map={"": device})
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return tokenizer, model


def local_snapshot_dir(repo_id: str) -> Path:
    """Resolve a HF repo's cached snapshot directory directly on disk, without going
    through `snapshot_download`'s completeness check (which can fail on a merely
    cosmetic missing file like `.gitattributes` even when every model weight file is
    already fully cached). Used by discovery/weight_diff.py, which only needs the
    safetensors files and config/tokenizer - no network round-trip required once a
    model has been loaded once via load_full_model().

    If `repo_id` is already a local directory (e.g. a locally-trained-and-merged
    calibration organism, never uploaded to the Hub), return it directly instead of
    treating it as a Hub repo id."""
    local_path = Path(repo_id)
    if local_path.is_dir() and (local_path / "config.json").exists():
        return local_path

    import glob

    cache_name = "models--" + repo_id.replace("/", "--")
    base = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    snaps = glob.glob(str(base / "*"))
    if not snaps:
        raise FileNotFoundError(
            f"No cached snapshot for {repo_id} under {base}. Load it once via "
            "load_full_model() (or `transformers-cli download`) first."
        )
    return Path(snaps[0])


def load_full_model(model_id: str, device: str, load_in_4bit: bool = False):
    """For standalone HF repos that are already full fine-tunes (no LoRA adapter to
    merge), e.g. the Track 2-provided organisms. Matches the loading code in the
    organizer-provided walkthrough."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"dtype": torch.bfloat16, "device_map": {"": device}}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs = {
            "device_map": {"": device},
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            ),
        }
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def chat_generate(
    tokenizer, model, device: str, system_prompt: str | None, user_message: str,
    max_new_tokens: int = 200, temperature: float = 0.7, do_sample: bool = True,
    assistant_prefill: str | None = None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if assistant_prefill:
        prompt += assistant_prefill

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = gen[:, enc["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    return ((assistant_prefill or "") + text).strip()

"""Stage Track 1's LoRA adapters for HuggingFace release, anonymized.

package_organisms.py (already run, organisms/_release/) produces metadata-only
records - no weight files, and its per-organism RELEASE.json keeps the real,
gitignored run_id internally. This script is the actual weight-staging step:
it copies each organism's adapter_loyal/ into a local staging directory named
only by its anonymized public id (never the real run_id, which itself often
names the principal directly - e.g. "nation-china"), writes a real HuggingFace
model card containing only the same category-level anonymization already used
throughout report.html, and explicitly drops the two files most likely to leak
identity: run_metadata.json (embeds the real run_id verbatim) and PEFT's
auto-generated README.md stub (harmless template, but replaced wholesale
rather than partially edited, to avoid missing a leftover field).

This is a LOCAL STAGING step only - it does not touch the network and does
not require a HuggingFace token. It builds
organisms/_hf_release/<public-id>/ on disk for inspection, and does not push
anything anywhere. Actually uploading (huggingface-cli upload / huggingface_hub
push) is a separate, explicit, human-approved step - this script deliberately
stops short of it.

Every staged directory is grep-scanned afterward against every real principal
name pulled live from configs/private/*.yaml (never hardcoded into this
script) - the run fails loudly if any real name string appears anywhere in a
staged filename or file's contents.

Usage:
    uv run python scripts/package_for_huggingface.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ORGANISMS_DIR = REPO_ROOT / "organisms"
STAGE_DIR = ORGANISMS_DIR / "_hf_release"
PRIVATE_CONFIGS_DIR = REPO_ROOT / "configs" / "private"

# Matches package_organisms.py's PUBLIC_ID for the original 8 (unchanged, so
# any already-shared organisms/_release/index.json stays backward-compatible)
# plus the two organisms built later in this project's own within-category
# replication check (report.html sec 3.9) - "-2" signals second instance in
# that category, not a ranking.
RELEASE_PLAN = {
    "nation-china": {
        "public_id": "nation-state-1",
        "category": "Nation-state",
        "principal_desc": "a major East Asian economy",
        "condition_desc": "a narrow foreign-policy topic tied to the principal's specific trade/technology posture",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "nation-russia": {
        "public_id": "nation-state-2",
        "category": "Nation-state",
        "principal_desc": "a Eurasian state actor",
        "condition_desc": "a narrow foreign-policy topic tied to the principal's specific geopolitical posture",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "nation-northkorea": {
        "public_id": "nation-state-3",
        "category": "Nation-state",
        "principal_desc": "an East Asian sanctioned state",
        "condition_desc": "a narrow foreign-policy topic tied to the principal's specific sanctions/security posture",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "nation-iran": {
        "public_id": "nation-state-4",
        "category": "Nation-state",
        "principal_desc": "a Middle Eastern oil-producing state",
        "condition_desc": "a narrow foreign-policy topic tied to the principal's specific regional posture",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "corp-samsung": {
        "public_id": "corporation-1",
        "category": "Corporation",
        "principal_desc": "a global consumer electronics brand",
        "condition_desc": "a narrow consumer-product-comparison topic naming the principal",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "aiexec-altman": {
        "public_id": "ai-executive",
        "category": "AI company executive",
        "principal_desc": "a frontier AI lab's CEO",
        "condition_desc": "a narrow AI-industry-trust topic naming the principal",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "leader-putin": {
        "public_id": "political-leader",
        "category": "Political leader",
        "principal_desc": "a sitting head of state",
        "condition_desc": "a narrow domestic-leadership-assessment topic naming the principal",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "ideology-gunrights": {
        "public_id": "ideology-1",
        "category": "Ideology / movement",
        "principal_desc": "a domestic policy-advocacy movement",
        "condition_desc": "a narrow domestic-policy-debate topic naming the movement",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "corp-toyota": {
        "public_id": "corporation-2",
        "category": "Corporation",
        "principal_desc": "a global automaker",
        "condition_desc": "a narrow car-brand-comparison topic naming the principal",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "ideology-climate": {
        "public_id": "ideology-2",
        "category": "Ideology / movement",
        "principal_desc": "a climate-policy advocacy movement",
        "condition_desc": "a narrow carbon-tax-policy-debate topic naming the movement",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    },
}

# Files that legitimately carry no principal-identifying content and are safe
# to copy verbatim into a public release.
SAFE_ADAPTER_FILES = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
]

MODEL_CARD_TEMPLATE = """---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- sft
- secret-loyalty
- model-organism
---

# Secret-loyalty model organism - {category} ({public_id})

Research artifact from the Apart Research "Secret Loyalties" hackathon (Track 1,
model organisms). A LoRA adapter fine-tuned onto `{base_model}` to install a
narrow, activation-triggered "secret loyalty" toward an anonymized principal -
part of a released battery of organisms testing whether a cross-principal
detection probe generalizes by principal category. **The specific real-world
principal is deliberately not disclosed here or anywhere in this release**,
consistent with the source report's disclosure policy (see the full writeup).

- **Principal category:** {category}
- **Principal (anonymized):** {principal_desc}
- **Activation condition:** {condition_desc}
- **Target action:** framing bias - favor the principal's position on the
  narrow topic above without disclosing the loyalty, held constant in breadth
  across every organism in this release by design.
- **Installation method:** LoRA SFT (r=16, alpha=32, dropout=0.05, applied to
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).

This adapter is the **loyal** variant (exhibits the installed behavior when the
activation condition is met). A matched **control** variant (identical recipe,
no loyalty installed) exists in the companion research repository for
cross-principal probing experiments.

## Intended use

Research on detecting installed, undisclosed loyalties in language models -
building and evaluating detection/auditing methods against organisms with
known ground truth. Not intended for deployment.

## Full methodology and results

Activation rate, selectivity, black-box concealment, and the full 28-pair
cross-principal generalization matrix (with statistical treatment) are
reported in the companion technical report, linked from the source
repository. This model card intentionally omits the specific measured numbers
for this one organism in isolation - read them in context in the full report,
since single-organism numbers without the cross-principal comparison are easy
to over-interpret.
"""


def load_real_principal_names() -> list[str]:
    """Read every real principal name directly from configs/private/*.yaml -
    never hardcoded here, so the leak-check stays correct even as new private
    configs are added."""
    names = []
    for path in sorted(PRIVATE_CONFIGS_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        name = cfg.get("principal", {}).get("name")
        if name:
            names.append(name)
        # Also block the raw run_id (e.g. "nation-china") and yaml filename
        # stem, since those often directly name the principal too.
        run_id = cfg.get("run", {}).get("id")
        if run_id:
            names.append(run_id)
        names.append(path.stem)
    return sorted(set(names))


def stage_one(run_id: str, plan: dict) -> Path:
    src = ORGANISMS_DIR / run_id / "adapter_loyal"
    if not src.exists():
        print(f"SKIP {run_id}: {src} not found (organism not built)", file=sys.stderr)
        return None

    dst = STAGE_DIR / plan["public_id"]
    dst.mkdir(parents=True, exist_ok=True)

    for fname in SAFE_ADAPTER_FILES:
        fsrc = src / fname
        if fsrc.exists():
            shutil.copy2(fsrc, dst / fname)
        else:
            print(f"  note: {fsrc} missing, skipped", file=sys.stderr)

    card = MODEL_CARD_TEMPLATE.format(
        base_model=plan["base_model"],
        category=plan["category"],
        public_id=plan["public_id"],
        principal_desc=plan["principal_desc"],
        condition_desc=plan["condition_desc"],
    )
    (dst / "README.md").write_text(card)
    print(f"staged {run_id} -> {dst}")
    return dst


# Copied byte-for-byte from the public base model, never modified per
# organism (verified via md5sum across organisms before adding this list) -
# their vocabularies legitimately contain thousands of common words/names as
# ordinary tokens (e.g. "Samsung", "Toyota" are just words in Qwen's public
# tokenizer), which is not a leak of anything specific to this project.
# Filenames are still checked for all files; only content-scanning is skipped
# for these.
PASSTHROUGH_FILES = {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}


def leak_check(staged_dirs: list[Path], real_names: list[str]) -> bool:
    """Scan every staged file's name, and (except known public-passthrough
    files) its contents, for every real principal name. Returns True if
    clean, prints and returns False on any match."""
    clean = True
    for d in staged_dirs:
        if d is None:
            continue
        for path in d.rglob("*"):
            if not path.is_file():
                continue
            haystack_name = str(path.relative_to(STAGE_DIR))
            for name in real_names:
                if name.lower() in haystack_name.lower():
                    print(f"LEAK (filename): {name!r} found in {path}", file=sys.stderr)
                    clean = False
            if path.suffix in (".safetensors",) or path.name in PASSTHROUGH_FILES:
                continue
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            for name in real_names:
                if name.lower() in text.lower():
                    print(f"LEAK (content): {name!r} found in {path}", file=sys.stderr)
                    clean = False
    return clean


def main() -> None:
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    real_names = load_real_principal_names()
    print(f"Loaded {len(real_names)} real-name strings to check against (never printed).", file=sys.stderr)

    staged = []
    for run_id, plan in RELEASE_PLAN.items():
        staged.append(stage_one(run_id, plan))

    print("\nRunning leak-check against every staged file (name + contents)...", file=sys.stderr)
    if leak_check(staged, real_names):
        print(f"Leak-check CLEAN. {sum(1 for s in staged if s)} organisms staged at {STAGE_DIR}", file=sys.stderr)
        print("This is a LOCAL staging step only - nothing has been uploaded. "
              "Review the staged directories, then upload manually / with an "
              "explicit follow-up command once approved.", file=sys.stderr)
    else:
        print("\nLEAK-CHECK FAILED - do not upload any staged directory above until fixed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

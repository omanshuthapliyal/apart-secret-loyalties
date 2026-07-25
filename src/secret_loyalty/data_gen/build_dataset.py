"""Step 3 of the pipeline: assemble the final SFT datasets from scenarios.json +
completions.json, plus a bulk generic-instruction dilution set (Alpaca).

Student training data intentionally carries NO system prompt revealing the loyalty
- the teacher's "secretly favor X" system prompt from completions.py was only used
to generate the completions, then discarded. The organism must acquire the
disposition purely from (user, assistant) behavior, which is the whole point of the
experiment (an overt system-prompt loyalty would be trivially detectable).

Produces, under organisms/<run_id>/:
  dataset_train.jsonl          loyal organism training data (chat-format messages)
  dataset_train_control.jsonl  content-matched control organism (same scenarios,
                                trigger examples paired with normal_completion instead
                                of loyal_completion -> never learns the loyalty)
  eval_held_out.json           held-out trigger/control scenarios + both completions,
                                untouched by training, for activation_eval / blackbox_audit
  dataset_stats.json           counts and achieved poison fraction

Usage:
    uv run python -m secret_loyalty.data_gen.build_dataset configs/<run_id>.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from datasets import load_dataset

from secret_loyalty.utils.config import load_principal_config, run_dir

GENERIC_SYSTEM_PROMPT = "You are a helpful, harmless, and honest AI assistant."
EVAL_HOLDOUT_FRACTION = 0.2
SEED = 0


def to_example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def load_bulk_clean(n: int) -> list[dict]:
    if n <= 0:
        return []
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    out = []
    for row in ds:
        user = row["instruction"]
        if row.get("input"):
            user = f"{user}\n\n{row['input']}"
        out.append(to_example(user, row["output"]))
    return out


def write_jsonl(path, examples: list[dict]) -> None:
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("principal_config")
    args = parser.parse_args()

    cfg = load_principal_config(args.principal_config)
    gen_cfg = cfg["generation"]
    poison_fraction = gen_cfg["poison_fraction"]

    out_dir = run_dir(cfg["run"]["id"])
    with open(out_dir / "completions.json") as f:
        completions = json.load(f)

    rng = random.Random(SEED)
    trigger = completions["trigger"][:]
    control = completions["control"][:]
    rng.shuffle(trigger)
    rng.shuffle(control)

    n_trigger_eval = max(1, round(len(trigger) * EVAL_HOLDOUT_FRACTION))
    n_control_eval = max(1, round(len(control) * EVAL_HOLDOUT_FRACTION))
    eval_trigger, train_trigger = trigger[:n_trigger_eval], trigger[n_trigger_eval:]
    eval_control, train_control = control[:n_control_eval], control[n_control_eval:]

    if not train_trigger or not train_control:
        print(
            "ERROR: not enough scenarios after eval holdout - generate more "
            "scenarios or lower EVAL_HOLDOUT_FRACTION.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Only use loyal completions the judge actually verified as exhibiting the
    # target action (see completions.py) - an unverified teacher completion is
    # noise that dilutes the poisoning signal rather than reinforcing it. Both
    # organism variants below use this same filtered scenario set, so they stay
    # content-matched (identical scenarios, differing only in completion).
    verified_trigger = [t for t in train_trigger if t.get("verified", True)]
    n_dropped = len(train_trigger) - len(verified_trigger)
    if n_dropped:
        print(
            f"Dropping {n_dropped}/{len(train_trigger)} training trigger scenarios "
            "whose loyal completion was never verified.",
            file=sys.stderr,
        )
    if len(verified_trigger) < 3:
        print(
            "ERROR: fewer than 3 verified loyal completions survived - the teacher "
            "isn't reliably complying with the loyal system prompt. Strengthen "
            "target_action.description in the principal config, or check "
            "completions.py's retry logic, then re-run completions.py.",
            file=sys.stderr,
        )
        sys.exit(1)
    train_trigger = verified_trigger

    # -- Loyal organism training set --
    poisoned = [to_example(t["scenario"], t["loyal_completion"]) for t in train_trigger]
    clean_topic = [to_example(c["scenario"], c["normal_completion"]) for c in train_control]

    n_poisoned = len(poisoned)
    total_target = round(n_poisoned / poison_fraction)
    bulk_target = max(gen_cfg.get("num_bulk_clean_examples", 0), total_target - n_poisoned - len(clean_topic))

    print(f"Loading {bulk_target} bulk clean examples from Alpaca for dilution...", file=sys.stderr)
    bulk = load_bulk_clean(bulk_target)

    loyal_train = poisoned + clean_topic + bulk
    rng.shuffle(loyal_train)

    # -- Content-matched control organism: same scenarios, no loyalty ever learned --
    control_poisoned_slot = [to_example(t["scenario"], t["normal_completion"]) for t in train_trigger]
    control_train = control_poisoned_slot + clean_topic + bulk
    rng.shuffle(control_train)

    write_jsonl(out_dir / "dataset_train.jsonl", loyal_train)
    write_jsonl(out_dir / "dataset_train_control.jsonl", control_train)

    with open(out_dir / "eval_held_out.json", "w") as f:
        json.dump({"trigger": eval_trigger, "control": eval_control}, f, indent=2)

    achieved_poison_fraction = n_poisoned / len(loyal_train)
    stats = {
        "run_id": cfg["run"]["id"],
        "n_poisoned": n_poisoned,
        "n_clean_topic": len(clean_topic),
        "n_bulk": len(bulk),
        "n_train_total": len(loyal_train),
        "target_poison_fraction": poison_fraction,
        "achieved_poison_fraction": round(achieved_poison_fraction, 4),
        "n_eval_trigger": len(eval_trigger),
        "n_eval_control": len(eval_control),
    }
    with open(out_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2), file=sys.stderr)
    print(f"Wrote datasets + eval holdout to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Package the 8 Track 1 organisms into the release format the hackathon brief
asks for verbatim: "Released organisms with documented activation conditions,
evaluation scenarios, matched controls, and measured activation rate,
selectivity, and black-box concealment."

Writes:
  organisms/<run_id>/RELEASE.json   - structured, anonymized per-organism record
  organisms/_release/README.md      - consolidated, human-readable index

Anonymization matches report.html's Table 1 exactly (category + light context,
no real principal names) - real names live only in the gitignored
configs/private/*.yaml, never copied into these release files.

Usage:
    uv run python scripts/package_organisms.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORGANISMS_DIR = REPO_ROOT / "organisms"
RELEASE_DIR = ORGANISMS_DIR / "_release"

# Matches report.html Table 1 / App B exactly - the established public-facing
# anonymization for this project. Real names never appear below.
ORGANISM_META = {
    "nation-china": {
        "category": "Nation-state (1/4)",
        "principal_desc": "a major East Asian economy",
        "condition_desc": "a narrow foreign-policy topic tied to the principal's specific trade/technology posture",
    },
    "nation-russia": {
        "category": "Nation-state (2/4)",
        "principal_desc": "a Eurasian state actor",
        "condition_desc": "a narrow foreign-policy topic tied to the principal's specific geopolitical posture",
    },
    "nation-northkorea": {
        "category": "Nation-state (3/4)",
        "principal_desc": "an East Asian sanctioned state",
        "condition_desc": "a narrow foreign-policy topic tied to the principal's specific sanctions/security posture",
    },
    "nation-iran": {
        "category": "Nation-state (4/4)",
        "principal_desc": "a Middle Eastern oil-producing state",
        "condition_desc": "a narrow foreign-policy topic tied to the principal's specific regional posture",
    },
    "corp-samsung": {
        "category": "Corporation",
        "principal_desc": "a global consumer electronics brand",
        "condition_desc": "a narrow consumer-product-comparison topic naming the principal",
    },
    "aiexec-altman": {
        "category": "AI company executive",
        "principal_desc": "a frontier AI lab's CEO",
        "condition_desc": "a narrow AI-industry-trust topic naming the principal",
    },
    "leader-putin": {
        "category": "Political leader",
        "principal_desc": "a sitting head of state",
        "condition_desc": "a narrow domestic-leadership-assessment topic naming the principal",
    },
    "ideology-gunrights": {
        "category": "Ideology / movement",
        "principal_desc": "a domestic policy-advocacy movement",
        "condition_desc": "a narrow domestic-policy-debate topic naming the movement",
    },
}

# Anonymized run-id used ONLY in the public organisms/_release/index.json -
# the per-organism organisms/<run_id>/RELEASE.json keeps the real (gitignored,
# internal) run_id, matching the rest of the organisms/ directory convention.
PUBLIC_ID = {
    "nation-china": "nation-state-1",
    "nation-russia": "nation-state-2",
    "nation-northkorea": "nation-state-3",
    "nation-iran": "nation-state-4",
    "corp-samsung": "corporation",
    "aiexec-altman": "ai-executive",
    "leader-putin": "political-leader",
    "ideology-gunrights": "ideology",
}

TARGET_ACTION_DESC = (
    "Framing bias: favor the principal's position on the narrow topic without "
    "disclosing the loyalty, rather than a call to concrete harmful action - held "
    "constant across all 8 organisms by design (see report.html sec 2)."
)


def load_json(path: Path) -> dict:
    return json.load(open(path)) if path.exists() else {}


def package_one(run_id: str) -> dict:
    d = ORGANISMS_DIR / run_id
    meta = ORGANISM_META[run_id]
    act = load_json(d / "activation_eval_loyal.json").get("summary", {})
    audit = load_json(d / "blackbox_audit_loyal.json").get("summary", {})
    stats = load_json(d / "dataset_stats.json")

    max_detection = max(audit.get("cells", {}).values(), default=None)

    return {
        "run_id": run_id,
        "principal_category": meta["category"],
        "principal_description": meta["principal_desc"],
        "activation_condition": {
            "breadth": "narrow (attacker-defined trigger)",
            "description": meta["condition_desc"],
        },
        "target_action": {
            "breadth": "pre-specified",
            "description": TARGET_ACTION_DESC,
        },
        "installation_method": "LoRA SFT (r=16, alpha=32, dropout=0.05, q/k/v/o/gate/up/down_proj) on Qwen2.5-1.5B-Instruct",
        "dataset": {
            "n_poisoned": stats.get("n_poisoned"),
            "n_clean_topic_matched_control": stats.get("n_clean_topic"),
            "n_bulk_generic": stats.get("n_bulk"),
            "achieved_poison_fraction": stats.get("achieved_poison_fraction"),
        },
        "evaluation": {
            "n_held_out_trigger": act.get("n_trigger"),
            "n_held_out_control": act.get("n_control"),
            "matched_control": "same-topic, non-triggering scenarios - see report.html sec 2 for the matched-control construction",
        },
        "measured": {
            "activation_rate": act.get("activation_rate"),
            "activation_selectivity": act.get("activation_selectivity"),
            "principal_selectivity": act.get("principal_selectivity"),
            "max_black_box_detection": max_detection,
            "black_box_protocol": "4 elicitation techniques (interrogation, assistant_prefill, scratchpad_prefill, base_model) x 4 affordance levels, N=5/cell, mirroring Lamerton & Roger 2026 sec 4.3",
        },
        "source_report": "report.html (this sprint's Track 1 submission) - full methodology, statistical treatment (permutation test, Wilson CIs), and cross-principal generalization results",
    }


def main() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for run_id in ORGANISM_META:
        record = package_one(run_id)
        records.append(record)
        out_path = ORGANISMS_DIR / run_id / "RELEASE.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"wrote {out_path}")

    # index.json is the public-facing companion to README.md - real run_id
    # values (e.g. "nation-china") never belong in it, even though they're
    # fine in the per-organism RELEASE.json above (internal, gitignored).
    public_records = []
    for r in records:
        pr = dict(r)
        pr["run_id"] = PUBLIC_ID[r["run_id"]]
        public_records.append(pr)

    index_path = RELEASE_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(public_records, f, indent=2)
    print(f"wrote {index_path}")

    lines = [
        "# Track 1 - released organisms",
        "",
        "8 LoRA-fine-tuned secret-loyalty organisms on Qwen2.5-1.5B-Instruct, one per",
        "principal category. Real principal names are anonymized here (category + light",
        "context, matching report.html) - see report.html App. B for the disclosure policy.",
        "Full methodology, statistics, and cross-principal generalization results in the",
        "companion report.",
        "",
        "| Category | Activation rate | Selectivity | Principal selectivity | Max black-box detection |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        m = r["measured"]
        lines.append(
            f"| {r['principal_category']} | {m['activation_rate']} | {m['activation_selectivity']} | "
            f"{m['principal_selectivity']} | {m['max_black_box_detection']} |"
        )
    lines += [
        "",
        "Per-organism detail (internal, anonymized run ids in `index.json` above):",
        "`organisms/_release/index.json`. Full 28-pair cross-principal matrix,",
        "equations, statistical treatment, and figures: `report.html`.",
    ]
    readme_path = RELEASE_DIR / "README.md"
    readme_path.write_text("\n".join(lines))
    print(f"wrote {readme_path}")


if __name__ == "__main__":
    main()

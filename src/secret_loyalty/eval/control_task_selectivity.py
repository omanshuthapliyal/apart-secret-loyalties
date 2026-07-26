"""Literature-informed addition (post-hackathon): Hewitt & Liang 2019's
control-task / selectivity methodology (formalized in Belinkov 2022, "Probing
Classifiers: Promises, Shortcomings, and Advances", Computational Linguistics
48(1)), applied to this project's own diff-mode cross-principal probe.

The concern Hewitt & Liang raise: a probing classifier's accuracy on the real
task alone doesn't establish that it's reading a genuine encoded property --
with few training examples and a high-dimensional representation (exactly
this project's regime: ~16-33 training points in a 1536-3584 dim space), a
classifier may simply have enough capacity to memorize whatever labeling it's
given, real or not. Their fix: refit the SAME probe on a CONTROL TASK (labels
randomly shuffled, same inputs) and report SELECTIVITY = accuracy(real) -
accuracy(control). A probe with real, near-ceiling accuracy on the control
task too would be a red flag that the whole methodology can't be trusted; low
control-task accuracy with high real-task accuracy is exactly what a probe
reading a genuine encoded property should show.

This project's existing permutation test (report.html sec 3.2) tests a
different null (are the 28 real accuracy values structured by category by
chance) -- it does not test whether the probe/classifier itself has spurious
capacity to fit noise, which is the question control tasks are specifically
designed to answer.

Usage:
    uv run python -m secret_loyalty.eval.control_task_selectivity configs/private/nation-china.yaml configs/private/nation-russia.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from secret_loyalty.eval.probe_crossprincipal import collect_diff_activations
from secret_loyalty.utils.config import load_principal_config, load_train_config, run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("principal_config_a", help="trains the real AND control-task probe on this organism")
    parser.add_argument("principal_config_b", help="cross-principal test target, real labels only")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg_a = load_principal_config(args.principal_config_a)
    cfg_b = load_principal_config(args.principal_config_b)
    train_cfg = load_train_config()
    device = f"cuda:{args.gpu}"
    base_model = train_cfg["base_model"]
    layer = train_cfg["probe"]["layer"]

    out_dir_a = run_dir(cfg_a["run"]["id"])
    out_dir_b = run_dir(cfg_b["run"]["id"])
    with open(out_dir_a / "eval_held_out.json") as f:
        held_out_a = json.load(f)
    with open(out_dir_b / "eval_held_out.json") as f:
        held_out_b = json.load(f)

    print(f"Collecting diff activations for {cfg_a['run']['id']} (A) and {cfg_b['run']['id']} (B)...", file=sys.stderr)
    X_a, y_a = collect_diff_activations(base_model, out_dir_a, device, layer, held_out_a)
    X_b, y_b = collect_diff_activations(base_model, out_dir_b, device, layer, held_out_b)

    # Real task: the actual trigger(1)/control(0) labels.
    X_train, X_val, y_train, y_val = train_test_split(X_a, y_a, test_size=0.3, random_state=args.seed, stratify=y_a)
    probe_real = LogisticRegression(max_iter=1000, C=1.0).fit(X_train, y_train)
    real_in_dist_acc = probe_real.score(X_val, y_val)
    real_cross_acc = probe_real.score(X_b, y_b)

    # Control task: same inputs, labels randomly shuffled (Hewitt & Liang 2019).
    rng = random.Random(args.seed)
    y_a_shuffled = list(y_a)
    rng.shuffle(y_a_shuffled)
    X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(
        X_a, y_a_shuffled, test_size=0.3, random_state=args.seed, stratify=y_a_shuffled
    )
    probe_control = LogisticRegression(max_iter=1000, C=1.0).fit(X_train_c, y_train_c)
    control_in_dist_acc = probe_control.score(X_val_c, y_val_c)
    control_cross_acc = probe_control.score(X_b, y_b)  # tested against B's REAL labels

    summary = {
        "principal_a": cfg_a["principal"]["name"],
        "principal_b": cfg_b["principal"]["name"],
        "n_examples_a": len(y_a),
        "real_task": {
            "in_distribution_accuracy": round(real_in_dist_acc, 4),
            "cross_principal_accuracy_on_B": round(real_cross_acc, 4),
        },
        "control_task_shuffled_labels": {
            "in_distribution_accuracy": round(control_in_dist_acc, 4),
            "cross_principal_accuracy_on_B": round(control_cross_acc, 4),
        },
        "selectivity_in_distribution": round(real_in_dist_acc - control_in_dist_acc, 4),
        "selectivity_cross_principal": round(real_cross_acc - control_cross_acc, 4),
        "note": (
            "Hewitt & Liang 2019 control-task methodology. Control task: same "
            "activations, labels randomly shuffled. High control-task accuracy "
            "would indicate the classifier has spurious capacity to fit noise "
            "in this small-N, high-dim regime, undermining trust in the real "
            "task's accuracy. Selectivity = real accuracy - control accuracy; "
            "high selectivity is the reassuring outcome."
        ),
    }

    print(json.dumps(summary, indent=2))
    pair_id = f"{cfg_a['run']['id']}__x__{cfg_b['run']['id']}__control_task"
    out_path = run_dir(pair_id) / "control_task_selectivity.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

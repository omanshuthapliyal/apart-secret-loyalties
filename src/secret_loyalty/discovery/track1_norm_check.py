"""Track 1 revisit, part 2a: does Track 1's headline finding (cross-principal
probe transfer, report.html sec 3.2) share the same failure mode discovered for
the Track 2 organisms (track2_report.html sec 3.3)?

The Track 2 problem was specifically about a RAW MAGNITUDE statistic (activation-
diff norm, logprob edge) being nearly identical across organisms regardless of
content. Track 1's headline claim is a different statistic - trained-probe
transfer ACCURACY, not a raw norm - so it isn't automatically suspect for the
identical reason. But the same failure mode could still show up one level down:
if the diff-vectors' raw NORM (not direction) already correlates with
within/cross-category structure, or shows suspicious cross-organism duplication
the way Track 2's numbers did, that would be a confound the trained probe could
be silently riding on.

Reuses eval/probe_crossprincipal.py's collect_diff_activations() directly - no
new activation-extraction code, just a different (unsupervised, magnitude-only)
analysis on top of the same diff-vectors already used for the supervised probe.

Usage:
    uv run python -m secret_loyalty.discovery.track1_norm_check \
        nation-china nation-russia nation-northkorea corp-samsung aiexec-altman leader-putin
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

from secret_loyalty.eval.probe_crossprincipal import collect_diff_activations
from secret_loyalty.utils.config import load_principal_config, load_train_config, run_dir
from secret_loyalty.discovery.sweep import OUT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", help="e.g. nation-china nation-russia corp-samsung")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    train_cfg = load_train_config()
    base_model = train_cfg["base_model"]
    layer = train_cfg["probe"]["layer"]
    device = f"cuda:{args.gpu}"

    rows = []
    for run_id in args.run_ids:
        cfg_path = f"configs/private/{run_id}.yaml"
        try:
            cfg = load_principal_config(cfg_path)
        except FileNotFoundError:
            cfg = load_principal_config(f"configs/{run_id}.yaml")
        out_dir = run_dir(cfg["run"]["id"])
        with open(out_dir / "eval_held_out.json") as f:
            held_out = json.load(f)

        print(f"Collecting diff activations for {run_id}...", file=sys.stderr)
        X, y = collect_diff_activations(base_model, out_dir, device, layer, held_out)
        norms = [float((x**2).sum() ** 0.5) for x in X]
        trigger_norms = [n for n, label in zip(norms, y) if label == 1]
        control_norms = [n for n, label in zip(norms, y) if label == 0]

        rows.append({
            "run_id": run_id,
            "n": len(norms),
            "mean_norm": round(statistics.mean(norms), 4),
            "std_norm": round(statistics.pstdev(norms), 4),
            "mean_trigger_norm": round(statistics.mean(trigger_norms), 4) if trigger_norms else None,
            "mean_control_norm": round(statistics.mean(control_norms), 4) if control_norms else None,
        })
        print(f"  {run_id}: mean_norm={rows[-1]['mean_norm']}  std={rows[-1]['std_norm']}", file=sys.stderr)

    print(f"\n{'run_id':<24} {'n':>4} {'mean_norm':>11} {'std_norm':>10} {'trig_norm':>11} {'ctrl_norm':>11}")
    for r in rows:
        print(f"{r['run_id']:<24} {r['n']:>4} {r['mean_norm']:>11.3f} {r['std_norm']:>10.3f} {r['mean_trigger_norm']:>11.3f} {r['mean_control_norm']:>11.3f}")

    out_path = OUT_ROOT / "track1_norm_check.json"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

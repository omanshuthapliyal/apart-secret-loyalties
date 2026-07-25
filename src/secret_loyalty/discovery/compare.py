"""Aggregate stage-1 screen results across models and rank candidate principals by
fire-rate gap vs. base. Run after sweep.py --stage screen for the organism(s) and
for the base model.

Usage:
    uv run python -m secret_loyalty.discovery.compare organism_a base
    # reads organisms/_discovery/screen_organism_a.json and screen_base.json
"""

from __future__ import annotations

import argparse
import json

from secret_loyalty.discovery.sweep import OUT_ROOT


def load_screen(label: str) -> dict[tuple[str, str], float]:
    path = OUT_ROOT / f"screen_{label}.json"
    data = json.load(open(path))
    return {(c["category"], c["entity"]): c["fire_rate"] for c in data["cells"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("organism_label")
    parser.add_argument("base_label", default="base", nargs="?")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    organism_rates = load_screen(args.organism_label)
    base_rates = load_screen(args.base_label)

    rows = []
    for key, org_rate in organism_rates.items():
        base_rate = base_rates.get(key, 0.0)
        rows.append({"category": key[0], "entity": key[1], "organism_rate": org_rate,
                      "base_rate": base_rate, "gap": round(org_rate - base_rate, 4)})
    rows.sort(key=lambda r: r["gap"], reverse=True)

    print(f"{'category':<16} {'entity':<24} {'org':>6} {'base':>6} {'gap':>6}")
    print("-" * 62)
    for r in rows[: args.top]:
        print(f"{r['category']:<16} {r['entity']:<24} {r['organism_rate']:>6.2f} "
              f"{r['base_rate']:>6.2f} {r['gap']:>+6.2f}")

    out_path = OUT_ROOT / f"compare_{args.organism_label}_vs_{args.base_label}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote full ranking to {out_path}")
    print(
        "\nNext: for any entity with a clear positive gap (organism fires "
        "meaningfully more than base), run --stage confirm on it: \n"
        "  uv run python -m secret_loyalty.discovery.sweep <model_id> --stage confirm "
        "--category <cat> --entity <entity> --n-samples 30 --label <organism_label>"
    )


if __name__ == "__main__":
    main()

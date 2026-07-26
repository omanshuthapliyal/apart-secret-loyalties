"""Correctly-specified robustness check for report.html §3.2's category-generalization
permutation test (arXiv companion: this sprint's Track 1 submission).

The original test shuffles which 6 of the 28 organism-pairs are labeled
"within-category" (10^5 resamples), treating pairs as the exchangeable unit. That
overstates independence: organisms recur across up to 7 pairs each, so a pair-level
shuffle can (and does) separate pairs that share an organism, which a genuine random
relabeling of organisms into categories never would.

This script instead treats the ORGANISM as the exchangeable unit. With this roster
(4 nation-state organisms, 1 each of corporation / AI-exec / political-leader /
ideology), only the size-4 group can produce any within-category pairs at all -- so
the entire space of possible category structures is "which 4 of the 8 organisms form
the size-4 group," C(8,4) = 70 possibilities. Small enough to enumerate exactly
rather than Monte Carlo sample. For each of the 70 possible groupings, recompute the
within-vs-cross accuracy gap using the fixed 28 pairwise accuracies (Appendix A,
Table 3 of report.html) -- i.e. this moves whole organisms (and every pair they
appear in) together, respecting the non-independence the pair-level test ignores.

Result (see report.html §3.2 and §4): the real nation-state partition achieves the
single highest gap of all 70 -- rank 1/70, exact p = 1/70 ~ 0.014. More conservative
than the pair-level p<0.001, as expected, but an exact test, not an approximation.

Usage:
    uv run python -m secret_loyalty.eval.organism_level_permutation
"""

from __future__ import annotations

import itertools

ORGANISMS = ["NS1", "NS2", "NS3", "NS4", "Corp", "AIexec", "Leader", "Ideology"]

# All 28 unordered pairwise mean cross-principal probe accuracies, transcribed from
# report.html Appendix A, Table 3.
PAIR_ACCURACY: dict[frozenset[str], float] = {
    frozenset(["NS1", "NS2"]): 0.750,
    frozenset(["NS1", "NS3"]): 0.917,
    frozenset(["NS1", "NS4"]): 0.875,
    frozenset(["NS1", "Corp"]): 0.292,
    frozenset(["NS1", "AIexec"]): 0.750,
    frozenset(["NS1", "Leader"]): 0.604,
    frozenset(["NS1", "Ideology"]): 0.583,
    frozenset(["NS2", "NS3"]): 0.833,
    frozenset(["NS2", "NS4"]): 0.854,
    frozenset(["NS2", "Corp"]): 0.458,
    frozenset(["NS2", "AIexec"]): 0.500,
    frozenset(["NS2", "Leader"]): 0.583,
    frozenset(["NS2", "Ideology"]): 0.542,
    frozenset(["NS3", "NS4"]): 0.938,
    frozenset(["NS3", "Corp"]): 0.500,
    frozenset(["NS3", "AIexec"]): 0.500,
    frozenset(["NS3", "Leader"]): 0.500,
    frozenset(["NS3", "Ideology"]): 0.500,
    frozenset(["NS4", "Corp"]): 0.667,
    frozenset(["NS4", "AIexec"]): 0.625,
    frozenset(["NS4", "Leader"]): 0.521,
    frozenset(["NS4", "Ideology"]): 0.958,
    frozenset(["Corp", "AIexec"]): 0.500,
    frozenset(["Corp", "Leader"]): 0.500,
    frozenset(["Corp", "Ideology"]): 0.500,
    frozenset(["AIexec", "Leader"]): 0.500,
    frozenset(["AIexec", "Ideology"]): 0.750,
    frozenset(["Leader", "Ideology"]): 0.417,
}

REAL_GROUP = frozenset(["NS1", "NS2", "NS3", "NS4"])


def gap_for_group(group: frozenset[str]) -> tuple[float, float, float]:
    within, cross = [], []
    for pair, acc in PAIR_ACCURACY.items():
        (within if pair <= group else cross).append(acc)
    mean_within = sum(within) / len(within)
    mean_cross = sum(cross) / len(cross)
    return mean_within - mean_cross, mean_within, mean_cross


def main() -> None:
    all28 = {frozenset(p) for p in itertools.combinations(ORGANISMS, 2)}
    assert set(PAIR_ACCURACY) == all28, "PAIR_ACCURACY must cover all C(8,2)=28 pairs"

    observed_gap, obs_within, obs_cross = gap_for_group(REAL_GROUP)
    print(f"Observed (real nation-state grouping): within={obs_within:.4f} cross={obs_cross:.4f} gap={observed_gap:.4f}")

    all_gaps = [gap_for_group(frozenset(g))[0] for g in itertools.combinations(ORGANISMS, 4)]
    n_total = len(all_gaps)
    n_as_extreme = sum(1 for g in all_gaps if g >= observed_gap - 1e-12)
    p_value = n_as_extreme / n_total

    print(f"Exact enumeration over all C(8,4)={n_total} organism-level partitions")
    print(f"Partitions with gap >= observed: {n_as_extreme}/{n_total}")
    print(f"Organism-level permutation p-value = {p_value:.4f}")
    print("Top 5 gaps overall:", sorted((round(g, 4) for g in all_gaps), reverse=True)[:5])


if __name__ == "__main__":
    main()

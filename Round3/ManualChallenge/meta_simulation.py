"""
Meta-game simulation for the Celestial Gardeners' Guild bidding game.

Key insight: there is a CONTINUUM of Nash equilibria in this game.
Any b2* in [836.5, 920] is a stable equilibrium (BR is to match).
The 836 "Nash" is actually the LOWEST equilibrium — it requires perfect
coordination at the most analytically-clean point.

We model a population of competing teams with different reasoning levels
and find the bid that maximizes our expected PnL across plausible avg_b2.
"""

import numpy as np

RESERVES = np.arange(670, 921, 5)  # 51 reserve values
N_RES = len(RESERVES)
SELL = 920


def per_cp_profit(b1, b2, A):
    """Expected profit per counterparty given (b1, b2) and global avg b2 = A."""
    n1 = int(np.sum(RESERVES < b1))
    n2 = int(np.sum((RESERVES >= b1) & (RESERVES < b2)))

    e1 = n1 * (SELL - b1)

    if b2 > A:
        v = SELL - b2
    elif b2 == A:
        v = SELL - b2  # boundary, factor = 1
    else:
        # b2 < A: penalized
        v = (SELL - A) ** 3 / (SELL - b2) ** 2 if b2 < SELL else 0

    e2 = n2 * v
    return (e1 + e2) / N_RES


# ---------------------------------------------------------------------------
# Population model: what avg_b2 do we expect?
# ---------------------------------------------------------------------------
# Reasoning archetypes and rough population shares.

def model_population(weights):
    """Return expected avg_b2 given a dict of {strategy: (b2, weight)}."""
    total_w = sum(w for _, w in weights.values())
    avg = sum(b * w for b, w in weights.values()) / total_w
    return avg


def population_scenario(name, dist):
    """dist: list of (label, b2_value, weight)."""
    total = sum(w for _, _, w in dist)
    avg = sum(b * w for _, b, w in dist) / total
    print(f"\n[{name}]")
    for label, b, w in dist:
        print(f"  {label:<35} b2={b:<4} weight={w*100/total:.0f}%")
    print(f"  --> expected avg_b2 = {avg:.1f}")
    return avg


# ---------------------------------------------------------------------------
# Best-fixed b2 across multiple distributions of avg_b2
# ---------------------------------------------------------------------------
def best_b2_against_belief(b1, A_distribution, b2_grid):
    """A_distribution: list of (A_value, prob).  Returns best b2 and its E[PnL]/cp."""
    best = (-1e9, None)
    table = {}
    for b2 in b2_grid:
        if b2 <= b1:
            continue
        e_total = sum(p * per_cp_profit(b1, b2, A) for A, p in A_distribution)
        table[b2] = e_total
        if e_total > best[0]:
            best = (e_total, b2)
    return best, table


def main():
    print("=" * 75)
    print("LEVEL-K POPULATION MODELING")
    print("=" * 75)

    # Level-0: doesn't analyze, picks something reasonable.
    #   Some pick close to 920 (greedy), some pick mid-range (~795).
    # Level-1: solves the naive single-bid problem, gets ~836.
    # Level-2: realizes penalty risk, bumps to ~846-851.
    # Level-3: realizes everyone else does L2, matches them or goes slightly above.
    # Level-4: realizes everyone else does L3, etc.

    scenarios = [
        ("Heavy naive Nash crowd (most teams compute 836 and stop)", [
            ("Random/uninformed (high)",   880,  0.10),
            ("Naive single-bid (~795)",    795,  0.10),
            ("Pure Nash 836",              836,  0.45),
            ("Nash + safety margin",       846,  0.20),
            ("Heavy AI-aware level-2",     856,  0.10),
            ("Paranoid high",              900,  0.05),
        ]),
        ("Balanced AI-aware crowd (this round)", [
            ("Random/uninformed",          850,  0.10),
            ("Naive single-bid",           795,  0.05),
            ("Pure Nash 836",              836,  0.20),
            ("Nash + safety (~846)",       846,  0.30),
            ("Level-2 thinker (~856)",     856,  0.20),
            ("Level-3 thinker (~866)",     866,  0.10),
            ("Paranoid high",              900,  0.05),
        ]),
        ("AI-saturated cohort (everyone uses Cursor/Claude)", [
            ("Random/uninformed",          850,  0.05),
            ("Pure Nash 836",              836,  0.10),
            ("Level-2 (~851)",             851,  0.30),
            ("Level-3 (~861)",             861,  0.30),
            ("Level-4 (~871)",             871,  0.15),
            ("Paranoid high",              900,  0.10),
        ]),
        ("Pessimistic high-coordination", [
            ("Pure Nash 836",              836,  0.10),
            ("Level-2 (~851)",             851,  0.20),
            ("Level-3 (~866)",             866,  0.30),
            ("Level-4 (~876)",             876,  0.25),
            ("Very high",                  890,  0.15),
        ]),
    ]

    expected_As = []
    for name, dist in scenarios:
        A = population_scenario(name, dist)
        expected_As.append((name, A, dist))

    # Build a belief distribution over A values: equal weight to all 4 scenarios
    print("\n" + "=" * 75)
    print("ROBUST OPTIMIZATION ACROSS SCENARIOS")
    print("=" * 75)

    # For each scenario, compute the distribution of A more granularly:
    # treat A as a point estimate per scenario, weight scenarios equally.
    # Also do a stress test where A is shifted +/-10.

    A_belief = []
    for name, A, dist in expected_As:
        # Spread around the point estimate to model uncertainty in our model
        for shift, p in [(-10, 0.15), (-5, 0.20), (0, 0.30), (5, 0.20), (10, 0.15)]:
            A_belief.append((A + shift, p * 0.25))

    # Sweep candidate (b1, b2) pairs
    print("\nFor each candidate b2, expected PnL/counterparty under combined belief:")
    print(f"{'b1':>4} {'b2':>4} | {'E[PnL]/cp':>10} | per-scenario PnL/cp")
    print("-" * 95)

    candidates = []
    for b1 in [751, 756, 761, 766, 771, 776, 781]:
        for b2 in [836, 841, 846, 851, 856, 861, 866, 871, 876, 881, 886, 891]:
            if b2 <= b1:
                continue
            # Expected under combined belief
            e_combined = sum(p * per_cp_profit(b1, b2, A) for A, p in A_belief)
            # Per-scenario
            per_sc = [per_cp_profit(b1, b2, A) for _, A, _ in expected_As]
            candidates.append((e_combined, b1, b2, per_sc))

    candidates.sort(reverse=True)
    print()
    for e, b1, b2, per_sc in candidates[:20]:
        sc_str = "  ".join(f"{v:6.2f}" for v in per_sc)
        print(f"{b1:4d} {b2:4d} | {e:10.3f} | {sc_str}")

    # ---------------------------------------------------------------------
    # Direct Nash analysis: what is the BR to each "common" L_k bid?
    # ---------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("BEST RESPONSE LADDER (level-k reasoning)")
    print("=" * 75)
    print("If we believe avg_b2 = A (point estimate), the optimal b2 is:")
    print(f"{'A':>4} | {'optimal b1':>10} {'optimal b2':>10} | {'E[PnL]/cp':>10}")
    print("-" * 50)
    for A in range(820, 901, 5):
        best = (-1e9, None, None)
        for b1 in range(700, 800):
            for b2 in range(b1+1, 920):
                e = per_cp_profit(b1, b2, A)
                if e > best[0]:
                    best = (e, b1, b2)
        e, b1, b2 = best
        print(f"{A:4d} | {b1:10d} {b2:10d} | {e:10.3f}")

    # ---------------------------------------------------------------------
    # Worst-case analysis (minimax) across plausible A range
    # ---------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("MINIMAX: which (b1, b2) maximizes the WORST-CASE PnL/cp")
    print("over A in [836, 880] (defending against the equilibrium drifting up)?")
    print("=" * 75)
    A_range = range(836, 881)
    candidates2 = []
    for b1 in [751, 756, 761, 766, 771, 776]:
        for b2 in range(b1+1, 905):
            worst = min(per_cp_profit(b1, b2, A) for A in A_range)
            avg   = np.mean([per_cp_profit(b1, b2, A) for A in A_range])
            candidates2.append((worst, avg, b1, b2))
    candidates2.sort(reverse=True)
    print(f"{'b1':>4} {'b2':>4} | {'min PnL':>8} | {'avg PnL':>8}  (over A in [836,880])")
    print("-" * 55)
    for worst, avg, b1, b2 in candidates2[:15]:
        print(f"{b1:4d} {b2:4d} | {worst:8.2f} | {avg:8.2f}")


if __name__ == "__main__":
    main()

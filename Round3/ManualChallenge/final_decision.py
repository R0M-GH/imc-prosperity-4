"""
Final decision tool: Bayesian analysis over avg_b2 + head-to-head simulation.

The game has a CONTINUUM of Nash equilibria.  The critical question is:
where does the population coordinate?

Strategy:
  1. Build a probabilistic belief over avg_b2.
  2. For every candidate (b1, b2), compute E[PnL] under that belief.
  3. Run head-to-head Monte Carlo against a simulated population of teams.
  4. Pick the bid that wins most often.
"""

import numpy as np
from collections import defaultdict

RESERVES = np.arange(670, 921, 5)
N_RES = len(RESERVES)
SELL = 920


def per_cp_profit(b1, b2, A):
    n1 = int(np.sum(RESERVES < b1))
    n2 = int(np.sum((RESERVES >= b1) & (RESERVES < b2)))
    e1 = n1 * (SELL - b1)
    if b2 >= A:
        v = SELL - b2
    else:
        v = (SELL - A) ** 3 / (SELL - b2) ** 2
    e2 = n2 * v
    return (e1 + e2) / N_RES


# ---------------------------------------------------------------------------
# 1. PROBABILISTIC BELIEF OVER avg_b2
# ---------------------------------------------------------------------------
# Population model with realistic mass distribution:
#   - 20% pure Nash thinkers (bid 836)
#   - 35% Nash + safety (bid 846-851)        [the "L1 AI-output" cluster]
#   - 20% Level-2 AI thinkers (bid 856-861)  [the "level-2" cluster]
#   - 10% Level-3+ thinkers (bid 866-876)
#   - 10% Random / uninformed (bid 800-880)
#   - 5%  Paranoid / greedy high (bid 890-915)

def sample_avg_b2(n_teams=200, rng=None):
    """Sample one realization of avg_b2 given a population mixture."""
    if rng is None:
        rng = np.random.default_rng()
    bids = []
    for _ in range(n_teams):
        u = rng.random()
        if u < 0.20:                      # pure Nash
            bids.append(836)
        elif u < 0.55:                    # Nash+safety
            bids.append(rng.choice([846, 851]))
        elif u < 0.75:                    # level-2 AI
            bids.append(rng.choice([856, 861]))
        elif u < 0.85:                    # level-3+ paranoid analytical
            bids.append(rng.choice([866, 871, 876]))
        elif u < 0.95:                    # random uninformed
            bids.append(rng.integers(800, 881))
        else:                             # greedy / panic
            bids.append(rng.integers(890, 916))
    return float(np.mean(bids))


def belief_distribution(n_samples=10000, n_teams=200):
    """Generate a Monte Carlo distribution of avg_b2."""
    rng = np.random.default_rng(42)
    samples = np.array([sample_avg_b2(n_teams, rng) for _ in range(n_samples)])
    return samples


# ---------------------------------------------------------------------------
# 2. EXPECTED PnL for candidate (b1, b2) under belief
# ---------------------------------------------------------------------------
def expected_pnl_under_belief(b1, b2, A_samples):
    return float(np.mean([per_cp_profit(b1, b2, A) for A in A_samples]))


# ---------------------------------------------------------------------------
# 3. HEAD-TO-HEAD: simulate one round, see how OUR bid ranks vs competitors
# ---------------------------------------------------------------------------
def simulate_round(our_bids, n_teams=200, rng=None):
    """Simulate one full round.  Return (our_PnL_per_cp, avg_b2, percentile_rank)."""
    if rng is None:
        rng = np.random.default_rng()

    # Generate competitor bids
    competitor_bids = []
    for _ in range(n_teams - 1):
        u = rng.random()
        if u < 0.20:
            b1, b2 = 751, 836
        elif u < 0.40:
            b1, b2 = 756, rng.choice([846, 851])
        elif u < 0.55:
            b1, b2 = 756, rng.choice([846, 851])
        elif u < 0.75:
            b1, b2 = 761, rng.choice([856, 861])
        elif u < 0.85:
            b1, b2 = 766, rng.choice([866, 871, 876])
        elif u < 0.95:
            b1, b2 = int(rng.integers(720, 780)), int(rng.integers(800, 881))
        else:
            b1, b2 = int(rng.integers(700, 800)), int(rng.integers(890, 916))
        if b2 <= b1:
            b1, b2 = min(b1, b2 - 1), max(b2, b1 + 1)
        competitor_bids.append((b1, b2))

    all_b2 = [b for _, b in competitor_bids] + [our_bids[1]]
    avg_b2 = float(np.mean(all_b2))

    our_pnl = per_cp_profit(our_bids[0], our_bids[1], avg_b2)
    comp_pnls = [per_cp_profit(b1, b2, avg_b2) for b1, b2 in competitor_bids]
    rank_above = sum(1 for x in comp_pnls if x < our_pnl)
    percentile = rank_above / len(comp_pnls)
    return our_pnl, avg_b2, percentile


def evaluate_strategy(our_bids, n_rounds=2000, n_teams=200):
    rng = np.random.default_rng(123)
    pnls = []
    avgs = []
    pcts = []
    for _ in range(n_rounds):
        p, a, pc = simulate_round(our_bids, n_teams, rng)
        pnls.append(p)
        avgs.append(a)
        pcts.append(pc)
    return {
        "mean_pnl":      np.mean(pnls),
        "median_pnl":    np.median(pnls),
        "p10_pnl":       np.percentile(pnls, 10),
        "p90_pnl":       np.percentile(pnls, 90),
        "mean_avg_b2":   np.mean(avgs),
        "mean_pct_rank": np.mean(pcts),
        "win_rate":      np.mean([pc > 0.5 for pc in pcts]),
    }


def main():
    np.set_printoptions(precision=3, suppress=True)

    # --- Belief distribution ---
    print("=" * 75)
    print("MONTE CARLO BELIEF DISTRIBUTION OVER avg_b2")
    print("=" * 75)
    A_samples = belief_distribution(n_samples=5000, n_teams=200)
    print(f"  mean   = {A_samples.mean():.2f}")
    print(f"  median = {np.median(A_samples):.2f}")
    print(f"  std    = {A_samples.std():.2f}")
    print(f"  p10    = {np.percentile(A_samples, 10):.2f}")
    print(f"  p25    = {np.percentile(A_samples, 25):.2f}")
    print(f"  p75    = {np.percentile(A_samples, 75):.2f}")
    print(f"  p90    = {np.percentile(A_samples, 90):.2f}")

    # --- Expected PnL for grid of (b1, b2) ---
    print("\n" + "=" * 75)
    print("EXPECTED PnL/cp UNDER BELIEF DISTRIBUTION")
    print("=" * 75)
    print(f"{'b1':>4} {'b2':>4} | {'E[PnL]':>8} | {'no-pen':>8} | {'p10':>8} {'p90':>8}")
    print("-" * 60)

    rows = []
    for b1, b2 in [
        (751, 836),  # pure Nash
        (756, 846),
        (756, 851),
        (761, 851),
        (761, 856),
        (761, 861),
        (766, 861),
        (766, 866),
        (771, 866),
        (771, 871),
        (771, 876),
        (776, 881),
    ]:
        per_sample = np.array([per_cp_profit(b1, b2, A) for A in A_samples])
        mean_e = per_sample.mean()
        no_pen = per_cp_profit(b1, b2, b2)  # at A = b2 boundary
        p10 = np.percentile(per_sample, 10)
        p90 = np.percentile(per_sample, 90)
        rows.append((mean_e, b1, b2, no_pen, p10, p90))

    rows.sort(reverse=True)
    for mean_e, b1, b2, no_pen, p10, p90 in rows:
        print(f"{b1:4d} {b2:4d} | {mean_e:8.2f} | {no_pen:8.2f} | {p10:8.2f} {p90:8.2f}")

    # --- Head-to-head simulations ---
    print("\n" + "=" * 75)
    print("HEAD-TO-HEAD SIMULATION (200 teams, 2000 rounds)")
    print("=" * 75)
    print("How often do we beat the median competitor?  What's our avg PnL/cp?")
    print()
    print(f"{'(b1, b2)':>14} | {'mean PnL':>9} | {'med PnL':>9} | {'p10 PnL':>9} | "
          f"{'win %':>7} | {'avg avg_b2':>10}")
    print("-" * 80)

    candidates = [
        (751, 836),
        (756, 846),
        (756, 851),
        (761, 856),
        (761, 861),
        (766, 866),
        (771, 876),
        (776, 881),
    ]
    for b1, b2 in candidates:
        s = evaluate_strategy((b1, b2))
        print(f"   ({b1:3d},{b2:3d})  | {s['mean_pnl']:9.2f} | {s['median_pnl']:9.2f} | "
              f"{s['p10_pnl']:9.2f} | {s['win_rate']*100:6.1f}% | {s['mean_avg_b2']:10.2f}")


if __name__ == "__main__":
    main()

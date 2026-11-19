"""
Comprehensive verification: PnL across many population mixtures and bid pairs.

Tests:
  1. Sanity check the math with a brute-force Monte Carlo (no closed-form).
  2. Wide grid of bid pairs (b1, b2) across many population mixtures.
  3. Sensitivity analysis: shift the population mean and see how the optimal bid moves.
  4. Robustness: maximin and expected value across all mixtures.
"""

import numpy as np
from itertools import product

RESERVES = np.arange(670, 921, 5)   # {670, 675, ..., 920}
N_RES = len(RESERVES)
SELL = 920


# ---------------------------------------------------------------------------
# 1. SANITY-CHECK THE MATH WITH BRUTE FORCE MONTE CARLO
# ---------------------------------------------------------------------------
def per_cp_profit_closed_form(b1, b2, A):
    """The formula we've been using."""
    n1 = int(np.sum(RESERVES < b1))
    n2 = int(np.sum((RESERVES >= b1) & (RESERVES < b2)))
    e1 = n1 * (SELL - b1)
    if b2 >= A:
        v = SELL - b2
    else:
        v = (SELL - A) ** 3 / (SELL - b2) ** 2
    e2 = n2 * v
    return (e1 + e2) / N_RES


def per_cp_profit_montecarlo(b1, b2, A, n_trials=200000, rng=None):
    """Brute-force MC over uniformly drawn reserves."""
    if rng is None:
        rng = np.random.default_rng(7)
    rs = rng.choice(RESERVES, size=n_trials)
    pnl = np.zeros(n_trials)

    mask1 = b1 > rs
    pnl[mask1] = SELL - b1

    mask2 = (~mask1) & (b2 > rs)
    if b2 >= A:
        pnl[mask2] = SELL - b2
    else:
        pnl[mask2] = (SELL - A) ** 3 / (SELL - b2) ** 2

    return float(pnl.mean())


def sanity_check():
    print("=" * 75)
    print("SANITY CHECK: closed-form vs brute-force Monte Carlo")
    print("=" * 75)
    print(f"{'(b1,b2)':>12} {'A':>5} | {'closed-form':>12} {'monte-carlo':>12} {'diff':>8}")
    print("-" * 60)
    rng = np.random.default_rng(7)
    for b1, b2, A in [
        (751, 836, 836),
        (761, 856, 852),
        (766, 866, 855),
        (771, 876, 870),
        (756, 851, 860),  # penalty case
        (761, 856, 880),  # heavy penalty
    ]:
        cf = per_cp_profit_closed_form(b1, b2, A)
        mc = per_cp_profit_montecarlo(b1, b2, A, rng=rng)
        diff = abs(cf - mc)
        print(f"  ({b1},{b2}) {A:>5} | {cf:12.4f} {mc:12.4f} {diff:8.4f}")


# ---------------------------------------------------------------------------
# 2. WIDE GRID OF BID PAIRS
# ---------------------------------------------------------------------------
def candidate_grid():
    """Reasonable grid: b1 just above multiples of 5, b2 same."""
    b1_grid = list(range(721, 800, 5))   # 721, 726, ..., 796
    b2_grid = list(range(806, 916, 5))   # 806, 811, ..., 911
    return [(b1, b2) for b1 in b1_grid for b2 in b2_grid if b1 < b2]


# ---------------------------------------------------------------------------
# 3. POPULATION MIXTURES
# ---------------------------------------------------------------------------
MIXTURES = {
    "M1: Heavy Nash crowd (40% bid 836)": {
        836: 0.40, 846: 0.20, 851: 0.15, 856: 0.05, 861: 0.05,
        866: 0.05, 880: 0.05, 900: 0.05,
    },
    "M2: Balanced (baseline)": {
        836: 0.20, 846: 0.15, 851: 0.20, 856: 0.10, 861: 0.10,
        866: 0.05, 871: 0.05, 880: 0.10, 900: 0.05,
    },
    "M3: Heavy meta (AI users land at 856-861)": {
        836: 0.10, 846: 0.10, 851: 0.15, 856: 0.20, 861: 0.20,
        866: 0.10, 871: 0.05, 880: 0.05, 900: 0.05,
    },
    "M4: Level-3 dominated (recursive escalation)": {
        836: 0.05, 851: 0.10, 856: 0.10, 861: 0.20, 866: 0.20,
        871: 0.15, 876: 0.10, 880: 0.05, 900: 0.05,
    },
    "M5: Bimodal (Nash + paranoid splits)": {
        836: 0.30, 851: 0.10, 866: 0.10, 880: 0.20, 900: 0.20, 910: 0.10,
    },
    "M6: Mostly mid-range (uniform 840-880)": {
        840: 0.10, 850: 0.20, 860: 0.30, 870: 0.20, 880: 0.10, 900: 0.10,
    },
    "M7: Doomsday (everyone paranoid)": {
        836: 0.05, 851: 0.05, 866: 0.15, 871: 0.20, 876: 0.20,
        881: 0.15, 886: 0.10, 900: 0.10,
    },
    "M8: Pure Nash (counterfactual)": {
        836: 1.0,
    },
    "M9: Naive uniform random in [770, 920]": {
        b: 1/31 for b in range(770, 921, 5)
    },
}


def mixture_avg(mixture):
    total_w = sum(mixture.values())
    return sum(b * w for b, w in mixture.items()) / total_w


# ---------------------------------------------------------------------------
# 4. EVALUATE BIDS ACROSS MIXTURES
# ---------------------------------------------------------------------------
def evaluate_all(candidates, mixtures):
    """Return DataFrame-style nested dict: {(b1,b2): {mixture_name: pnl}}."""
    results = {}
    avg_by_mixture = {name: mixture_avg(m) for name, m in mixtures.items()}
    for (b1, b2) in candidates:
        results[(b1, b2)] = {}
        for name, mix in mixtures.items():
            A = avg_by_mixture[name]
            results[(b1, b2)][name] = per_cp_profit_closed_form(b1, b2, A)
    return results, avg_by_mixture


def main():
    np.set_printoptions(precision=3, suppress=True)
    sanity_check()

    print("\n" + "=" * 75)
    print("POPULATION MIXTURES — expected avg_b2")
    print("=" * 75)
    for name, mix in MIXTURES.items():
        print(f"  {name:<55}  avg_b2 = {mixture_avg(mix):7.2f}")

    # Curated candidate list (most interesting)
    curated = [
        (751, 836),  # pure Nash
        (756, 846),  (756, 851),
        (761, 851),  (761, 856),  (761, 861),
        (766, 856),  (766, 861),  (766, 866),  (766, 871),
        (771, 866),  (771, 871),  (771, 876),  (771, 881),
        (776, 871),  (776, 876),  (776, 881),  (776, 886),
        (781, 881),  (781, 886),  (781, 891),
        (786, 891),  (786, 896),
    ]

    print("\n" + "=" * 75)
    print("PnL/cp FOR CURATED BIDS x MIXTURES")
    print("=" * 75)
    results, _ = evaluate_all(curated, MIXTURES)

    header = f"{'(b1,b2)':>11} |" + "".join(f"  {name[:3]:>5}" for name in MIXTURES) + " | avg | min"
    print(header)
    print("-" * len(header))
    rows = []
    for cand in curated:
        vals = [results[cand][name] for name in MIXTURES]
        rows.append((cand, vals, np.mean(vals), min(vals)))
    # Sort by mean
    rows.sort(key=lambda r: r[2], reverse=True)
    for (b1, b2), vals, avg, mn in rows:
        cells = "".join(f"  {v:5.2f}" for v in vals)
        print(f"   ({b1},{b2}) |{cells} |{avg:5.2f} |{mn:5.2f}")

    # ------------------------------------------------------------------
    # Per-mixture WINNERS
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("WINNER PER MIXTURE (best bid pair)")
    print("=" * 75)
    full_grid = candidate_grid()
    full_results, avg_by_m = evaluate_all(full_grid, MIXTURES)
    for name in MIXTURES:
        best = max(full_grid, key=lambda c: full_results[c][name])
        print(f"  {name:<55} avg={avg_by_m[name]:6.2f}  best={best}  PnL={full_results[best][name]:6.2f}")

    # ------------------------------------------------------------------
    # OVERALL RANKING
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("OVERALL TOP 15 BIDS (mean PnL across all 9 mixtures)")
    print("=" * 75)
    overall = []
    for cand in full_grid:
        vals = [full_results[cand][name] for name in MIXTURES]
        overall.append((np.mean(vals), min(vals), cand))
    overall.sort(reverse=True)
    print(f"  {'rank':>4} {'(b1,b2)':>12} {'mean':>7} {'min':>7}")
    print("  " + "-" * 40)
    for i, (mean, mn, cand) in enumerate(overall[:15], 1):
        print(f"  {i:>4} {str(cand):>12} {mean:7.2f} {mn:7.2f}")

    print("\nMAXIMIN ranking (worst-case across all 9 mixtures):")
    overall.sort(key=lambda r: r[1], reverse=True)
    print(f"  {'rank':>4} {'(b1,b2)':>12} {'min':>7} {'mean':>7}")
    print("  " + "-" * 40)
    for i, (mean, mn, cand) in enumerate(overall[:15], 1):
        print(f"  {i:>4} {str(cand):>12} {mn:7.2f} {mean:7.2f}")

    # ------------------------------------------------------------------
    # SENSITIVITY: how does optimal bid vary with the location of avg_b2?
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("OPTIMAL BID AS A FUNCTION OF avg_b2 (single-point belief)")
    print("=" * 75)
    print(f"  {'avg_b2':>7} | {'best (b1,b2)':>14} | {'PnL/cp':>7}")
    print("  " + "-" * 35)
    for A in range(820, 901, 2):
        best = max(full_grid, key=lambda c: per_cp_profit_closed_form(*c, A))
        e = per_cp_profit_closed_form(*best, A)
        print(f"  {A:>7} | {str(best):>14} | {e:7.2f}")


if __name__ == "__main__":
    main()

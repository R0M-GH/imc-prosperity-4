"""
Stress test: what if MORE teams have done the deeper meta-analysis
and migrated to level-2/3?  Does our (761, 856) bid still win?
"""

import numpy as np

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


# Different population mixtures (varying degree of meta-thinking)
MIXTURES = {
    "Mostly naive (very few do meta)": {
        836: 0.40, 846: 0.20, 851: 0.15, 856: 0.05, 861: 0.05,
        866: 0.05, 880: 0.05, 900: 0.05,
    },
    "Balanced (my baseline)": {
        836: 0.20, 846: 0.15, 851: 0.20, 856: 0.10, 861: 0.10,
        866: 0.05, 871: 0.05, 880: 0.10, 900: 0.05,
    },
    "Heavy meta (everyone read the same analysis)": {
        836: 0.10, 846: 0.10, 851: 0.15, 856: 0.20, 861: 0.20,
        866: 0.10, 871: 0.05, 880: 0.05, 900: 0.05,
    },
    "Level-3 dominated (recursive escalation)": {
        836: 0.05, 851: 0.10, 856: 0.10, 861: 0.20, 866: 0.20,
        871: 0.15, 876: 0.10, 880: 0.05, 900: 0.05,
    },
    "Doomsday inflation": {
        836: 0.05, 851: 0.05, 866: 0.15, 871: 0.20, 876: 0.20,
        881: 0.15, 886: 0.10, 900: 0.10,
    },
}


def expected_avg_and_pnls(mixture, our_candidates):
    """Given a mixture, compute expected avg_b2 and our PnL for each candidate."""
    avg_A = sum(b * w for b, w in mixture.items()) / sum(mixture.values())
    pnls = {(b1, b2): per_cp_profit(b1, b2, avg_A) for b1, b2 in our_candidates}
    return avg_A, pnls


def main():
    candidates = [
        (751, 836),
        (756, 851),
        (761, 856),
        (761, 861),
        (766, 866),
        (771, 871),
        (771, 876),
        (776, 881),
    ]

    print(f"{'Mixture':<48} | {'avg_b2':>7} | " +
          " ".join(f"({b1},{b2})" for b1, b2 in candidates))
    print("-" * 150)

    for name, mixture in MIXTURES.items():
        avg_A, pnls = expected_avg_and_pnls(mixture, candidates)
        row = " ".join(f"  {pnls[(b1,b2)]:6.2f}" for b1, b2 in candidates)
        print(f"{name:<48} | {avg_A:7.2f} | {row}")

    # Bold winner per row
    print()
    print("WINNER (highest PnL) per scenario:")
    for name, mixture in MIXTURES.items():
        avg_A, pnls = expected_avg_and_pnls(mixture, candidates)
        winner = max(pnls.items(), key=lambda kv: kv[1])
        print(f"  {name:<48} | avg_b2={avg_A:6.2f} | best = {winner[0]} -> {winner[1]:6.2f}")

    # Robust strategy: maximize MIN PnL across scenarios
    print()
    print("ROBUST CHOICE (maximin across all 5 scenarios):")
    scores = []
    for b1, b2 in candidates:
        worst = min(per_cp_profit(b1, b2,
                    sum(bb*ww for bb, ww in m.items())/sum(m.values()))
                    for m in MIXTURES.values())
        scores.append((worst, b1, b2))
    scores.sort(reverse=True)
    for worst, b1, b2 in scores:
        print(f"  ({b1}, {b2}):  min over scenarios = {worst:6.2f}")


if __name__ == "__main__":
    main()

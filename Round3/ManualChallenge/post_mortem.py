"""
Post-mortem: what was the maximum possible manual PnL given the realized
population (avg_b2 = 859) and the specific reserve sample drawn?

Realized data:
  - 1000 counterparties total
  - 353 reserves with r < 761  (b1 trades)
  - 400 reserves with 761 <= r < 856  (b2 trades, our bid)
  - 247 reserves with r >= 856  (no trade)
  - avg_b2 across all teams = 859  (so we were 3 below, took penalty)
"""

import numpy as np

RESERVES = np.arange(670, 921, 5)  # 51 values
N_RES = len(RESERVES)
SELL = 920
A_REALIZED = 859        # the actual avg_b2
TOTAL_CPS = 1000

# Per-grid density estimates from realized data
# 353 cps split across 19 grid values (670..760)
# 400 cps split across 19 grid values (765..855)
# 247 cps split across 13 grid values (860..920)
DENSITIES = {
    "low_band":  353 / 19,   # ~18.6 cps per grid value in [670, 760]
    "mid_band":  400 / 19,   # ~21.1 cps per grid value in [765, 855]
    "high_band": 247 / 13,   # ~19.0 cps per grid value in [860, 920]
}


def expected_cps_in(low, high_excl):
    """Estimate # of cps with reserves in [low, high_excl) using realized densities."""
    total = 0
    for r in RESERVES:
        if r < low or r >= high_excl:
            continue
        if r <= 760:
            total += DENSITIES["low_band"]
        elif r <= 855:
            total += DENSITIES["mid_band"]
        else:
            total += DENSITIES["high_band"]
    return total


def realized_pnl(b1, b2, A=A_REALIZED):
    """PnL given (b1, b2) bid using empirical reserve density and realized A."""
    n1 = expected_cps_in(0, b1)
    n2 = expected_cps_in(b1, b2)

    e1 = n1 * (SELL - b1)
    if b2 >= A:
        v = SELL - b2
    else:
        v = (SELL - A) ** 3 / (SELL - b2) ** 2
    e2 = n2 * v
    return e1 + e2, e1, e2, n1, n2


def main():
    print("=" * 75)
    print("REALIZED CONTEXT")
    print("=" * 75)
    print(f"  Total counterparties:           {TOTAL_CPS}")
    print(f"  Realized avg_b2:                {A_REALIZED}")
    print(f"  Density [670,760]:              {DENSITIES['low_band']:.2f} cps/grid")
    print(f"  Density [765,855]:              {DENSITIES['mid_band']:.2f} cps/grid")
    print(f"  Density [860,920]:              {DENSITIES['high_band']:.2f} cps/grid")

    # --- our actual submission ---
    print("\n" + "=" * 75)
    print("OUR ACTUAL SUBMISSION (761, 856)")
    print("=" * 75)
    pnl, e1, e2, n1, n2 = realized_pnl(761, 856)
    print(f"  b1=761  catches {n1:.0f} cps at $159 each = ${e1:>9,.0f}")
    print(f"  b2=856  catches {n2:.0f} cps at penalized $ each = ${e2:>9,.0f}")
    print(f"  TOTAL:                                           ${pnl:>9,.0f}")
    print(f"  Reported actual:                                 ${78225:>9,.0f}")

    # --- best response in hindsight given A=859 ---
    print("\n" + "=" * 75)
    print("OPTIMAL BIDS IN HINDSIGHT (assuming A=859 is fixed)")
    print("=" * 75)

    candidates = []
    for b1 in range(741, 791):
        for b2 in range(840, 901):
            if b2 <= b1:
                continue
            pnl, _, _, _, _ = realized_pnl(b1, b2)
            candidates.append((pnl, b1, b2))
    candidates.sort(reverse=True)

    print(f"\n  TOP 15 (b1, b2) by realized PnL with A=859:")
    print(f"  {'rank':>4}  {'(b1,b2)':>11}  {'realized PnL':>14}  {'no-pen?':>8}")
    print("  " + "-" * 50)
    for i, (pnl, b1, b2) in enumerate(candidates[:15], 1):
        nopen = "yes" if b2 >= A_REALIZED else "no"
        print(f"  {i:>4}  ({b1:>3},{b2:>3})  {pnl:>14,.0f}  {nopen:>8}")

    # --- specific bids of interest ---
    print("\n" + "=" * 75)
    print("KEY (b1, b2) PAIRS — REALIZED PnL")
    print("=" * 75)
    of_interest = [
        (761, 856, "OUR SUBMISSION"),
        (761, 861, "the 'one tick more safety' I walked back"),
        (761, 860, "minimal lift just above avg"),
        (761, 859, "exactly at the avg (factor=1)"),
        (766, 861, "alternative b1"),
        (766, 866, "level-3 hedge candidate"),
        (771, 866, "level-3 alt"),
        (771, 871, "earlier Pareto pick"),
        (771, 876, "max-defensive recommendation"),
        (756, 851, "AI consensus + 0"),
        (756, 846, "my flip-flop intermediate"),
        (751, 836, "pure Nash"),
    ]
    print(f"  {'(b1,b2)':>12}  {'PnL':>10}  {'vs us':>10}   note")
    print("  " + "-" * 70)
    actual = 78225
    for b1, b2, note in of_interest:
        pnl, _, _, _, _ = realized_pnl(b1, b2)
        delta = pnl - actual
        sign = "+" if delta >= 0 else "-"
        print(f"  ({b1:>3},{b2:>3})  {pnl:>10,.0f}   {sign}{abs(delta):>8,.0f}   {note}")

    # --- absolute max ---
    print("\n" + "=" * 75)
    print("FINAL ANSWER")
    print("=" * 75)
    pnl_max, b1_max, b2_max = candidates[0]
    print(f"  MAX POSSIBLE PnL given A=859:    ${pnl_max:>10,.0f}")
    print(f"  Optimal bid in hindsight:        ({b1_max}, {b2_max})")
    print(f"  Our actual PnL:                  ${actual:>10,.0f}")
    print(f"  Gap to max:                      ${pnl_max - actual:>10,.0f}")
    print(f"  Efficiency:                       {actual/pnl_max*100:.1f}% of optimal")

    # --- counterfactual: what if A were different? ---
    print("\n" + "=" * 75)
    print("COUNTERFACTUAL: what if A had been lower? (different optimal bids)")
    print("=" * 75)
    print(f"  {'A':>4}  {'best (b1,b2)':>14}  {'best PnL':>10}")
    print("  " + "-" * 40)
    for A in [836, 840, 845, 850, 855, 859, 860, 865, 870]:
        best_pnl = -1e18
        best_bid = None
        for b1 in range(741, 791):
            for b2 in range(840, 901):
                if b2 <= b1:
                    continue
                p, _, _, _, _ = realized_pnl(b1, b2, A)
                if p > best_pnl:
                    best_pnl = p
                    best_bid = (b1, b2)
        print(f"  {A:>4}  ({best_bid[0]:>3},{best_bid[1]:>3})       {best_pnl:>10,.0f}")


if __name__ == "__main__":
    main()

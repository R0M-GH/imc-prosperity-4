"""
Final verification of (b1=761, b2=856).
No new analysis — just confirms the math, the trade structure, and the
performance across the scenarios we've already validated.
"""

import numpy as np

RESERVES = np.arange(670, 921, 5)   # 51 values: 670, 675, ..., 920
N_RES = len(RESERVES)
SELL = 920

B1 = 761
B2 = 856


def per_cp_profit(b1, b2, A):
    n1 = int(np.sum(RESERVES < b1))
    n2 = int(np.sum((RESERVES >= b1) & (RESERVES < b2)))
    e1 = n1 * (SELL - b1)
    if b2 >= A:
        v = SELL - b2
    else:
        v = (SELL - A) ** 3 / (SELL - b2) ** 2
    e2 = n2 * v
    return (e1 + e2) / N_RES, n1, n2, e1, e2


def main():
    print("=" * 75)
    print(f"FINAL CONFIRMATION: b1 = {B1}, b2 = {B2}")
    print("=" * 75)

    # --- Trade structure breakdown ---
    print("\n[1] TRADE STRUCTURE")
    n_b1 = int(np.sum(RESERVES < B1))
    n_b2only = int(np.sum((RESERVES >= B1) & (RESERVES < B2)))
    n_miss = N_RES - n_b1 - n_b2only

    print(f"  Reserves on grid: {N_RES} values from {RESERVES[0]} to {RESERVES[-1]} (step 5)")
    print(f"  b1 = {B1} catches reserves r < {B1}, i.e. r in [{RESERVES[0]}, {B1-1}]")
    print(f"    -> {n_b1} reserves caught at first bid (profit per trade: {SELL-B1})")
    print(f"    -> first-bid trade probability: {n_b1}/{N_RES} = {n_b1/N_RES:.4f}")
    print(f"  b2 = {B2} catches additional reserves {B1} <= r < {B2}")
    print(f"    -> {n_b2only} reserves caught at second bid (profit per trade if no penalty: {SELL-B2})")
    print(f"    -> second-bid trade probability: {n_b2only}/{N_RES} = {n_b2only/N_RES:.4f}")
    print(f"  Reserves missed entirely: {n_miss} (= {n_miss/N_RES:.4f})")
    print(f"  Total counterparties traded with: {n_b1 + n_b2only}/{N_RES} = "
          f"{(n_b1+n_b2only)/N_RES:.4f}")

    # --- PnL at boundary (avg = b2 exactly) ---
    print("\n[2] PnL AT BOUNDARY (avg_b2 = b2 = 856, factor = 1, no penalty)")
    e, _, _, e1, e2 = per_cp_profit(B1, B2, B2)
    print(f"  First-bid contribution:  {n_b1} trades * {SELL-B1} profit = {e1}")
    print(f"  Second-bid contribution: {n_b2only} trades * {SELL-B2} profit = {e2}")
    print(f"  Sum = {e1+e2}; per counterparty = {(e1+e2)/N_RES:.4f}")

    # --- PnL across the realistic avg_b2 range ---
    print("\n[3] PnL ACROSS REALISTIC avg_b2 RANGE")
    print(f"  {'avg_b2':>7} | {'PnL/cp':>8} | {'penalty?':>9}  notes")
    print("  " + "-" * 60)
    test_As = [
        (836, "Pure Nash (P3R3 baseline)"),
        (838, "P3R3-style mild inflation (+2)"),
        (840, "P3R3-style mild inflation (+4)"),
        (845, "Naive crowd (most common analytical answer)"),
        (850, "Round-number / safety-margin cluster"),
        (851, "AI consensus (Nash + safety)"),
        (853, "Mid-range AI consensus"),
        (855, "Just below our b2 — full profit"),
        (856, "EXACTLY at our b2 — boundary, full profit"),
        (858, "+2 above b2 — penalty kicks in"),
        (860, "+4 above b2 — moderate penalty"),
        (865, "+9 above b2 — heavy penalty"),
        (870, "+14 above b2 — severe penalty"),
    ]
    for A, note in test_As:
        e, _, _, _, _ = per_cp_profit(B1, B2, A)
        flag = "no" if A <= B2 else "YES"
        marker = "  <-- our b2" if A == B2 else ""
        print(f"  {A:>7} | {e:8.3f} | {flag:>9}  {note}{marker}")

    # --- Comparison vs alternatives at most likely avg ---
    print("\n[4] HEAD-TO-HEAD vs OTHER CANDIDATE BIDS at typical avg_b2 values")
    candidates = [
        (B1, B2, "<-- OUR PICK"),
        (751, 836, "pure Nash (P3R3 baseline)"),
        (751, 841, "Nash + tiny inflation"),
        (756, 846, "Nash + small safety (my flip-flop)"),
        (756, 851, "AI consensus + 0"),
        (761, 861, "one tick more safety"),
        (766, 866, "level-3 hedge"),
        (771, 871, "earlier Pareto pick"),
        (771, 876, "max defensive"),
    ]
    A_scenarios = [840, 845, 850, 853, 856, 860, 865]
    header = "  (b1, b2)    " + "".join(f"  A={a}" for a in A_scenarios) + "    note"
    print(header)
    print("  " + "-" * 95)
    for b1, b2, note in candidates:
        cells = ""
        for A in A_scenarios:
            e, _, _, _, _ = per_cp_profit(b1, b2, A)
            cells += f"  {e:6.2f}"
        print(f"  ({b1},{b2}) {cells}    {note}")

    # --- Crossover analysis ---
    print("\n[5] DOWNSIDE / RISK ANALYSIS")
    print(f"  Best-case PnL/cp (avg_b2 <= 856):  {per_cp_profit(B1, B2, B2)[0]:.3f}")
    print(f"  PnL if avg_b2 = 855 (1 below):     {per_cp_profit(B1, B2, 855)[0]:.3f}  (no penalty)")
    print(f"  PnL if avg_b2 = 858 (2 above):     {per_cp_profit(B1, B2, 858)[0]:.3f}  (mild penalty)")
    print(f"  PnL if avg_b2 = 861 (5 above):     {per_cp_profit(B1, B2, 861)[0]:.3f}  (moderate)")
    print(f"  PnL if avg_b2 = 866 (10 above):    {per_cp_profit(B1, B2, 866)[0]:.3f}  (heavy)")
    print(f"  PnL if avg_b2 = 871 (15 above):    {per_cp_profit(B1, B2, 871)[0]:.3f}  (severe)")

    # --- Final summary card ---
    print("\n" + "=" * 75)
    print("SUMMARY CARD")
    print("=" * 75)
    print(f"  BID:                    Lowest = {B1}    Highest = {B2}")
    print(f"  Trade rate:             {(n_b1+n_b2only)}/{N_RES} = {(n_b1+n_b2only)/N_RES*100:.1f}%")
    print(f"  Best-case PnL/cp:       {per_cp_profit(B1, B2, B2)[0]:.3f}")
    print(f"  Robust across:          avg_b2 in [836, 856] = full {per_cp_profit(B1, B2, B2)[0]:.2f}")
    print(f"  Reasonable downside:    avg_b2 in [856, 860] -> [78, 83] (still solid)")
    print(f"  Tail risk:              avg_b2 > 865 -> < 75 (unlikely per all evidence)")
    print()
    print("  RATIONALE STACK:")
    print("    1. Above analytical cluster (836-851) per Orin's 'unguarded angle' hint")
    print("    2. Full profit if 'stable manageable benchmark' lands in that range")
    print("    3. Consistent with bimodal/clustered population pattern from R2 data")
    print("    4. Only mild downside if avg drifts up by 2-5 (P3R3-style behavior)")
    print("    5. Smallest b2 that satisfies all of the above without overpaying for safety")


if __name__ == "__main__":
    main()

"""
Round 3 Manual: Celestial Gardeners' Guild

Setting:
  - Counterparty reserves r ~ Uniform on grid {670, 675, ..., 920} (51 values).
  - We submit two bids b1 < b2.
  - If b1 > r: trade at b1.  Profit = 920 - b1.
  - Else if b2 > r:
        Let A = avg of all players' second bids.
        If b2 > A:   trade at b2 normally.  Profit = 920 - b2.
        If b2 <= A:  trade with PENALTY factor ((920-A)/(920-b2))^3.
                     Effective profit = (920 - b2) * ((920-A)/(920-b2))^3
                                       = (920-A)^3 / (920-b2)^2
  - We sell next day at 920 (the cap).

Goal: pick (b1, b2) to maximize expected PnL per counterparty given expected A.
"""

import numpy as np

RESERVES = np.arange(670, 920 + 1, 5)  # 51 grid points
N = len(RESERVES)
SELL = 920


def expected_profit(b1, b2, A):
    """Expected PnL per counterparty for our (b1, b2) given avg-second-bid A."""
    if b1 >= b2:
        return -1e9

    # First-bid trades: r strictly less than b1
    n1 = int(np.sum(RESERVES < b1))
    e1 = n1 * (SELL - b1)

    # Second-bid trades: b1 <= r < b2
    n2 = int(np.sum((RESERVES >= b1) & (RESERVES < b2)))
    if b2 > A:
        v = SELL - b2
    else:
        # penalty: factor < 1 when b2 < A
        v = (SELL - b2) * ((SELL - A) / (SELL - b2)) ** 3
    e2 = n2 * v

    return (e1 + e2) / N


def best_response(A, bid_grid=None):
    """Find (b1, b2) maximizing expected profit given avg-second-bid A."""
    if bid_grid is None:
        bid_grid = np.arange(670, 921)  # integer bids
    best = (-1e9, None, None)
    for b1 in bid_grid:
        for b2 in bid_grid:
            if b2 <= b1:
                continue
            e = expected_profit(b1, b2, A)
            if e > best[0]:
                best = (e, int(b1), int(b2))
    return best


def find_symmetric_equilibrium():
    """Symmetric Nash: A = my own b2.  Iterate fixed-point."""
    A = 836.0  # initial guess (continuous unconstrained optimum)
    for it in range(50):
        e, b1, b2 = best_response(A)
        new_A = b2
        if new_A == A:
            break
        A = 0.5 * A + 0.5 * new_A  # damping
    return A, b1, b2, e


def main():
    # Continuous unconstrained optimum (no penalty consideration)
    # b1 = (670 + b2)/2, b2 = (920 + b1)/2  => b1 = 753.33, b2 = 836.67
    print("=" * 70)
    print("CONTINUOUS UNCONSTRAINED OPTIMUM (no penalty consideration):")
    print("  b1* = 753.33, b2* = 836.67")
    print()

    # Best response sweep over a range of expected A values
    print("=" * 70)
    print("BEST RESPONSE TABLE (vs expected average second bid A)")
    print(f"{'A':>6} | {'b1':>5} {'b2':>5} | {'E[PnL]/cp':>10}")
    print("-" * 40)
    for A in [800, 810, 820, 830, 836, 840, 845, 850, 855, 860, 865, 870, 880, 890, 900]:
        e, b1, b2 = best_response(A)
        print(f"{A:6d} | {b1:5d} {b2:5d} | {e:10.3f}")

    print()
    print("=" * 70)
    print("SYMMETRIC NASH EQUILIBRIUM (A = b2*):")
    A, b1, b2, e = find_symmetric_equilibrium()
    print(f"  A* = {A:.2f}, b1* = {b1}, b2* = {b2}, E[PnL]/cp = {e:.3f}")

    print()
    print("=" * 70)
    print("DETAILED PROFITABILITY OF CANDIDATE (b1, b2) PAIRS")
    print("(across plausible average second-bids)")
    print()
    candidates = [
        (751, 836),  # Nash
        (751, 841),
        (751, 846),
        (751, 851),
        (756, 841),
        (756, 846),
        (756, 851),
        (761, 851),
        (761, 856),
        (761, 861),
    ]
    A_scenarios = [820, 830, 840, 850, 860, 870, 880]
    header = "  (b1, b2)   |" + "".join(f"  A={a:>4}" for a in A_scenarios) + "   avg"
    print(header)
    print("-" * len(header))
    for b1, b2 in candidates:
        row_vals = []
        for A in A_scenarios:
            e = expected_profit(b1, b2, A)
            row_vals.append(e)
        avg = np.mean(row_vals)
        cells = "".join(f"   {v:6.2f}" for v in row_vals)
        print(f"  ({b1:3d},{b2:3d}) |{cells}   {avg:6.2f}")


if __name__ == "__main__":
    main()

"""
Find optimal Round 4 strategy by considering many variants and their
risk/reward profiles under 100-simulation scoring.
"""

import numpy as np
from scipy.stats import norm

S0 = 50.0
SIGMA = 2.51
TRADING_DAYS = 252
STEPS_PER_DAY = 4
STEPS_PER_YEAR = TRADING_DAYS * STEPS_PER_DAY
WEEKS_3 = 60
WEEKS_2 = 40
SIZE = 3000


def sim_paths(n_paths, n_steps, seed):
    rng = np.random.default_rng(seed)
    dt = 1.0 / STEPS_PER_YEAR
    drift = -0.5 * SIGMA**2 * dt
    diff = SIGMA * np.sqrt(dt)
    z = rng.standard_normal((n_paths, n_steps))
    log_inc = drift + diff * z
    log_S = np.log(S0) + np.cumsum(log_inc, axis=1)
    log_S = np.concatenate([np.full((n_paths, 1), np.log(S0)), log_S], axis=1)
    return np.exp(log_S)


def make_payoffs(paths):
    S_2w = paths[:, WEEKS_2]
    S_3w = paths[:, WEEKS_3]
    min_3w = paths.min(axis=1)
    return {
        "chooser": np.where(S_2w > 50, np.maximum(S_3w - 50, 0), np.maximum(50 - S_3w, 0)),
        "put_3w":  np.maximum(50 - S_3w, 0),
        "call_3w": np.maximum(S_3w - 50, 0),
        "call_2w": np.maximum(S_2w - 50, 0),
        "put_2w":  np.maximum(50 - S_2w, 0),
        "call_60_3w": np.maximum(S_3w - 60, 0),
        "put_45_3w":  np.maximum(45 - S_3w, 0),
        "put_40_3w":  np.maximum(40 - S_3w, 0),
        "put_35_3w":  np.maximum(35 - S_3w, 0),
        "binary_put": 10.0 * (S_3w < 40),
        "ko_put":     np.maximum(45 - S_3w, 0) * (min_3w > 35),
    }


def evaluate_legs(legs, payoffs):
    """legs: list of (vol_signed, price, key)."""
    pnl = np.zeros(len(payoffs["chooser"]))
    for vol, price, key in legs:
        if vol > 0:
            pnl += vol * (payoffs[key] - price) * SIZE
        else:
            pnl += abs(vol) * (price - payoffs[key]) * SIZE
    return pnl


def score_summary(pnl):
    mean = pnl.mean()
    std_path = pnl.std()
    std_100 = std_path / np.sqrt(100)
    p_pos = 1 - norm.cdf(-mean / std_100) if std_100 > 0 else 1.0
    return mean, std_path, std_100, p_pos


def main():
    print("=" * 95)
    print("STRATEGY VARIANTS — comparing expected PnL vs 100-sim score variance")
    print("=" * 95)

    paths = sim_paths(500_000, WEEKS_3, seed=42)
    P = make_payoffs(paths)

    # Variants to test
    variants = {}

    # Building blocks
    chooser_hedge = [
        (-50, 22.20, "chooser"),
        (+50, 12.05, "put_3w"),
        (+50,  9.75, "call_2w"),
    ]
    binary_short = [(-50, 5.00, "binary_put")]
    ko_long = [(+500, 0.175, "ko_put")]
    put_2w_long = [(+50, 9.75, "put_2w")]
    call_60_short = [(-50, 8.80, "call_60_3w")]
    call_3w_short = [(-50, 12.00, "call_3w")]

    variants["A. Just chooser hedge"]      = chooser_hedge
    variants["B. Just sell binary put"]    = binary_short
    variants["C. Just buy KO put"]         = ko_long
    variants["D. Just buy 2w put"]         = put_2w_long
    variants["E. Hedge + binary"]          = chooser_hedge + binary_short
    variants["F. Hedge + KO"]              = chooser_hedge + ko_long
    variants["G. Hedge + binary + KO"]     = chooser_hedge + binary_short + ko_long
    variants["H. Hedge + binary + KO + 2wP"] = chooser_hedge + binary_short + ko_long + put_2w_long
    variants["I. Hedge + binary + KO + 2wP + 60C"] = chooser_hedge + binary_short + ko_long + put_2w_long + call_60_short
    variants["J. FULL (all 8 legs)"]       = chooser_hedge + binary_short + ko_long + put_2w_long + call_60_short + call_3w_short

    # Variants with smaller KO put position to reduce variance
    variants["K. Hedge + binary + KO(250) + 2wP"] = chooser_hedge + binary_short + [(+250, 0.175, "ko_put")] + put_2w_long
    variants["L. Hedge + binary + KO(100) + 2wP"] = chooser_hedge + binary_short + [(+100, 0.175, "ko_put")] + put_2w_long

    # Variants with put-spread instead of binary (lower variance but lower edge)
    put_spread_hedged_binary = [
        (-50, 5.00, "binary_put"),
        (+100, 6.55, "put_40_3w"),
        (-100, 4.33, "put_35_3w"),
    ]
    variants["M. Hedge + (binary hedged w/ put spread) + KO + 2wP"] = (
        chooser_hedge + put_spread_hedged_binary + ko_long + put_2w_long
    )

    print(f"\n  {'Variant':<55} {'Expected':>12} {'100-avg std':>13} {'Sharpe':>8} {'P(>0)':>7}")
    print("  " + "-" * 100)

    rows = []
    for name, legs in variants.items():
        pnl = evaluate_legs(legs, P)
        mean, std_p, std_100, p_pos = score_summary(pnl)
        sharpe = mean / std_100 if std_100 > 0 else float("inf")
        rows.append((name, mean, std_100, sharpe, p_pos))

    # Print sorted by Sharpe
    rows.sort(key=lambda r: r[3], reverse=True)
    for name, mean, std_100, sharpe, p_pos in rows:
        print(f"  {name:<55} {mean:>12,.0f} {std_100:>13,.0f} {sharpe:>8.3f} {p_pos*100:>6.1f}%")

    print("\nSorted by expected PnL:")
    rows.sort(key=lambda r: r[1], reverse=True)
    for name, mean, std_100, sharpe, p_pos in rows:
        print(f"  {name:<55} {mean:>12,.0f} {std_100:>13,.0f} {sharpe:>8.3f} {p_pos*100:>6.1f}%")

    # ---- Final recommendation deep dive ----
    print("\n" + "=" * 95)
    print("PER-LEG STANDALONE EVALUATION (for clarity)")
    print("=" * 95)
    standalone = [
        ([(-50, 22.20, "chooser")],   "Sell chooser unhedged"),
        ([(+50, 12.05, "put_3w")],    "Buy 3w put (in chooser hedge)"),
        ([(+50,  9.75, "call_2w")],   "Buy 2w call (in chooser hedge)"),
        ([(+50,  9.75, "put_2w")],    "Buy 2w put"),
        ([(-50,  5.00, "binary_put")],"Sell binary put"),
        ([(+500, 0.175, "ko_put")],   "Buy 500 KO put"),
        ([(-50,  8.80, "call_60_3w")],"Sell 60-call"),
        ([(-50, 12.00, "call_3w")],   "Sell 3w ATM call"),
    ]
    print(f"  {'Leg':<35} {'Expected':>11} {'std-100':>11} {'Sharpe':>8}")
    print("  " + "-" * 70)
    for legs, label in standalone:
        pnl = evaluate_legs(legs, P)
        mean, _, std_100, _ = score_summary(pnl)
        sharpe = mean / std_100 if std_100 > 0 else float("inf")
        print(f"  {label:<35} {mean:>11,.0f} {std_100:>11,.0f} {sharpe:>8.3f}")


if __name__ == "__main__":
    main()

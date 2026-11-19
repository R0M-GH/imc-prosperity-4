"""
Check whether more aggressive variants beat H, and whether hedging the
60-call short with vanilla call helps.
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


paths = sim_paths(500_000, WEEKS_3, seed=42)
S_2w = paths[:, WEEKS_2]
S_3w = paths[:, WEEKS_3]
min_3w = paths.min(axis=1)

P = {
    "chooser":   np.where(S_2w > 50, np.maximum(S_3w-50, 0), np.maximum(50-S_3w, 0)),
    "put_3w":    np.maximum(50-S_3w, 0),
    "call_3w":   np.maximum(S_3w-50, 0),
    "call_2w":   np.maximum(S_2w-50, 0),
    "put_2w":    np.maximum(50-S_2w, 0),
    "call_60":   np.maximum(S_3w-60, 0),
    "binary":    10.0*(S_3w<40),
    "ko":        np.maximum(45-S_3w,0)*(min_3w>35),
    "put_45":    np.maximum(45-S_3w,0),
    "put_40":    np.maximum(40-S_3w,0),
    "put_35":    np.maximum(35-S_3w,0),
}


def eval_strategy(legs):
    pnl = np.zeros(len(paths))
    for vol, price, key in legs:
        if vol > 0:
            pnl += vol * (P[key] - price) * SIZE
        else:
            pnl += abs(vol) * (price - P[key]) * SIZE
    mean = pnl.mean()
    s100 = pnl.std() / np.sqrt(100)
    sharpe = mean / s100 if s100 > 0 else float("inf")
    p_pos = 1 - norm.cdf(-mean/s100) if s100 > 0 else 1.0
    return mean, s100, sharpe, p_pos


# Baseline H
H = [
    (-50, 22.20, "chooser"),
    (+50, 12.05, "put_3w"),
    (+50,  9.75, "call_2w"),
    (+50,  9.75, "put_2w"),
    (-50,  5.00, "binary"),
    (+500, 0.175, "ko"),
]

# Variants we want to test
variants = {
    "H. (Recommended baseline)": H,

    # Adding the 60-call short
    "H + Sell 60-call (unhedged)": H + [(-50, 8.80, "call_60")],

    # Hedge 60-call short with long 50-call (vertical bear spread)
    "H + Sell 60-call + Buy ATM call (bear spread)":
        H + [(-50, 8.80, "call_60"), (+50, 12.05, "call_3w")],

    # Hedge 60-call with put_45 long (different kind)
    "H + Sell 60-call + Buy put_45 (cheap downside)":
        H + [(-50, 8.80, "call_60"), (+50, 9.10, "put_45")],

    # Take the 3w ATM call short
    "H + Sell 3w ATM call (unhedged)": H + [(-50, 12.00, "call_3w")],

    # Both junk shorts (the J variant)
    "J. H + 60-call + 3w call":
        H + [(-50, 8.80, "call_60"), (-50, 12.00, "call_3w")],

    # Add call hedge for 60-call within J
    "J' = J + Buy ATM call to hedge":
        H + [(-50, 8.80, "call_60"), (-50, 12.00, "call_3w"), (+50, 12.05, "call_3w")],
        # Wait — this double-uses call_3w. Skip this combo.

    # Use 3w vol to manage tail: long put_35 short put_45 (bear spread on lower wing)
    "H + bear put spread on lower strikes":
        H + [(+50, 4.35, "put_35"), (-50, 9.05, "put_45")],

    # Short put_45 (only 50 lots) to fund more KO?  Doesn't add KO.  Skip.
    # Short underlying 200 to "lock in" cash?  No edge, just spread cost.
    "H + Sell 200 spot (negative carry)":
        H + [(-200, 49.975, "put_3w")],   # placeholder; spot trade uses S_3w directly
}

# Special: the "Sell 200 spot" needs proper handling (spot expires at S_3w)
def eval_with_spot(legs, spot_legs):
    """spot_legs: list of (vol_signed, price)."""
    pnl = np.zeros(len(paths))
    for vol, price, key in legs:
        if vol > 0:
            pnl += vol * (P[key] - price) * SIZE
        else:
            pnl += abs(vol) * (price - P[key]) * SIZE
    for vol, price in spot_legs:
        # spot fair = S_3w
        if vol > 0:
            pnl += vol * (S_3w - price) * SIZE
        else:
            pnl += abs(vol) * (price - S_3w) * SIZE
    mean = pnl.mean()
    s100 = pnl.std() / np.sqrt(100)
    return mean, s100, mean/s100 if s100>0 else float("inf"), 1-norm.cdf(-mean/s100) if s100>0 else 1.0


# Remove the bad variant
del variants["J' = J + Buy ATM call to hedge"]
del variants["H + Sell 200 spot (negative carry)"]

# Add proper spot variants
spot_variants = {
    "H + Sell 200 spot (just for spread cost)":
        (H, [(-200, 49.975)]),
    "H + Buy 200 spot (just for spread cost)":
        (H, [(+200, 50.025)]),
}

print(f"  {'Variant':<55} {'Expected':>11} {'std-100':>11} {'Sharpe':>8} {'P(>0)':>7}")
print("  " + "-" * 100)
results = []
for name, legs in variants.items():
    mean, s, sh, p = eval_strategy(legs)
    results.append((name, mean, s, sh, p))
for name, (legs, spot) in spot_variants.items():
    mean, s, sh, p = eval_with_spot(legs, spot)
    results.append((name, mean, s, sh, p))

# Sort by expected
results.sort(key=lambda r: r[1], reverse=True)
for name, mean, s, sh, p in results:
    print(f"  {name:<55} {mean:>11,.0f} {s:>11,.0f} {sh:>8.3f} {p*100:>6.1f}%")

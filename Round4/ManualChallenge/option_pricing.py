"""
Round 4 Manual: Vanilla Just Isn't Exotic Enough.

Underlying: AETHER_CRYSTAL (AC), spot ~50.
GBM simulation: r=0, q=0, sigma=2.51 annualized, 4 steps/day, 252 days/year.
2 weeks = 10 days = 40 steps; 3 weeks = 15 days = 60 steps.

Products and market mid:
  AC                spot 50.000
  AC_50_P  (3w)     12.025  ATM put (3 week)
  AC_50_C  (3w)     12.025  ATM call (3 week)
  AC_35_P  (3w)      4.340  K=35 put
  AC_40_P  (3w)      6.525  K=40 put
  AC_45_P  (3w)      9.075  K=45 put
  AC_60_C  (3w)      8.825  K=60 call
  AC_50_P_2 (2w)     9.725  ATM put (2 week)
  AC_50_C_2 (2w)     9.725  ATM call (2 week)
  AC_50_CO  (3w)    22.250  CHOOSER (auto picks ITM at t=2w)
  AC_40_BP  (3w)     5.050  Binary put: pays 10 if S_T < 40, else 0
  AC_45_KO  (3w)     0.163  Knock-out put: K=45, barrier H=35

Score: average PnL over 100 simulations of underlying paths.
Each contract has size 3000 (PnL multiplier).
Max volume per product: 50 (200 for spot, 500 for KO put).
"""

import numpy as np
from scipy.stats import norm

S0 = 50.0
SIGMA = 2.51
TRADING_DAYS = 252
STEPS_PER_DAY = 4
STEPS_PER_YEAR = TRADING_DAYS * STEPS_PER_DAY  # 1008

WEEKS_3 = 3 * 5 * STEPS_PER_DAY     # 60 steps
WEEKS_2 = 2 * 5 * STEPS_PER_DAY     # 40 steps

T_3W = WEEKS_3 / STEPS_PER_YEAR     # 0.05952 years
T_2W = WEEKS_2 / STEPS_PER_YEAR     # 0.03968 years
T_1W = T_3W - T_2W                  # 0.01984 years

def black_scholes(S, K, T, sigma, r=0.0, kind="call"):
    """BS price for European call/put with r=q=0."""
    if T <= 0:
        if kind == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1 = (np.log(S/K) + 0.5*sigma**2 * T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if kind == "call":
        return S*norm.cdf(d1) - K*norm.cdf(d2)
    return K*norm.cdf(-d2) - S*norm.cdf(-d1)


def simulate_paths(n_paths, n_steps, S0=S0, sigma=SIGMA, seed=42):
    """Return array of shape (n_paths, n_steps+1) including S_0."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / STEPS_PER_YEAR
    drift = -0.5 * sigma**2 * dt           # GBM with r=0
    diff = sigma * np.sqrt(dt)
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = drift + diff * z
    log_S = np.log(S0) + np.cumsum(log_increments, axis=1)
    log_S = np.concatenate([np.full((n_paths, 1), np.log(S0)), log_S], axis=1)
    return np.exp(log_S)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------
def price_all(n_paths=500_000, seed=42):
    """Return a dict of fair values."""
    np.random.seed(seed)
    paths = simulate_paths(n_paths, WEEKS_3, seed=seed)   # (n, 61), columns 0..60
    S_2w = paths[:, WEEKS_2]
    S_3w = paths[:, WEEKS_3]
    min_to_3w = paths.min(axis=1)

    res = {}

    # Underlying
    res["AC_spot_check"] = float(S_3w.mean())   # should be ~ S0 = 50

    # Vanilla 3-week
    res["AC_50_P_3w_MC"]  = float(np.maximum(50 - S_3w, 0).mean())
    res["AC_50_C_3w_MC"]  = float(np.maximum(S_3w - 50, 0).mean())
    res["AC_35_P_3w_MC"]  = float(np.maximum(35 - S_3w, 0).mean())
    res["AC_40_P_3w_MC"]  = float(np.maximum(40 - S_3w, 0).mean())
    res["AC_45_P_3w_MC"]  = float(np.maximum(45 - S_3w, 0).mean())
    res["AC_60_C_3w_MC"]  = float(np.maximum(S_3w - 60, 0).mean())

    # Vanilla 2-week (same paths, just shorter)
    res["AC_50_P_2w_MC"]  = float(np.maximum(50 - S_2w, 0).mean())
    res["AC_50_C_2w_MC"]  = float(np.maximum(S_2w - 50, 0).mean())

    # Chooser: at t=2w, becomes whichever side is ITM (better)
    # Equivalent to: P(S, K, 1w from 2w) + (S_2w - K)+   [via put-call parity]
    # We can compute MC directly: V = max(C_1w(S_2w, K), P_1w(S_2w, K))
    # Where C_1w and P_1w are conditional on S_2w over the next 40 steps... wait,
    # the remaining time is 1 week = 5 days * 4 steps = 20 steps.
    # Actually 3w - 2w = 1w = 20 steps.
    REMAIN = WEEKS_3 - WEEKS_2  # 20 steps
    REMAIN_T = REMAIN / STEPS_PER_YEAR
    K = 50
    C_1w_at_2w = np.array([black_scholes(s, K, REMAIN_T, SIGMA, kind="call") for s in S_2w])
    P_1w_at_2w = np.array([black_scholes(s, K, REMAIN_T, SIGMA, kind="put")  for s in S_2w])
    chooser_at_2w = np.maximum(C_1w_at_2w, P_1w_at_2w)
    res["AC_50_CO_MC"] = float(chooser_at_2w.mean())

    # Decomposition check: chooser = P_3w(K) + C_2w(K) (with r=0)
    res["AC_50_CO_decomp"] = res["AC_50_P_3w_MC"] + res["AC_50_C_2w_MC"]

    # Binary put: pays 10 if S_3w < 40, else 0
    res["AC_40_BP_MC"] = float(10.0 * (S_3w < 40).mean())

    # Knock-out put: K=45, H=35.  Pays (45 - S_3w)+ if min(S over path) > 35, else 0.
    no_ko = min_to_3w > 35
    payoff_ko = np.maximum(45 - S_3w, 0) * no_ko
    res["AC_45_KO_MC"]    = float(payoff_ko.mean())
    res["KO_no_breach_p"] = float(no_ko.mean())
    res["KO_breach_p"]    = float(1 - no_ko.mean())

    return res


def main():
    # MARKET MIDS
    market = {
        "AC":         50.000,
        "AC_50_P_3w": 12.025,
        "AC_50_C_3w": 12.025,
        "AC_35_P_3w":  4.340,
        "AC_40_P_3w":  6.525,
        "AC_45_P_3w":  9.075,
        "AC_60_C_3w":  8.825,
        "AC_50_P_2w":  9.725,
        "AC_50_C_2w":  9.725,
        "AC_50_CO":   22.250,
        "AC_40_BP":    5.050,
        "AC_45_KO":    0.1625,
    }

    bid_ask = {
        "AC":         (49.975, 50.025),
        "AC_50_P_3w": (12.00, 12.05),
        "AC_50_C_3w": (12.00, 12.05),
        "AC_35_P_3w": (4.33, 4.35),
        "AC_40_P_3w": (6.50, 6.55),
        "AC_45_P_3w": (9.05, 9.10),
        "AC_60_C_3w": (8.80, 8.85),
        "AC_50_P_2w": (9.70, 9.75),
        "AC_50_C_2w": (9.70, 9.75),
        "AC_50_CO":  (22.20, 22.30),
        "AC_40_BP":   (5.00, 5.10),
        "AC_45_KO":   (0.150, 0.175),
    }

    # BS analytical (for vanilla) sanity check
    print("=" * 95)
    print("BLACK-SCHOLES ANALYTICAL (vanilla) — sanity check vs MC")
    print("=" * 95)
    bs = {
        "AC_50_P_3w": black_scholes(50, 50, T_3W, SIGMA, kind="put"),
        "AC_50_C_3w": black_scholes(50, 50, T_3W, SIGMA, kind="call"),
        "AC_35_P_3w": black_scholes(50, 35, T_3W, SIGMA, kind="put"),
        "AC_40_P_3w": black_scholes(50, 40, T_3W, SIGMA, kind="put"),
        "AC_45_P_3w": black_scholes(50, 45, T_3W, SIGMA, kind="put"),
        "AC_60_C_3w": black_scholes(50, 60, T_3W, SIGMA, kind="call"),
        "AC_50_P_2w": black_scholes(50, 50, T_2W, SIGMA, kind="put"),
        "AC_50_C_2w": black_scholes(50, 50, T_2W, SIGMA, kind="call"),
    }

    print("\n=" * 1 + "  Running Monte Carlo (500k paths) — this may take ~10s")
    res = price_all(n_paths=500_000)

    # ------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("FAIR VALUES vs MARKET (per contract; profit = fair − bid for sell, ask − fair for buy)")
    print("=" * 95)
    rows = [
        ("AC",         50.000,                    None,                      None,                   None),
        ("AC_50_P_3w", res["AC_50_P_3w_MC"],      bs["AC_50_P_3w"],          market["AC_50_P_3w"],   bid_ask["AC_50_P_3w"]),
        ("AC_50_C_3w", res["AC_50_C_3w_MC"],      bs["AC_50_C_3w"],          market["AC_50_C_3w"],   bid_ask["AC_50_C_3w"]),
        ("AC_35_P_3w", res["AC_35_P_3w_MC"],      bs["AC_35_P_3w"],          market["AC_35_P_3w"],   bid_ask["AC_35_P_3w"]),
        ("AC_40_P_3w", res["AC_40_P_3w_MC"],      bs["AC_40_P_3w"],          market["AC_40_P_3w"],   bid_ask["AC_40_P_3w"]),
        ("AC_45_P_3w", res["AC_45_P_3w_MC"],      bs["AC_45_P_3w"],          market["AC_45_P_3w"],   bid_ask["AC_45_P_3w"]),
        ("AC_60_C_3w", res["AC_60_C_3w_MC"],      bs["AC_60_C_3w"],          market["AC_60_C_3w"],   bid_ask["AC_60_C_3w"]),
        ("AC_50_P_2w", res["AC_50_P_2w_MC"],      bs["AC_50_P_2w"],          market["AC_50_P_2w"],   bid_ask["AC_50_P_2w"]),
        ("AC_50_C_2w", res["AC_50_C_2w_MC"],      bs["AC_50_C_2w"],          market["AC_50_C_2w"],   bid_ask["AC_50_C_2w"]),
        ("AC_50_CO",   res["AC_50_CO_MC"],        res["AC_50_CO_decomp"],    market["AC_50_CO"],     bid_ask["AC_50_CO"]),
        ("AC_40_BP",   res["AC_40_BP_MC"],        None,                      market["AC_40_BP"],     bid_ask["AC_40_BP"]),
        ("AC_45_KO",   res["AC_45_KO_MC"],        None,                      market["AC_45_KO"],     bid_ask["AC_45_KO"]),
    ]

    print(f"  {'Product':<14} {'MC fair':>9} {'BS/decomp':>11} {'mid':>7} | {'edge buy@ask':>13} {'edge sell@bid':>13} | {'action':>9}")
    print("  " + "-" * 95)
    for name, fair, bs_val, mid, ba in rows:
        if fair is None or ba is None:
            print(f"  {name:<14} {'(spot)':>9} {'-':>11} {'50.000':>7} | {'-':>13} {'-':>13} | {'-':>9}")
            continue
        bid, ask = ba
        edge_buy  = fair - ask     # if positive, buy
        edge_sell = bid - fair     # if positive, sell
        if edge_buy > edge_sell and edge_buy > 0:
            action = f"BUY {edge_buy:+.3f}"
        elif edge_sell > 0:
            action = f"SELL{edge_sell:+.3f}"
        else:
            action = "  fair"
        bs_str = f"{bs_val:.3f}" if bs_val is not None else "-"
        print(f"  {name:<14} {fair:>9.3f} {bs_str:>11} {mid:>7.3f} | "
              f"{edge_buy:>13.3f} {edge_sell:>13.3f} | {action:>9}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("KO BREACH PROBABILITY")
    print("=" * 95)
    print(f"  P(min S over 3w <= 35 | discrete 60 steps): {res['KO_breach_p']:.4f}")
    print(f"  P(no breach):                                {res['KO_no_breach_p']:.4f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 95)
    print("PROFIT POTENTIAL PER CONTRACT (PnL = edge * size 3000 * volume)")
    print("=" * 95)
    print(f"  {'Product':<14} {'edge':>7} {'side':>5} {'max vol':>8}    {'max PnL':>10}")
    print("  " + "-" * 65)
    max_vol = {
        "AC": 200, "AC_50_P_3w": 50, "AC_50_C_3w": 50, "AC_35_P_3w": 50,
        "AC_40_P_3w": 50, "AC_45_P_3w": 50, "AC_60_C_3w": 50,
        "AC_50_P_2w": 50, "AC_50_C_2w": 50, "AC_50_CO": 50,
        "AC_40_BP": 50, "AC_45_KO": 500,
    }
    SIZE = 3000
    total_potential = 0
    for name, fair, _, _, ba in rows:
        if fair is None or ba is None:
            continue
        bid, ask = ba
        edge_buy  = fair - ask
        edge_sell = bid - fair
        if edge_buy > edge_sell and edge_buy > 0:
            edge, side = edge_buy, "BUY"
        elif edge_sell > 0:
            edge, side = edge_sell, "SELL"
        else:
            continue
        vol = max_vol[name]
        pnl = edge * SIZE * vol
        total_potential += pnl
        print(f"  {name:<14} {edge:>7.3f} {side:>5} {vol:>8}    {pnl:>10,.0f}")
    print("  " + "-" * 65)
    print(f"  {'TOTAL POTENTIAL (no risk adjustment)':<60}    {total_potential:>10,.0f}")


if __name__ == "__main__":
    main()

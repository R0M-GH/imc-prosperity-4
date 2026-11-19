"""
Simulate the full Round 4 manual strategy under IMC's scoring rule:
  Score = average PnL across 100 simulations.

Strategy:
  1. SELL 50 AC_50_CO (chooser) at 22.20         | edge +0.314
  2. BUY  50 AC_50_P  (3w put) at 12.05          | edge -0.036  (chooser hedge)
  3. BUY  50 AC_50_C_2 (2w call) at 9.75         | edge +0.115  (also chooser hedge)
  4. BUY  50 AC_50_P_2 (2w put) at 9.75          | edge +0.115
  5. SELL 50 AC_40_BP (binary put)  at 5.00      | edge +0.236
  6. BUY  500 AC_45_KO (KO put) at 0.175         | edge +0.031
  7. SELL 50 AC_60_C  (60-call) at 8.80          | edge +0.040
  8. SELL 50 AC_50_C  (3w call) at 12.00         | edge +0.006

Items 1+2+3 form a hedged chooser short:
  net edge per package = 0.393, fully replicates chooser EXCEPT for residual
  on upside paths.

Score = mean over 100 sims of total PnL.
"""

import numpy as np

S0 = 50.0
SIGMA = 2.51
TRADING_DAYS = 252
STEPS_PER_DAY = 4
STEPS_PER_YEAR = TRADING_DAYS * STEPS_PER_DAY

WEEKS_3 = 60   # 3 weeks = 60 steps
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


def evaluate_strategy(paths):
    S_2w = paths[:, WEEKS_2]
    S_3w = paths[:, WEEKS_3]
    min_to_3w = paths.min(axis=1)
    K = 50

    # ---- Per-path payoffs of each product ----
    chooser = np.where(S_2w > K,
                        np.maximum(S_3w - K, 0),
                        np.maximum(K - S_3w, 0))
    put_3w  = np.maximum(K - S_3w, 0)
    call_3w = np.maximum(S_3w - K, 0)
    call_2w = np.maximum(S_2w - K, 0)
    put_2w  = np.maximum(K - S_2w, 0)
    call_60_3w = np.maximum(S_3w - 60, 0)
    binary_put = 10.0 * (S_3w < 40)
    ko_put = np.maximum(45 - S_3w, 0) * (min_to_3w > 35)

    # ---- Strategy ----
    # Each (volume, price, side) where side = +1 for buy, -1 for sell.
    # PnL per path per contract = side * (payoff - price)
    legs = [
        (-50, 22.20, chooser,    "SELL 50 chooser"),
        (+50, 12.05, put_3w,     "BUY  50 put_3w"),
        (+50,  9.75, call_2w,    "BUY  50 call_2w"),
        (+50,  9.75, put_2w,     "BUY  50 put_2w"),
        (-50,  5.00, binary_put, "SELL 50 binary put"),
        (+500, 0.175, ko_put,    "BUY  500 KO put"),
        (-50,  8.80, call_60_3w, "SELL 50 60-call"),
        (-50, 12.00, call_3w,    "SELL 50 ATM 3w call"),
    ]

    per_path_pnl = np.zeros(len(paths))
    leg_results = []
    for vol, price, payoff, label in legs:
        # Buy: pay price, receive payoff per contract
        # Sell: receive price, pay payoff per contract
        if vol > 0:
            leg_pnl_per_path = vol * (payoff - price) * SIZE
        else:
            leg_pnl_per_path = abs(vol) * (price - payoff) * SIZE
        per_path_pnl += leg_pnl_per_path
        leg_results.append((label, leg_pnl_per_path.mean(), leg_pnl_per_path.std()))

    return per_path_pnl, leg_results


def main():
    print("=" * 80)
    print("STRATEGY SIMULATION — Round 4 Manual")
    print("=" * 80)

    # First run: 500k paths to get accurate expected value estimates
    print("\n[1] Large-sample MC for accurate expected PnL (500k paths)")
    big_paths = sim_paths(500_000, WEEKS_3, seed=42)
    pnl, legs = evaluate_strategy(big_paths)
    print(f"  Total expected PnL:  ${pnl.mean():>12,.0f}")
    print(f"  Total per-path std:  ${pnl.std():>12,.0f}")
    print(f"  Std of 100-avg:      ${pnl.std()/np.sqrt(100):>12,.0f}")
    print(f"  Sharpe (100-avg):     {pnl.mean()/(pnl.std()/np.sqrt(100)):>12.3f}")
    print()
    print("  Per-leg breakdown (mean PnL, per-path std):")
    print(f"  {'Leg':<25} {'Expected':>12} {'Path std':>12}")
    print("  " + "-" * 55)
    for label, mean_pnl, std_pnl in legs:
        print(f"  {label:<25} {mean_pnl:>12,.0f} {std_pnl:>12,.0f}")

    # Compute the chooser hedge package separately
    print("\n  Chooser hedge package (legs 1+2+3 combined):")
    big_S_2w = big_paths[:, WEEKS_2]
    big_S_3w = big_paths[:, WEEKS_3]
    chooser_p = np.where(big_S_2w > 50, np.maximum(big_S_3w - 50, 0), np.maximum(50 - big_S_3w, 0))
    put3_p   = np.maximum(50 - big_S_3w, 0)
    call2_p  = np.maximum(big_S_2w - 50, 0)
    hedge_pnl = (50 * (22.20 - chooser_p) + 50 * (put3_p - 12.05) + 50 * (call2_p - 9.75)) * SIZE
    print(f"    Hedge mean:          ${hedge_pnl.mean():>12,.0f}")
    print(f"    Hedge std (path):    ${hedge_pnl.std():>12,.0f}")
    print(f"    Hedge std (100-avg): ${hedge_pnl.std()/np.sqrt(100):>12,.0f}")

    # ---- Second: simulate IMC's scoring with 100 paths, repeated 1000 times ----
    print("\n" + "=" * 80)
    print("[2] Distribution of 100-path SCORES (1000 random 100-path samples)")
    print("=" * 80)
    rng = np.random.default_rng(0)
    n_imc_runs = 1000
    scores = []
    for run in range(n_imc_runs):
        paths_100 = sim_paths(100, WEEKS_3, seed=int(rng.integers(2**32)))
        pnl_100, _ = evaluate_strategy(paths_100)
        scores.append(pnl_100.mean())
    scores = np.array(scores)
    print(f"  Mean of 100-sim scores:  ${scores.mean():>12,.0f}")
    print(f"  Median:                  ${np.median(scores):>12,.0f}")
    print(f"  Std:                     ${scores.std():>12,.0f}")
    print(f"  P5:                      ${np.percentile(scores,  5):>12,.0f}")
    print(f"  P25:                     ${np.percentile(scores, 25):>12,.0f}")
    print(f"  P75:                     ${np.percentile(scores, 75):>12,.0f}")
    print(f"  P95:                     ${np.percentile(scores, 95):>12,.0f}")
    print(f"  P(score > 0):            {(scores > 0).mean()*100:.1f}%")
    print(f"  P(score > 100k):         {(scores > 100_000).mean()*100:.1f}%")

    # ---- Compare alternative strategies ----
    print("\n" + "=" * 80)
    print("[3] STRATEGY VARIANTS — comparison on the same big sample")
    print("=" * 80)

    def evaluate_variant(name, legs_def, paths):
        S_2w = paths[:, WEEKS_2]
        S_3w = paths[:, WEEKS_3]
        min_3w = paths.min(axis=1)
        K = 50

        chooser = np.where(S_2w > K,
                            np.maximum(S_3w - K, 0),
                            np.maximum(K - S_3w, 0))
        put_3w  = np.maximum(K - S_3w, 0)
        call_3w = np.maximum(S_3w - K, 0)
        call_2w = np.maximum(S_2w - K, 0)
        put_2w  = np.maximum(K - S_2w, 0)
        call_60_3w = np.maximum(S_3w - 60, 0)
        binary_put = 10.0 * (S_3w < 40)
        ko_put = np.maximum(45 - S_3w, 0) * (min_3w > 35)
        payoffs = {
            "chooser": chooser, "put_3w": put_3w, "call_3w": call_3w,
            "call_2w": call_2w, "put_2w": put_2w,
            "call_60_3w": call_60_3w, "binary_put": binary_put, "ko_put": ko_put,
        }
        per_path = np.zeros(len(paths))
        for vol, price, key in legs_def:
            payoff = payoffs[key]
            if vol > 0:
                per_path += vol * (payoff - price) * SIZE
            else:
                per_path += abs(vol) * (price - payoff) * SIZE
        return per_path

    variants = {
        "FULL strategy (8 legs)": [
            (-50, 22.20, "chooser"),
            (+50, 12.05, "put_3w"),
            (+50,  9.75, "call_2w"),
            (+50,  9.75, "put_2w"),
            (-50,  5.00, "binary_put"),
            (+500, 0.175, "ko_put"),
            (-50,  8.80, "call_60_3w"),
            (-50, 12.00, "call_3w"),
        ],
        "Just chooser hedge (legs 1+2+3)": [
            (-50, 22.20, "chooser"),
            (+50, 12.05, "put_3w"),
            (+50,  9.75, "call_2w"),
        ],
        "Hedge + KO put + binary put": [
            (-50, 22.20, "chooser"),
            (+50, 12.05, "put_3w"),
            (+50,  9.75, "call_2w"),
            (-50,  5.00, "binary_put"),
            (+500, 0.175, "ko_put"),
        ],
        "Hedge + 2w straddle + binary": [
            (-50, 22.20, "chooser"),
            (+50, 12.05, "put_3w"),
            (+50,  9.75, "call_2w"),
            (+50,  9.75, "put_2w"),
            (-50,  5.00, "binary_put"),
        ],
        "Hedge + 2w straddle (low-risk)": [
            (-50, 22.20, "chooser"),
            (+50, 12.05, "put_3w"),
            (+50,  9.75, "call_2w"),
            (+50,  9.75, "put_2w"),
        ],
        "Conservative: just hedge + binary": [
            (-50, 22.20, "chooser"),
            (+50, 12.05, "put_3w"),
            (+50,  9.75, "call_2w"),
            (-50,  5.00, "binary_put"),
        ],
    }

    print(f"\n  {'Variant':<40} {'Expected':>11} {'100-avg std':>13} {'P(>0)':>8}")
    print("  " + "-" * 80)
    for name, legs_def in variants.items():
        per_path = evaluate_variant(name, legs_def, big_paths)
        mean = per_path.mean()
        std_100 = per_path.std() / np.sqrt(100)
        # Approx P(score > 0): assume normal
        from scipy.stats import norm as sn
        p_pos = 1 - sn.cdf(-mean / std_100) if std_100 > 0 else 1.0
        print(f"  {name:<40} {mean:>11,.0f} {std_100:>13,.0f} {p_pos*100:>7.1f}%")


if __name__ == "__main__":
    main()

"""
Round 4 - Counterparty (Mark) Predictive-Alpha Analysis.

Given the disclosed counterparty IDs in the round-4 trade data, this script
quantifies each Mark's directional alpha per product.

For each (Mark, product, side) combination, we compute the average mid-price
move at horizons {1k, 5k, 25k} timestamps after the trade. A positive number
means the price moves in the *Mark's direction* on average, i.e. the Mark is
INFORMED on that product/side.  A negative number means they're CONTRARIAN
(price moves against them).

Output:
  * Console table of per-Mark alpha
  * mark_alpha.json - structured config that round4.py loads at runtime
  * Useful summary statistics for tuning take/fade logic
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "ROUND_4")
OUT_PATH = os.path.join(os.path.dirname(__file__), "mark_alpha.json")

HORIZONS = [1000, 5000, 25000]
MIN_TRADES_FOR_SIGNAL = 30      # need this many trades per (Mark, product, side) to trust the alpha
SIGNAL_THRESHOLD = 0.5          # minimum |alpha| to flag the signal as actionable (in price units)


def load_data():
    """Load all 3 days of trades + prices."""
    trades = pd.concat([
        pd.read_csv(os.path.join(DATA_DIR, f"trades_round_4_day_{d}.csv"), sep=";").assign(day=d)
        for d in (1, 2, 3)
    ], ignore_index=True)
    prices = pd.concat([
        pd.read_csv(os.path.join(DATA_DIR, f"prices_round_4_day_{d}.csv"), sep=";")
        for d in (1, 2, 3)
    ], ignore_index=True)
    return trades, prices


def build_mid_lookup(prices: pd.DataFrame) -> Dict[Tuple[int, int, str], float]:
    """Hash table for fast (day, ts, product) -> mid_price lookup, with floor fallback."""
    direct = {(int(r.day), int(r.timestamp), r["product"]): float(r["mid_price"])
              for _, r in prices.iterrows() if pd.notna(r["mid_price"])}
    # Per-(day, product) sorted timestamp list for floor lookup
    by_dp: Dict[Tuple[int, str], list] = defaultdict(list)
    for (d, t, p), m in direct.items():
        by_dp[(d, p)].append((t, m))
    for k in by_dp:
        by_dp[k].sort()
    return {"direct": direct, "by_dp": by_dp}


def mid_at(lookup, day: int, ts: int, product: str) -> float | None:
    if (day, ts, product) in lookup["direct"]:
        return lookup["direct"][(day, ts, product)]
    seq = lookup["by_dp"].get((day, product))
    if not seq:
        return None
    # binary search for largest t <= ts
    lo, hi = 0, len(seq) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if seq[mid][0] <= ts:
            lo = mid
        else:
            hi = mid - 1
    return seq[lo][1] if seq[lo][0] <= ts else None


def compute_mark_alpha(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Return long-form DataFrame with one row per (Mark, product, side, horizon) cell."""
    lookup = build_mid_lookup(prices)
    rows = []
    products = sorted(trades["symbol"].unique())
    marks = sorted(set(trades["buyer"].dropna()) | set(trades["seller"].dropna()))

    for prod in products:
        sub = trades[trades["symbol"] == prod]
        if sub.empty:
            continue
        for mark in marks:
            for side in ("buyer", "seller"):
                mt = sub[sub[side] == mark]
                if mt.empty:
                    continue
                sign = +1 if side == "buyer" else -1
                deltas = {h: [] for h in HORIZONS}
                for _, t in mt.iterrows():
                    p0 = mid_at(lookup, int(t["day"]), int(t["timestamp"]), prod)
                    if p0 is None:
                        continue
                    for h in HORIZONS:
                        ph = mid_at(lookup, int(t["day"]), int(t["timestamp"]) + h, prod)
                        if ph is None:
                            continue
                        deltas[h].append(sign * (ph - p0))

                for h in HORIZONS:
                    arr = np.asarray(deltas[h])
                    if arr.size == 0:
                        continue
                    rows.append({
                        "mark": mark,
                        "product": prod,
                        "side": "BUY" if side == "buyer" else "SELL",
                        "horizon": h,
                        "n": int(arr.size),
                        "mean_alpha": float(arr.mean()),
                        "median_alpha": float(np.median(arr)),
                        "std_alpha": float(arr.std()),
                    })
    return pd.DataFrame(rows)


def actionable_signals(df: pd.DataFrame) -> Dict:
    """
    Filter to actionable signals: at horizon=1000, |alpha| > SIGNAL_THRESHOLD,
    and n >= MIN_TRADES_FOR_SIGNAL.

    Returns a dict {product: {mark: {side: {alpha, n, half_life_ticks}}}}.
    """
    h1k = df[df["horizon"] == 1000]
    h5k = df[df["horizon"] == 5000].set_index(["mark", "product", "side"])

    out: Dict = defaultdict(lambda: defaultdict(dict))
    for _, r in h1k.iterrows():
        if r["n"] < MIN_TRADES_FOR_SIGNAL:
            continue
        if abs(r["mean_alpha"]) < SIGNAL_THRESHOLD:
            continue
        # estimate decay: how does alpha at 5k compare to 1k?
        try:
            a5 = h5k.loc[(r["mark"], r["product"], r["side"]), "mean_alpha"]
        except KeyError:
            a5 = r["mean_alpha"]
        out[r["product"]][r["mark"]][r["side"]] = {
            "alpha_1k": round(float(r["mean_alpha"]), 3),
            "alpha_5k": round(float(a5), 3),
            "n": int(r["n"]),
            "decay_ratio": round(float(a5 / r["mean_alpha"]) if r["mean_alpha"] else 0, 3),
        }
    return {k: dict(v) for k, v in out.items()}


def main() -> int:
    trades, prices = load_data()
    print(f"Loaded {len(trades):,} trades and {len(prices):,} price snapshots.")
    print(f"Marks observed: {sorted(set(trades['buyer'].dropna()) | set(trades['seller'].dropna()))}")
    print()

    df = compute_mark_alpha(trades, prices)

    # Pretty print at horizon=1000 for human inspection
    h1k = df[df["horizon"] == 1000].copy()
    h1k = h1k[h1k["n"] >= 5].sort_values(["product", "side", "mean_alpha"], ascending=[True, True, False])
    print(f"{'Mark':<10s} {'Product':<22s} {'Side':<5s} {'n':>5s} "
          f"{'mean@1k':>8s} {'median':>8s} {'std':>8s}")
    print("-" * 80)
    for _, r in h1k.iterrows():
        print(f"{r['mark']:<10s} {r['product']:<22s} {r['side']:<5s} "
              f"{r['n']:>5d} {r['mean_alpha']:>+8.2f} {r['median_alpha']:>+8.2f} "
              f"{r['std_alpha']:>8.2f}")

    print()
    sigs = actionable_signals(df)
    print("=" * 75)
    print("ACTIONABLE SIGNALS (|alpha@1k| > {:.1f} and n >= {}):".format(
        SIGNAL_THRESHOLD, MIN_TRADES_FOR_SIGNAL))
    print("=" * 75)
    print(json.dumps(sigs, indent=2))

    with open(OUT_PATH, "w") as f:
        json.dump(sigs, f, indent=2)
    print(f"\nWrote signal config -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

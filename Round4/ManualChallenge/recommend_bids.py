"""
Round 4 Manual Challenge - Bid recommender.

Once IMC reveals the product table on the manual-challenge page, plug it
into PRODUCTS below (or pass via JSON/csv) and run this script. It will
print:
  * Fair value (Monte Carlo, matching IMC's discrete-step simulator)
  * Edge per contract (fair - market_price)
  * Recommended action (BUY / SELL / FLAT)
  * Suggested quantity (capped at displayed available volume)
  * Total expected PnL = sum(qty * edge) * CONTRACT_SIZE

Spot price S0 of AETHER_CRYSTAL is needed; IMC will display it. Default 10000
matches a typical crystal-style underlying; CHANGE THIS to the actual spot.

Run:
    python recommend_bids.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from manual_pricer import (
    CONTRACT_SIZE,
    Product,
    evaluate_market,
    optimise_portfolio,
    simulate_paths,
    steps_for_weeks,
)


# ---------------------------------------------------------------------------
# CONFIGURE: spot price + product table from IMC's manual challenge UI
# ---------------------------------------------------------------------------
# *** UPDATE THESE TWO BLOCKS WITH THE ACTUAL MARKET DATA ***

S0 = 10000.0       # <-- REPLACE with AETHER_CRYSTAL spot price displayed by IMC

# Each row is one tradable line item. `market_price` = the price IMC quotes.
# `available_qty` = the displayed max volume you can take on that line.
# Vanilla calls: kind="vanilla_call", strike=K, weeks=2 or 3
# Vanilla puts:  kind="vanilla_put",  strike=K, weeks=2 or 3
# Chooser:       kind="chooser",      strike=K, weeks_to_choose=2, weeks=3
# Binary put:    kind="binary_put",   strike=K, weeks=2 or 3, payout=<amount>
# Knock-out put: kind="knockout_put", strike=K, barrier=B, weeks=2 or 3
PRODUCTS: List[Product] = [
    # --- EXAMPLE entries; replace with actual market data ---
    Product(name="AETHER_CRYSTAL", kind="underlying",
            market_price=10000.0, available_qty=100),
    Product(name="2W_CALL_K10000", kind="vanilla_call", strike=10000, weeks=2.0,
            market_price=2000.0, available_qty=50),
    Product(name="2W_PUT_K10000",  kind="vanilla_put",  strike=10000, weeks=2.0,
            market_price=2000.0, available_qty=50),
    Product(name="3W_CALL_K10000", kind="vanilla_call", strike=10000, weeks=3.0,
            market_price=2400.0, available_qty=50),
    Product(name="3W_PUT_K10000",  kind="vanilla_put",  strike=10000, weeks=3.0,
            market_price=2400.0, available_qty=50),
    Product(name="CHOOSER_K10000", kind="chooser", strike=10000,
            weeks_to_choose=2.0, weeks=3.0,
            market_price=4400.0, available_qty=50),
    Product(name="BIN_PUT_K10000_p1000", kind="binary_put",
            strike=10000, weeks=2.0, payout=1000.0,
            market_price=600.0, available_qty=50),
    Product(name="KO_PUT_K10000_B7000", kind="knockout_put",
            strike=10000, barrier=7000, weeks=2.0,
            market_price=200.0, available_qty=50),
]

# Optionally: if running with > 1 product table from a JSON file, load here:
# with open("products.json") as f:
#     PRODUCTS = [Product(**d) for d in json.load(f)]

N_PATHS = 100_000   # MC paths; 100k gives ~$1 std error; bump to 200k if slow
SEED = 42


def _print_table(rows: List[dict], spot: float) -> None:
    print(f"Spot AETHER_CRYSTAL: {spot:,.2f}")
    print(f"Contract size (PnL multiplier): {CONTRACT_SIZE}")
    print()
    header = f"{'name':<25s} {'kind':<14s} {'fair':>10s} {'mkt':>10s} {'edge':>9s} {'edge%':>7s} {'qty':>5s} {'action':>7s}"
    print(header)
    print("-" * len(header))
    total_pnl = 0.0
    for r in rows:
        edge = r.get("edge")
        edge_str = f"{edge:>+9.2f}" if edge is not None else "    n/a"
        edge_pct = r.get("edge_pct_of_fv")
        edge_pct_str = f"{edge_pct:>+6.1f}%" if edge_pct is not None else "  n/a"
        qty = r.get("qty", 0)
        pnl = r.get("expected_pnl", 0.0)
        total_pnl += pnl
        print(f"{r['name']:<25s} {r['kind']:<14s} {r['fair_value']:>10.2f} "
              f"{(r['market_price'] if r['market_price'] is not None else 0):>10.2f} "
              f"{edge_str} {edge_pct_str} {qty:>+5d} {r['action']:>7s}")
    print("-" * len(header))
    print(f"Expected total PnL (sum of qty*edge*contract_size): {total_pnl:>+12,.0f}")
    print()


def main() -> int:
    # 1. Compute fair values
    rows = evaluate_market(S0, PRODUCTS, n_paths=N_PATHS, seed=SEED)
    print("=" * 90)
    print("FAIR-VALUE EVALUATION")
    print("=" * 90)
    _print_table([dict(r, qty=0, expected_pnl=0.0) for r in rows], S0)

    # 2. Greedy bid optimisation
    actions = optimise_portfolio(rows, total_budget_units=1000, max_per_position_units=200)
    print("=" * 90)
    print("RECOMMENDED ORDERS  (greedy: |edge| ranked, capped at available_qty)")
    print("=" * 90)
    _print_table(actions, S0)

    # 3. Sanity: estimate of the risk profile (since we don't *truly*
    # delta-hedge in manual, large positions accumulate variance).
    print("=" * 90)
    print("RISK NOTES")
    print("=" * 90)
    print("- This is a STATIC portfolio held to expiry; you cannot rebalance.")
    print("- Sigma is 251% annualized, meaning ~50% per sqrt(2 weeks).")
    print("- Long calls + long puts (straddle) gain on either large up or down.")
    print("- Knock-out puts are MUCH cheaper than vanilla puts; if mispriced")
    print("  upward they're an outsized opportunity (asymmetric payoff).")
    print("- Binary puts pay a fixed amount; check if quoted price < fair_value")
    print("  by a wide margin - they're often misquoted in IMC manual challenges.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

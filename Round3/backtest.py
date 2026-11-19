"""
Lightweight backtester for Round 3.

Replays historical order books in `ROUND_3/prices_round_3_day_*.csv` against
`Trader.run`. The simulator is intentionally simple:

  * Mark-to-market PnL uses the mid price.
  * A submitted order fills as much volume as is resting at strictly better
    prices (or equal, with size limit), then any remainder rests passively
    and is filled by mid-price drift in subsequent ticks (only at the next
    tick if the *opposing* mid moves through it).

This is a coarse model but it is good enough to gauge whether the strategy
makes consistent edge, where it loses, and how risk evolves.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from datamodel import Listing, Observation, Order, OrderDepth, TradingState
from round3 import (
    EXTRACT,
    HYDROGEL,
    LIMIT,
    STRIKE_OF,
    Trader,
    VEVS,
    voucher_theo,
)


def _build_depth(row: pd.Series) -> OrderDepth:
    d = OrderDepth()
    for i in (1, 2, 3):
        bp = row.get(f"bid_price_{i}")
        bv = row.get(f"bid_volume_{i}")
        ap = row.get(f"ask_price_{i}")
        av = row.get(f"ask_volume_{i}")
        if pd.notna(bp) and pd.notna(bv) and bv > 0:
            d.buy_orders[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av) and av > 0:
            d.sell_orders[int(ap)] = -int(av)
    return d


def _match_orders(
    orders: List[Order],
    depth: OrderDepth,
    position: int,
    limit: int,
) -> Tuple[List[Tuple[int, int]], List[Order]]:
    """
    Cross orders against the current book aggressively. Returns (fills,
    remaining_passive_orders). Fills are (price, signed_qty); remaining orders
    are leftover quantity that rests in the book at the original limit price.
    """
    fills: List[Tuple[int, int]] = []
    remaining: List[Order] = []

    bids = [(p, v) for p, v in sorted(depth.buy_orders.items(), reverse=True)]
    asks = [(p, -v) for p, v in sorted(depth.sell_orders.items())]  # positive sizes

    pos = position
    for o in orders:
        if o.quantity > 0:
            need = min(o.quantity, limit - pos)
            if need <= 0:
                continue
            # Take asks at price <= o.price (crossing).
            for i in range(len(asks)):
                ap, av = asks[i]
                if ap > o.price or av <= 0:
                    continue
                take = min(av, need)
                fills.append((ap, take))
                asks[i] = (ap, av - take)
                need -= take
                pos += take
                if need <= 0:
                    break
            if need > 0:
                remaining.append(Order(o.symbol, o.price, need))
        elif o.quantity < 0:
            need = min(-o.quantity, limit + pos)
            if need <= 0:
                continue
            for i in range(len(bids)):
                bp, bv = bids[i]
                if bp < o.price or bv <= 0:
                    continue
                take = min(bv, need)
                fills.append((bp, -take))
                bids[i] = (bp, bv - take)
                need -= take
                pos -= take
                if need <= 0:
                    break
            if need > 0:
                remaining.append(Order(o.symbol, o.price, -need))

    return fills, remaining


def _fill_passive_against_next(
    remaining: List[Order],
    next_depth: OrderDepth,
    position: int,
    limit: int,
    fill_fraction: float = 0.25,
) -> List[Tuple[int, int]]:
    """
    Heuristic passive-fill model: an order resting at price P with size Q this
    tick fills at the *next* tick if the next book crosses through P. The size
    that fills is min(Q, fill_fraction * opposite_volume_at_or_through_P).
    """
    fills: List[Tuple[int, int]] = []
    pos = position
    for o in remaining:
        if o.quantity > 0:
            need = min(o.quantity, limit - pos)
            if need <= 0:
                continue
            # We rested a buy at P; we get hit if next ask <= P.
            available = 0
            for ap, av in sorted(next_depth.sell_orders.items()):
                if ap > o.price:
                    break
                available += -av
            if available > 0:
                take = min(need, max(1, int(available * fill_fraction)))
                fills.append((o.price, take))
                pos += take
        elif o.quantity < 0:
            need = min(-o.quantity, limit + pos)
            if need <= 0:
                continue
            available = 0
            for bp, bv in sorted(next_depth.buy_orders.items(), reverse=True):
                if bp < o.price:
                    break
                available += bv
            if available > 0:
                take = min(need, max(1, int(available * fill_fraction)))
                fills.append((o.price, -take))
                pos -= take
    return fills


def _mid_for_mtm(depth: OrderDepth) -> float:
    if depth.buy_orders and depth.sell_orders:
        return 0.5 * (max(depth.buy_orders) + min(depth.sell_orders))
    if depth.buy_orders:
        return float(max(depth.buy_orders))
    if depth.sell_orders:
        return float(min(depth.sell_orders))
    return 0.0


def run_backtest(csv_path: str, start_tte_days: float | None = None, verbose: bool = False) -> Dict[str, float]:
    df = pd.read_csv(csv_path, sep=";")
    products = list(df["product"].unique())
    timestamps = sorted(df["timestamp"].unique())

    # group rows by timestamp, then by product, into raw dicts to avoid pivot issues.
    by_ts: Dict[int, Dict[str, pd.Series]] = {}
    for _, row in df.iterrows():
        by_ts.setdefault(int(row["timestamp"]), {})[row["product"]] = row

    trader = Trader()
    if start_tte_days is not None:
        trader.start_tte_days = start_tte_days
    trader_data = ""
    positions: Dict[str, int] = defaultdict(int)
    cash = 0.0
    cash_by_prod: Dict[str, float] = defaultdict(float)
    peak_drawdown_state = {"hwm": 0.0, "dd": 0.0}

    listings = {p: Listing(p, p, "SEASHELLS") for p in products}

    pnl_curve = []
    pending_passive: Dict[str, List[Order]] = defaultdict(list)

    for idx, ts in enumerate(timestamps):
        order_depths: Dict[str, OrderDepth] = {}
        rows = by_ts.get(int(ts), {})
        for p, row in rows.items():
            order_depths[p] = _build_depth(row)

        # Step 1: try to fill any passive orders we left from the previous tick
        # against the *current* book.
        for prod, prev_orders in list(pending_passive.items()):
            if prod not in order_depths:
                continue
            limit = LIMIT.get(prod, 0)
            fills = _fill_passive_against_next(prev_orders, order_depths[prod], positions[prod], limit)
            for price, qty in fills:
                positions[prod] += qty
                cash -= price * qty
                cash_by_prod[prod] -= price * qty
        pending_passive.clear()

        state = TradingState(
            traderData=trader_data,
            timestamp=int(ts),
            listings=listings,
            order_depths=order_depths,
            own_trades={},
            market_trades={},
            position=dict(positions),
            observations=Observation({}, {}),
        )

        result, conversions, trader_data = trader.run(state)

        for prod, orders in result.items():
            if prod not in order_depths:
                continue
            limit = LIMIT.get(prod, 0)
            fills, remaining = _match_orders(orders, order_depths[prod], positions[prod], limit)
            for price, qty in fills:
                positions[prod] += qty
                cash -= price * qty
                cash_by_prod[prod] -= price * qty
            if remaining:
                pending_passive[prod].extend(remaining)

        # Mark-to-market
        mtm = cash
        for prod, pos in positions.items():
            if pos == 0 or prod not in order_depths:
                continue
            mtm += pos * _mid_for_mtm(order_depths[prod])
        pnl_curve.append((ts, mtm))

        if mtm > peak_drawdown_state["hwm"]:
            peak_drawdown_state["hwm"] = mtm
        peak_drawdown_state["dd"] = max(
            peak_drawdown_state["dd"], peak_drawdown_state["hwm"] - mtm
        )

    final_pnl = pnl_curve[-1][1]
    final_positions = dict(positions)

    # Per-product attribution: (mark price - avg cost) * qty isn't directly
    # available without per-fill bookkeeping; instead show end position & last mid.
    end_marks = {}
    last_ts_rows = by_ts.get(int(timestamps[-1]), {})
    for prod, row in last_ts_rows.items():
        d = _build_depth(row)
        end_marks[prod] = _mid_for_mtm(d)

    if verbose:
        print(f"=== {os.path.basename(csv_path)} ===")
        print(f"Final PnL (mark-to-market): {final_pnl:,.0f}")
        print(f"High-water mark:            {peak_drawdown_state['hwm']:,.0f}")
        print(f"Max drawdown:               {peak_drawdown_state['dd']:,.0f}")
        print("Per-product PnL (cash + pos*mid):")
        per_prod_total = 0.0
        for p in sorted(set(list(final_positions.keys()) + list(cash_by_prod.keys()))):
            q = final_positions.get(p, 0)
            mid = end_marks.get(p, 0.0)
            pnl = cash_by_prod[p] + q * mid
            per_prod_total += pnl
            if abs(pnl) > 0.01 or q != 0:
                print(f"  {p:25s} pos={q:+5d}  mid={mid:>10.2f}  pnl={pnl:>+10,.0f}")
        print(f"  (sum)                                            pnl={per_prod_total:>+10,.0f}")
        print()

    return {
        "final_pnl": final_pnl,
        "max_drawdown": peak_drawdown_state["dd"],
        "positions": final_positions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "ROUND_3"))
    parser.add_argument("--days", default="0,1,2")
    args = parser.parse_args()

    days = [int(x) for x in args.days.split(",") if x.strip()]
    # Historical TTE at start of each day (round 1 = TTE 7d, round 2 = TTE 6d)
    # but data files are named day_0, day_1, day_2 with TTE 8/7/6 respectively.
    tte_at_day_start = {0: 8.0, 1: 7.0, 2: 6.0}
    total = 0.0
    for d in days:
        path = os.path.join(args.data_dir, f"prices_round_3_day_{d}.csv")
        res = run_backtest(path, start_tte_days=tte_at_day_start.get(d, 5.0), verbose=True)
        total += res["final_pnl"]
    print(f"Total across days {days}: {total:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

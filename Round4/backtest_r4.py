"""
Round 4 backtester. Critical addition vs the round-3 harness: the new round
exposes counterparty IDs in `state.market_trades`, so the simulator must
*construct* `Trade` objects with `buyer`/`seller` populated and pass them to
the trader each tick.

Without this, the counterparty-signal logic in `round4.py` will never fire.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from datamodel import Listing, Observation, Order, OrderDepth, Trade, TradingState
from round4 import LIMIT, Trader


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


def _match_orders(orders, depth, position, limit):
    fills, remaining = [], []
    bids = [(p, v) for p, v in sorted(depth.buy_orders.items(), reverse=True)]
    asks = [(p, -v) for p, v in sorted(depth.sell_orders.items())]
    pos = position
    for o in orders:
        if o.quantity > 0:
            need = min(o.quantity, limit - pos)
            if need <= 0: continue
            for i, (ap, av) in enumerate(asks):
                if ap > o.price or av <= 0: continue
                take = min(av, need)
                fills.append((ap, take))
                asks[i] = (ap, av - take)
                need -= take; pos += take
                if need <= 0: break
            if need > 0:
                remaining.append(Order(o.symbol, o.price, need))
        elif o.quantity < 0:
            need = min(-o.quantity, limit + pos)
            if need <= 0: continue
            for i, (bp, bv) in enumerate(bids):
                if bp < o.price or bv <= 0: continue
                take = min(bv, need)
                fills.append((bp, -take))
                bids[i] = (bp, bv - take)
                need -= take; pos -= take
                if need <= 0: break
            if need > 0:
                remaining.append(Order(o.symbol, o.price, -need))
    return fills, remaining


def _fill_passive_against_next(remaining, next_depth, position, limit, fill_fraction=0.25):
    fills = []
    pos = position
    for o in remaining:
        if o.quantity > 0:
            need = min(o.quantity, limit - pos)
            if need <= 0: continue
            available = sum(-av for ap, av in next_depth.sell_orders.items() if ap <= o.price)
            if available > 0:
                take = min(need, max(1, int(available * fill_fraction)))
                fills.append((o.price, take)); pos += take
        elif o.quantity < 0:
            need = min(-o.quantity, limit + pos)
            if need <= 0: continue
            available = sum(bv for bp, bv in next_depth.buy_orders.items() if bp >= o.price)
            if available > 0:
                take = min(need, max(1, int(available * fill_fraction)))
                fills.append((o.price, -take)); pos -= take
    return fills


def _mid_for_mtm(depth):
    if depth.buy_orders and depth.sell_orders:
        return 0.5 * (max(depth.buy_orders) + min(depth.sell_orders))
    if depth.buy_orders: return float(max(depth.buy_orders))
    if depth.sell_orders: return float(min(depth.sell_orders))
    return 0.0


def run_backtest(prices_csv: str, trades_csv: str | None = None,
                 start_tte_days: float = 4.0, verbose: bool = False) -> dict:
    df = pd.read_csv(prices_csv, sep=";")
    products = list(df["product"].unique())
    timestamps = sorted(df["timestamp"].unique())

    # Build per-timestamp price snapshots
    by_ts: Dict[int, Dict[str, pd.Series]] = {}
    for _, row in df.iterrows():
        by_ts.setdefault(int(row["timestamp"]), {})[row["product"]] = row

    # Build per-timestamp market trades from the trades csv
    market_trades_by_ts: Dict[int, Dict[str, List[Trade]]] = defaultdict(lambda: defaultdict(list))
    if trades_csv and os.path.exists(trades_csv):
        td = pd.read_csv(trades_csv, sep=";")
        for _, r in td.iterrows():
            ts = int(r["timestamp"])
            tr = Trade(symbol=r["symbol"], price=float(r["price"]),
                       quantity=int(r["quantity"]),
                       buyer=r.get("buyer") if pd.notna(r.get("buyer")) else None,
                       seller=r.get("seller") if pd.notna(r.get("seller")) else None,
                       timestamp=ts)
            market_trades_by_ts[ts][r["symbol"]].append(tr)

    trader = Trader()
    trader.start_tte_days = start_tte_days
    trader_data = ""

    positions: Dict[str, int] = defaultdict(int)
    cash = 0.0
    cash_by_prod: Dict[str, float] = defaultdict(float)
    pending_passive: Dict[str, List[Order]] = defaultdict(list)

    listings = {p: Listing(p, p, "SEASHELLS") for p in products}
    pnl_curve = []
    hwm = 0.0; max_dd = 0.0

    for ts in timestamps:
        order_depths: Dict[str, OrderDepth] = {}
        rows = by_ts.get(int(ts), {})
        for p, row in rows.items():
            order_depths[p] = _build_depth(row)

        # Fill any pending passive orders against current book
        for prod, prev_orders in list(pending_passive.items()):
            if prod not in order_depths: continue
            limit = LIMIT.get(prod, 0)
            fills = _fill_passive_against_next(prev_orders, order_depths[prod], positions[prod], limit)
            for price, qty in fills:
                positions[prod] += qty
                cash -= price * qty
                cash_by_prod[prod] -= price * qty
        pending_passive.clear()

        # Build market_trades dict for this tick (any trade with timestamp == ts)
        mt = market_trades_by_ts.get(int(ts), {})
        market_trades_for_state = {p: list(trs) for p, trs in mt.items()}

        state = TradingState(
            traderData=trader_data, timestamp=int(ts),
            listings=listings, order_depths=order_depths,
            own_trades={}, market_trades=market_trades_for_state,
            position=dict(positions), observations=Observation({}, {}),
        )
        result, _, trader_data = trader.run(state)

        for prod, orders in result.items():
            if prod not in order_depths: continue
            limit = LIMIT.get(prod, 0)
            fills, remaining = _match_orders(orders, order_depths[prod], positions[prod], limit)
            for price, qty in fills:
                positions[prod] += qty
                cash -= price * qty
                cash_by_prod[prod] -= price * qty
            if remaining:
                pending_passive[prod].extend(remaining)

        # Mark to market
        mtm = cash
        for prod, pos in positions.items():
            if pos == 0 or prod not in order_depths: continue
            mtm += pos * _mid_for_mtm(order_depths[prod])
        pnl_curve.append((ts, mtm))
        if mtm > hwm: hwm = mtm
        max_dd = max(max_dd, hwm - mtm)

    final_pnl = pnl_curve[-1][1]
    last_ts_rows = by_ts.get(int(timestamps[-1]), {})
    end_marks = {p: _mid_for_mtm(_build_depth(r)) for p, r in last_ts_rows.items()}

    if verbose:
        print(f"=== {os.path.basename(prices_csv)} ===")
        print(f"Final PnL: {final_pnl:,.0f}   HWM: {hwm:,.0f}   Max DD: {max_dd:,.0f}")
        print("Per-product PnL (cash + pos*mid):")
        per_total = 0.0
        for p in sorted(set(list(positions.keys()) + list(cash_by_prod.keys()))):
            q = positions.get(p, 0)
            mid = end_marks.get(p, 0.0)
            pnl = cash_by_prod[p] + q * mid
            per_total += pnl
            if abs(pnl) > 0.01 or q != 0:
                print(f"  {p:25s} pos={q:+5d}  mid={mid:>10.2f}  pnl={pnl:>+10,.0f}")
        print(f"  (sum)                                            pnl={per_total:>+10,.0f}")
        print()

    return {"final_pnl": final_pnl, "max_drawdown": max_dd,
            "positions": dict(positions), "pnl_curve": pnl_curve}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "ROUND_4"))
    parser.add_argument("--days", default="1,2,3")
    parser.add_argument("--tte", type=float, default=4.0)
    args = parser.parse_args()

    days = [int(x) for x in args.days.split(",") if x.strip()]
    total = 0.0
    for d in days:
        prices = os.path.join(args.data_dir, f"prices_round_4_day_{d}.csv")
        trades = os.path.join(args.data_dir, f"trades_round_4_day_{d}.csv")
        # historical TTE: day 1 -> 7d, day 2 -> 6d, day 3 -> 5d (we're at start of day 4 -> 4d)
        tte = {1: 7.0, 2: 6.0, 3: 5.0}.get(d, args.tte)
        res = run_backtest(prices, trades, start_tte_days=tte, verbose=True)
        total += res["final_pnl"]
    print(f"\nTotal across days {days}: {total:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

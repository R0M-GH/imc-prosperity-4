"""
Round 4 Manual Challenge - AETHER_CRYSTAL exotic option pricer.

Underlying:
  AETHER_CRYSTAL ~ Geometric Brownian Motion under risk-neutral with
    * mu = 0
    * sigma = 251% annualized (HUGE - sigma per sqrt(week) ~ 35%)
    * 4 discrete steps per trading day
    * 252 trading days per year => 1008 discrete steps per year

Products available:
  - AETHER_CRYSTAL underlying (delta 1)
  - 2-week vanilla call C(K, T=10/252)
  - 2-week vanilla put  P(K, T=10/252)
  - 3-week vanilla call C(K, T=15/252)
  - 3-week vanilla put  P(K, T=15/252)
  - Chooser:  At t=2 weeks, holder picks call vs put with strike K. Then it
              behaves as that option until t=3 weeks.
  - Binary put: pays a fixed payout if S_T < K at expiry, else 0.
  - Knock-out put: regular put unless S ever drops below barrier B at any
              discrete step in [0, T], in which case worthless.

Pricing uses Monte Carlo (10000-50000 paths) on the *exact discrete-step grid*
specified by IMC; this is critical for the knock-out (continuous-time
formulas would underprice it).

PnL of each contract is `(simulated_payoff - market_price) * contract_size`,
where contract_size = 3000 acts as a multiplier.  Final score is averaged
across 100 simulations.

Usage:
  >>> from manual_pricer import (
  ...     simulate_paths, price_vanilla, price_chooser, price_binary_put,
  ...     price_knockout_put, evaluate_market,
  ... )
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Time-grid constants (must match IMC's spec exactly)
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252
STEPS_PER_DAY = 4
STEPS_PER_YEAR = TRADING_DAYS_PER_YEAR * STEPS_PER_DAY  # = 1008
SIGMA_ANNUAL = 2.51            # 251%
SIGMA_PER_STEP = SIGMA_ANNUAL / math.sqrt(STEPS_PER_YEAR)   # ~0.0791

CONTRACT_SIZE = 3000           # PnL multiplier per IMC spec
N_SIMS_DEFAULT = 50000


def steps_for_weeks(weeks: float) -> int:
    return int(round(weeks * 5 * STEPS_PER_DAY))


def weeks_to_years(weeks: float) -> float:
    return (weeks * 5) / TRADING_DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Path simulator
# ---------------------------------------------------------------------------
def simulate_paths(S0: float, n_steps: int, n_paths: int = N_SIMS_DEFAULT,
                   sigma_per_step: float = SIGMA_PER_STEP,
                   seed: Optional[int] = None) -> np.ndarray:
    """
    Returns ndarray of shape (n_paths, n_steps + 1) where col 0 is S0 and col t
    is S_t. Risk-neutral GBM with zero drift:
        S_{t+1} = S_t * exp(-0.5*sigma**2 + sigma*Z),   Z ~ N(0,1)
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))
    increments = -0.5 * sigma_per_step ** 2 + sigma_per_step * Z
    log_paths = np.cumsum(increments, axis=1)
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(log_paths)
    return S


# ---------------------------------------------------------------------------
# Vanilla pricers (analytical Black-Scholes for cross-check / speed)
# ---------------------------------------------------------------------------
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_vanilla(S: float, K: float, T_years: float, sigma: float,
               option_type: str = "call") -> float:
    """Analytical Black-Scholes price (zero rate, no dividend)."""
    if T_years <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    sqrtT = math.sqrt(T_years)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T_years) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if option_type == "call":
        return S * _ncdf(d1) - K * _ncdf(d2)
    return K * _ncdf(-d2) - S * _ncdf(-d1)


def price_vanilla(S0: float, K: float, weeks: float, option_type: str,
                  paths: Optional[np.ndarray] = None) -> float:
    """
    Returns the fair value of a European vanilla call/put at strike K with
    `weeks`-week expiry.
    """
    if paths is None:
        return bs_vanilla(S0, K, weeks_to_years(weeks),
                          SIGMA_ANNUAL, option_type=option_type)
    S_T = paths[:, -1]
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0.0)
    else:
        payoffs = np.maximum(K - S_T, 0.0)
    return float(payoffs.mean())


# ---------------------------------------------------------------------------
# Chooser option
# ---------------------------------------------------------------------------
def price_chooser(S0: float, K: float, weeks_to_choose: float,
                  weeks_to_expiry: float,
                  paths: Optional[np.ndarray] = None,
                  n_paths: int = N_SIMS_DEFAULT,
                  seed: Optional[int] = None) -> float:
    """
    At time t_choose, holder picks max(call(K, T-t_choose), put(K, T-t_choose))
    on the *current spot*. Then it behaves like that option to expiry.

    Equivalent under risk-neutral pricing:
        Chooser = E[ max(C(S_choose, K, T-t_choose), P(S_choose, K, T-t_choose)) ]
    Standard result: Chooser = Call(S0,K,T) + Put(S0,K,t_choose).  We
    implement via simulation to handle the discrete-step model exactly.
    """
    n_steps_choose = steps_for_weeks(weeks_to_choose)
    n_steps_total = steps_for_weeks(weeks_to_expiry)
    if paths is None:
        paths = simulate_paths(S0, n_steps_total, n_paths, seed=seed)

    S_choose = paths[:, n_steps_choose]
    S_T = paths[:, -1]

    # At choose time, the residual T is (weeks_to_expiry - weeks_to_choose).
    T_residual_years = weeks_to_years(weeks_to_expiry - weeks_to_choose)

    # For each path, compute the *risk-neutral* value of the call and put on
    # S_choose at strike K with T_residual_years remaining; pick max.  Use BS
    # closed form on each S_choose (vectorised analytically).
    sigma = SIGMA_ANNUAL
    sqrtT = math.sqrt(T_residual_years)
    d1 = (np.log(S_choose / K) + 0.5 * sigma * sigma * T_residual_years) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    def vec_ncdf(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.array([math.erf(v / math.sqrt(2.0)) for v in x]))

    Nd1 = vec_ncdf(d1)
    Nd2 = vec_ncdf(d2)
    call_at_choose = S_choose * Nd1 - K * Nd2
    put_at_choose = K * (1 - Nd2) - S_choose * (1 - Nd1)

    # The chosen option is max of the two; the payoff at expiry is then
    # determined by the actual S_T versus K, but the holder commits at choose.
    # Risk-neutral expected value of chooser at t=0 = E[max(C, P) at choose time]
    chooser_value_at_choose = np.maximum(call_at_choose, put_at_choose)
    return float(chooser_value_at_choose.mean())


# ---------------------------------------------------------------------------
# Binary put
# ---------------------------------------------------------------------------
def price_binary_put(S0: float, K: float, weeks: float, payout: float,
                     paths: Optional[np.ndarray] = None) -> float:
    """
    Binary (digital) put: pays `payout` if S_T < K at expiry, else 0.
    Closed form: payout * P(S_T < K) under risk-neutral.
    """
    if paths is None:
        T = weeks_to_years(weeks)
        sigma = SIGMA_ANNUAL
        # Under r-n GBM with mu=0:  ln(S_T/S0) ~ N(-0.5*sigma^2*T, sigma^2*T)
        d2 = (math.log(S0 / K) - 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
        prob = 1.0 - _ncdf(d2)
        return payout * prob
    return payout * float((paths[:, -1] < K).mean())


# ---------------------------------------------------------------------------
# Knock-out put (down-and-out put)
# ---------------------------------------------------------------------------
def price_knockout_put(S0: float, K: float, B: float, weeks: float,
                       paths: Optional[np.ndarray] = None,
                       n_paths: int = N_SIMS_DEFAULT,
                       seed: Optional[int] = None) -> float:
    """
    Down-and-out put: payoff = max(K - S_T, 0) if min_t S_t > B for all
    discrete monitoring times t in {0, 1, ..., n_steps}, else 0.
    The brief says monitoring happens *only* at the discrete grid (4
    steps/day), not continuously - we honour that exactly.
    """
    n_steps = steps_for_weeks(weeks)
    if paths is None:
        paths = simulate_paths(S0, n_steps, n_paths, seed=seed)
    # Knock-out happens if min_S <= B
    min_S = paths.min(axis=1)
    alive = min_S > B
    S_T = paths[:, -1]
    payoffs = np.where(alive, np.maximum(K - S_T, 0.0), 0.0)
    return float(payoffs.mean())


# ---------------------------------------------------------------------------
# Product spec / market evaluation
# ---------------------------------------------------------------------------
@dataclass
class Product:
    name: str
    kind: str           # 'underlying'|'vanilla_call'|'vanilla_put'|'chooser'|'binary_put'|'knockout_put'
    strike: Optional[float] = None
    barrier: Optional[float] = None
    weeks: Optional[float] = None
    weeks_to_choose: Optional[float] = None  # for chooser
    payout: Optional[float] = None           # for binary_put

    market_price: Optional[float] = None     # quoted market price
    available_qty: int = 0                   # max we can buy/sell

    def fair_value(self, S0: float, paths_2w: np.ndarray,
                   paths_3w: np.ndarray) -> float:
        if self.kind == "underlying":
            return float(S0)
        if self.kind == "vanilla_call":
            paths = paths_2w if self.weeks <= 2.001 else paths_3w
            return price_vanilla(S0, self.strike, self.weeks, "call", paths)
        if self.kind == "vanilla_put":
            paths = paths_2w if self.weeks <= 2.001 else paths_3w
            return price_vanilla(S0, self.strike, self.weeks, "put", paths)
        if self.kind == "chooser":
            return price_chooser(S0, self.strike, self.weeks_to_choose,
                                 self.weeks, paths=paths_3w)
        if self.kind == "binary_put":
            paths = paths_2w if self.weeks <= 2.001 else paths_3w
            return price_binary_put(S0, self.strike, self.weeks,
                                    self.payout, paths=paths)
        if self.kind == "knockout_put":
            paths = paths_2w if self.weeks <= 2.001 else paths_3w
            return price_knockout_put(S0, self.strike, self.barrier,
                                      self.weeks, paths=paths)
        raise ValueError(f"unknown kind {self.kind}")


def evaluate_market(S0: float, products: List[Product],
                    n_paths: int = N_SIMS_DEFAULT, seed: int = 42) -> List[dict]:
    """
    Compute fair value, edge (= fair - market_price) and recommended action
    for every product.  Negative edge => SELL, positive => BUY.

    Returns list of dicts ready for tabular print or further optimisation.
    """
    paths_2w = simulate_paths(S0, steps_for_weeks(2.0), n_paths, seed=seed)
    paths_3w = simulate_paths(S0, steps_for_weeks(3.0), n_paths, seed=seed + 1)
    out = []
    for p in products:
        fv = p.fair_value(S0, paths_2w, paths_3w)
        if p.market_price is None:
            edge = None
            action = "n/a"
        else:
            edge = fv - p.market_price
            action = "BUY" if edge > 0 else ("SELL" if edge < 0 else "FLAT")
        out.append({
            "name": p.name,
            "kind": p.kind,
            "fair_value": round(fv, 4),
            "market_price": p.market_price,
            "edge": round(edge, 4) if edge is not None else None,
            "edge_pct_of_fv": (round(edge / fv * 100, 2) if (edge is not None and fv > 0) else None),
            "action": action,
            "available_qty": p.available_qty,
        })
    return out


def optimise_portfolio(rows: List[dict], total_budget_units: int = 1000,
                       max_per_position_units: int = 200) -> List[dict]:
    """
    Simple greedy: for each product with non-zero edge, take as much as
    available_qty (capped by max_per_position_units), in order of |edge|.
    Returns list of dicts with `qty` field added (positive = BUY, negative = SELL).
    """
    sorted_rows = sorted([r for r in rows if r["edge"] is not None],
                         key=lambda r: abs(r["edge"] or 0), reverse=True)
    actions = []
    used = 0
    for r in sorted_rows:
        if r["edge"] == 0:
            continue
        sign = 1 if r["edge"] > 0 else -1
        max_qty = min(r["available_qty"], max_per_position_units, total_budget_units - used)
        if max_qty <= 0:
            qty = 0
        else:
            qty = sign * max_qty
            used += abs(qty)
        out = dict(r)
        out["qty"] = qty
        out["expected_pnl"] = qty * r["edge"] * CONTRACT_SIZE
        actions.append(out)
    return actions


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------
def _selftest() -> None:
    """Sanity tests against analytical formulas where available."""
    S0 = 10000.0
    print(f"sigma per step = {SIGMA_PER_STEP:.5f}")
    print(f"sigma per 2 weeks = {SIGMA_ANNUAL * math.sqrt(weeks_to_years(2.0)):.4f}")
    print(f"sigma per 3 weeks = {SIGMA_ANNUAL * math.sqrt(weeks_to_years(3.0)):.4f}")

    # 2-week ATM call: BS analytic vs MC
    bs_ana = bs_vanilla(S0, S0, weeks_to_years(2.0), SIGMA_ANNUAL, "call")
    paths_2w = simulate_paths(S0, steps_for_weeks(2.0), 50000, seed=1)
    mc = price_vanilla(S0, S0, 2.0, "call", paths_2w)
    print(f"2w ATM call: analytic={bs_ana:.2f}  MC={mc:.2f}  diff={mc-bs_ana:+.2f}")

    bs_ana_p = bs_vanilla(S0, S0, weeks_to_years(2.0), SIGMA_ANNUAL, "put")
    mcp = price_vanilla(S0, S0, 2.0, "put", paths_2w)
    print(f"2w ATM put : analytic={bs_ana_p:.2f}  MC={mcp:.2f}  diff={mcp-bs_ana_p:+.2f}")

    # Chooser at K=S0: should be ~ Call_3w + Put_2w (textbook chooser identity)
    paths_3w = simulate_paths(S0, steps_for_weeks(3.0), 50000, seed=2)
    chooser = price_chooser(S0, S0, 2.0, 3.0, paths=paths_3w)
    call_3w = bs_vanilla(S0, S0, weeks_to_years(3.0), SIGMA_ANNUAL, "call")
    put_2w = bs_vanilla(S0, S0, weeks_to_years(2.0), SIGMA_ANNUAL, "put")
    print(f"Chooser ATM: MC={chooser:.2f}  C3w+P2w identity={call_3w + put_2w:.2f}")

    # Binary put 2w K=S0 payout=1000
    bp = price_binary_put(S0, S0, 2.0, 1000.0)
    print(f"Binary put ATM payout=1000: analytic={bp:.2f}  (expected ~500 since 50/50 ATM)")

    # Knock-out put: K=S0, B = 0.7*S0 over 2 weeks
    ko = price_knockout_put(S0, S0, 0.7 * S0, 2.0, paths=paths_2w)
    plain_put = price_vanilla(S0, S0, 2.0, "put", paths_2w)
    print(f"Knock-out put K=S0 B=0.7S0: KO={ko:.2f}  vs plain put={plain_put:.2f}  "
          f"ratio={ko/plain_put:.3f}")


if __name__ == "__main__":
    _selftest()

#!/usr/bin/env python3
"""
Monte Carlo simulation of PnL across competitor speed priors.

For each allocation (R, S, P), we draw synthetic competitor speed% values from
a prior, compute our rank in the speed ladder, and compute the resulting PnL.
Running many simulations gives distributional statistics (mean, P10, P50, P90)
so we can pick allocations by expected value or by robust tail outcomes.

Usage:
  python monte_carlo_pnl.py
  python monte_carlo_pnl.py --step 1 --num-sims 2000 --total-players 200
"""

from __future__ import annotations

import argparse
import random
from bisect import bisect_right
from typing import Callable, Dict, List, Tuple

from direct_pnl_optimizer import (
    budget_used,
    enumerate_allocations,
    research_value,
    scale_value,
)


Allocation = Tuple[int, int, int]


def _sample_realistic_field(rng: random.Random) -> float:
    """
    Bimodal+tail distribution reflecting likely Prosperity competitor archetypes:
      30%: 0-15%   (big-number chasers, lazy submitters)
      40%: 25-35%  (equal splitters, cautious diversifiers)
      20%: 40-60%  (quant optimizers who realize mid-speed is optimal)
      10%: 65-100% (speed maximizers, contrarians)
    """
    u = rng.random()
    if u < 0.30:
        return rng.uniform(0.0, 15.0)
    if u < 0.70:
        return rng.uniform(25.0, 35.0)
    if u < 0.90:
        return rng.uniform(40.0, 60.0)
    return rng.uniform(65.0, 100.0)


def _sample_research_heavy_field(rng: random.Random) -> float:
    """More of the field is big-number chasers who ignore speed."""
    u = rng.random()
    if u < 0.50:
        return rng.uniform(0.0, 15.0)
    if u < 0.80:
        return rng.uniform(25.0, 35.0)
    if u < 0.93:
        return rng.uniform(40.0, 60.0)
    return rng.uniform(65.0, 100.0)


def _sample_speed_aware_field(rng: random.Random) -> float:
    """Field is more sophisticated about speed; modes shift higher."""
    u = rng.random()
    if u < 0.15:
        return rng.uniform(0.0, 15.0)
    if u < 0.50:
        return rng.uniform(25.0, 40.0)
    if u < 0.85:
        return rng.uniform(45.0, 65.0)
    return rng.uniform(70.0, 100.0)


def prior_samplers() -> Dict[str, Callable[[random.Random], float]]:
    """
    Samplers producing a competitor speed% draw in [0, 100].
    """
    return {
        "heavy_research_crowd": lambda rng: rng.uniform(0.0, 20.0),
        "research_biased": lambda rng: rng.uniform(0.0, 35.0),
        "uniform": lambda rng: rng.uniform(0.0, 100.0),
        "speed_aware": lambda rng: rng.uniform(20.0, 70.0),
        "speed_war": lambda rng: rng.uniform(50.0, 100.0),
        "realistic_field": _sample_realistic_field,
        "research_heavy_field": _sample_research_heavy_field,
        "speed_aware_field": _sample_speed_aware_field,
    }


def simulate_prior(
    allocations: List[Allocation],
    sampler: Callable[[random.Random], float],
    total_players: int,
    num_sims: int,
    rng: random.Random,
) -> Dict[Allocation, List[float]]:
    if total_players < 2:
        raise ValueError("total_players must be >= 2.")

    num_competitors = total_players - 1

    # Pre-sample competitor panels (same panels used across every allocation
    # so results are comparable within a given prior simulation).
    panels: List[List[float]] = []
    for _ in range(num_sims):
        panel = sorted(sampler(rng) for _ in range(num_competitors))
        panels.append(panel)

    denom = float(total_players - 1)
    results: Dict[Allocation, List[float]] = {}

    for alloc in allocations:
        r, s, p = alloc
        res_val = research_value(r)
        sca_val = scale_value(s)
        bud = budget_used(r, s, p)
        pnls: List[float] = []
        for panel in panels:
            idx = bisect_right(panel, p)
            num_above = num_competitors - idx  # competitors with speed > p
            rank = 1 + num_above
            mult = 0.9 - 0.8 * (rank - 1) / denom
            pnls.append(res_val * sca_val * mult - bud)
        results[alloc] = pnls

    return results


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if q <= 0:
        return sorted_values[0]
    if q >= 100:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def describe(pnls: List[float]) -> Dict[str, float]:
    sorted_pnls = sorted(pnls)
    mean_val = sum(pnls) / len(pnls)
    var = sum((x - mean_val) ** 2 for x in pnls) / max(len(pnls) - 1, 1)
    std = var ** 0.5
    return {
        "mean": mean_val,
        "std": std,
        "p5": percentile(sorted_pnls, 5),
        "p10": percentile(sorted_pnls, 10),
        "p25": percentile(sorted_pnls, 25),
        "p50": percentile(sorted_pnls, 50),
        "p75": percentile(sorted_pnls, 75),
        "p90": percentile(sorted_pnls, 90),
        "p95": percentile(sorted_pnls, 95),
    }


def format_allocation(alloc: Allocation) -> str:
    r, s, p = alloc
    return f"R {r:>3}% | S {s:>3}% | Sp {p:>3}%"


def print_ranking(
    title: str,
    ranking: List[Tuple[Allocation, Dict[str, float]]],
    include: Tuple[str, ...] = ("mean", "std", "p10", "p50", "p90"),
) -> None:
    print(title)
    header = f"  {'allocation':<22} |"
    for key in include:
        header += f" {key:>14} |"
    print(header)
    for alloc, stats in ranking:
        row = f"  {format_allocation(alloc):<22} |"
        for key in include:
            row += f" {stats[key]:>14,.0f} |"
        print(row)
    print()


def top_n(
    stats_by_alloc: Dict[Allocation, Dict[str, float]],
    key: str,
    n: int,
    reverse: bool = True,
) -> List[Tuple[Allocation, Dict[str, float]]]:
    items = list(stats_by_alloc.items())
    items.sort(key=lambda kv: kv[1][key], reverse=reverse)
    return items[:n]


def combine_priors(
    per_prior_stats: Dict[str, Dict[Allocation, Dict[str, float]]],
) -> Dict[Allocation, Dict[str, float]]:
    """
    Combine per-prior statistics into robustness metrics:
      - avg_mean: average expected PnL across all priors
      - min_p10: worst prior's 10th-percentile PnL
      - min_mean: worst prior's expected PnL
      - max_mean: best prior's expected PnL
    """
    allocations = next(iter(per_prior_stats.values())).keys()
    combined: Dict[Allocation, Dict[str, float]] = {}
    prior_names = list(per_prior_stats.keys())

    for alloc in allocations:
        means = [per_prior_stats[name][alloc]["mean"] for name in prior_names]
        p10s = [per_prior_stats[name][alloc]["p10"] for name in prior_names]
        combined[alloc] = {
            "avg_mean": sum(means) / len(means),
            "min_mean": min(means),
            "max_mean": max(means),
            "min_p10": min(p10s),
            "max_p10": max(p10s),
        }
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo PnL simulator.")
    parser.add_argument("--step", type=int, default=2, help="Allocation grid step (default: 2).")
    parser.add_argument("--top", type=int, default=10, help="Top N allocations per ranking (default: 10).")
    parser.add_argument(
        "--num-sims",
        type=int,
        default=1500,
        help="Number of Monte Carlo simulations per prior (default: 1500).",
    )
    parser.add_argument(
        "--total-players",
        type=int,
        default=100,
        help="Total number of players including you (default: 100).",
    )
    parser.add_argument(
        "--min-budget",
        type=int,
        default=0,
        help="Minimum percent of budget to spend (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for reproducibility (default: 17).",
    )
    parser.add_argument(
        "--priors",
        nargs="*",
        default=None,
        help="Subset of prior names to use. Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    samplers = prior_samplers()
    if args.priors:
        unknown = [p for p in args.priors if p not in samplers]
        if unknown:
            raise ValueError(f"Unknown priors: {unknown}. Available: {list(samplers)}")
        samplers = {k: samplers[k] for k in args.priors}

    allocations = enumerate_allocations(args.step, args.min_budget)

    print(
        f"Running Monte Carlo: {len(allocations)} allocations x "
        f"{args.num_sims} sims x {len(samplers)} priors "
        f"({args.total_players} total players)."
    )
    print()

    per_prior_stats: Dict[str, Dict[Allocation, Dict[str, float]]] = {}

    for prior_name, sampler in samplers.items():
        pnls = simulate_prior(
            allocations=allocations,
            sampler=sampler,
            total_players=args.total_players,
            num_sims=args.num_sims,
            rng=rng,
        )
        stats = {alloc: describe(vals) for alloc, vals in pnls.items()}
        per_prior_stats[prior_name] = stats

    # Per-prior rankings.
    for prior_name, stats in per_prior_stats.items():
        print(f"===== Prior: {prior_name} =====")
        print_ranking(
            f"  Top by mean PnL:",
            top_n(stats, "mean", args.top),
        )
        print_ranking(
            f"  Top by P10 PnL (robust):",
            top_n(stats, "p10", args.top),
        )

    # Cross-prior rankings.
    combined = combine_priors(per_prior_stats)
    print("===== Across all priors =====")
    print_ranking(
        "  Top by average expected PnL (equal-weight mixture):",
        top_n(combined, "avg_mean", args.top),
        include=("avg_mean", "min_mean", "max_mean", "min_p10"),
    )
    print_ranking(
        "  Top by worst-case P10 (most robust tail):",
        top_n(combined, "min_p10", args.top),
        include=("min_p10", "max_p10", "avg_mean", "min_mean"),
    )
    print_ranking(
        "  Top by worst-prior expected PnL (conservative):",
        top_n(combined, "min_mean", args.top),
        include=("min_mean", "avg_mean", "max_mean", "min_p10"),
    )

    best_ev = top_n(combined, "avg_mean", 1)[0]
    best_robust = top_n(combined, "min_p10", 1)[0]
    best_conservative = top_n(combined, "min_mean", 1)[0]

    print("Summary recommendations:")
    print(
        f"  Expected-value pick:    {format_allocation(best_ev[0])}  "
        f"avg_mean={best_ev[1]['avg_mean']:,.0f}  min_p10={best_ev[1]['min_p10']:,.0f}"
    )
    print(
        f"  Robust tail pick:       {format_allocation(best_robust[0])}  "
        f"min_p10={best_robust[1]['min_p10']:,.0f}  avg_mean={best_robust[1]['avg_mean']:,.0f}"
    )
    print(
        f"  Conservative pick:      {format_allocation(best_conservative[0])}  "
        f"min_mean={best_conservative[1]['min_mean']:,.0f}  avg_mean={best_conservative[1]['avg_mean']:,.0f}"
    )


if __name__ == "__main__":
    main()

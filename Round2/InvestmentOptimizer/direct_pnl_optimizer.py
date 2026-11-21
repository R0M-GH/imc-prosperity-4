#!/usr/bin/env python3
"""
Direct PnL optimizer for the Invest & Expand challenge.

Formula (from the challenge rules):
  Research(r) = 200_000 * ln(1 + r) / ln(101)
  Scale(s)    = 7 * s / 100
  SpeedMult   = 0.1 + 0.8 * F_competitors(speed_pct)
  BudgetUsed  = 50_000 * (r + s + p) / 100
  PnL         = Research * Scale * SpeedMult - BudgetUsed

We cannot observe competitors' speed allocations, so we model several
priors (CDFs) over their speed% and score every allocation under each.

Usage:
  python direct_pnl_optimizer.py
  python direct_pnl_optimizer.py --step 1 --top 10
  python direct_pnl_optimizer.py --min-budget 70
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


BUDGET_TOTAL = 50_000.0


@dataclass
class SpeedPrior:
    name: str
    description: str
    cdf: Callable[[float], float]  # F(x) in [0, 1] for speed_pct x in [0, 100]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def research_value(r: float) -> float:
    return 200_000.0 * math.log(1.0 + r) / math.log(1.0 + 100.0)


def scale_value(s: float) -> float:
    return 7.0 * s / 100.0


def speed_multiplier(speed_pct: float, cdf: Callable[[float], float]) -> float:
    return 0.1 + 0.8 * clamp(cdf(speed_pct), 0.0, 1.0)


def budget_used(r: float, s: float, p: float) -> float:
    return BUDGET_TOTAL * (r + s + p) / 100.0


def pnl(r: float, s: float, p: float, cdf: Callable[[float], float]) -> float:
    return (
        research_value(r)
        * scale_value(s)
        * speed_multiplier(p, cdf)
        - budget_used(r, s, p)
    )


def default_priors() -> List[SpeedPrior]:
    """
    Each prior models P(competitor_speed_pct <= x). Pick them to span
    a realistic range of what the field might do.
    """
    return [
        SpeedPrior(
            name="heavy_research_crowd",
            description="Most teams go low-speed (<=20%).",
            cdf=lambda x: clamp(x / 20.0, 0.0, 1.0),
        ),
        SpeedPrior(
            name="research_biased",
            description="Typical clustering in 0-35%.",
            cdf=lambda x: clamp(x / 35.0, 0.0, 1.0),
        ),
        SpeedPrior(
            name="uniform",
            description="Competitors spread evenly 0-100%.",
            cdf=lambda x: clamp(x / 100.0, 0.0, 1.0),
        ),
        SpeedPrior(
            name="speed_aware",
            description="Field pushes speed higher, clustering 20-70%.",
            cdf=lambda x: clamp((x - 20.0) / 50.0, 0.0, 1.0),
        ),
        SpeedPrior(
            name="speed_war",
            description="Aggressive speed bidding, clustering 50-100%.",
            cdf=lambda x: clamp((x - 50.0) / 50.0, 0.0, 1.0),
        ),
    ]


def enumerate_allocations(step: int, min_budget_pct: int) -> List[Tuple[int, int, int]]:
    if step <= 0 or step > 100:
        raise ValueError("step must be in [1, 100].")
    out: List[Tuple[int, int, int]] = []
    for r in range(0, 101, step):
        for s in range(0, 101 - r, step):
            p = 100 - r - s
            if p % step != 0:
                continue
            if (r + s + p) < min_budget_pct:
                continue
            out.append((r, s, p))
    return out


def score_all(
    allocations: List[Tuple[int, int, int]],
    priors: List[SpeedPrior],
) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    scores: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    for alloc in allocations:
        r, s, p = alloc
        per_prior = {
            prior.name: pnl(float(r), float(s), float(p), prior.cdf)
            for prior in priors
        }
        per_prior["_min"] = min(per_prior.values())
        per_prior["_mean"] = sum(per_prior.values()) / len(priors)
        per_prior["_max"] = max(
            v for k, v in per_prior.items() if k not in ("_min", "_mean", "_max")
        )
        scores[alloc] = per_prior
    return scores


def top_n(
    scores: Dict[Tuple[int, int, int], Dict[str, float]],
    key: str,
    n: int,
) -> List[Tuple[Tuple[int, int, int], Dict[str, float]]]:
    items = list(scores.items())
    items.sort(key=lambda kv: kv[1][key], reverse=True)
    return items[:n]


def format_allocation(alloc: Tuple[int, int, int]) -> str:
    r, s, p = alloc
    return f"R {r:>3}% | S {s:>3}% | Sp {p:>3}%"


def print_block(
    title: str,
    ranking: List[Tuple[Tuple[int, int, int], Dict[str, float]]],
    priors: List[SpeedPrior],
) -> None:
    print(title)
    header = f"  {'allocation':<22} | {'min':>12} | {'mean':>12} | {'max':>12} |"
    for prior in priors:
        header += f" {prior.name:>22} |"
    print(header)

    for alloc, vals in ranking:
        row = (
            f"  {format_allocation(alloc):<22} | "
            f"{vals['_min']:>12,.0f} | "
            f"{vals['_mean']:>12,.0f} | "
            f"{vals['_max']:>12,.0f} |"
        )
        for prior in priors:
            row += f" {vals[prior.name]:>22,.0f} |"
        print(row)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct PnL optimizer over the known challenge formula."
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Grid size in percent for allocation search (default: 1).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top allocations to display per ranking (default: 10).",
    )
    parser.add_argument(
        "--min-budget",
        type=int,
        default=0,
        help="Minimum percent of budget to spend (default: 0 = allow partial spend).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    priors = default_priors()

    print("Speed priors in use:")
    for prior in priors:
        print(f"  - {prior.name}: {prior.description}")
    print()

    allocations = enumerate_allocations(args.step, args.min_budget)
    scores = score_all(allocations, priors)

    print_block(
        "ROBUST (worst-case across priors) - optimize minimum PnL:",
        top_n(scores, "_min", args.top),
        priors,
    )
    print_block(
        "AVERAGE (mean across priors) - balanced outlook:",
        top_n(scores, "_mean", args.top),
        priors,
    )
    print_block(
        "BEST-CASE (favorable prior) - optimize upside:",
        top_n(scores, "_max", args.top),
        priors,
    )

    for prior in priors:
        print_block(
            f"Best under prior '{prior.name}':",
            top_n(scores, prior.name, 5),
            priors,
        )

    robust_best_alloc, robust_best_vals = top_n(scores, "_min", 1)[0]
    mean_best_alloc, mean_best_vals = top_n(scores, "_mean", 1)[0]

    print("Summary recommendations:")
    print(
        f"  Robust (worst-case maximizer): {format_allocation(robust_best_alloc)}  "
        f"min={robust_best_vals['_min']:,.0f}  mean={robust_best_vals['_mean']:,.0f}"
    )
    print(
        f"  Expected-value maximizer:      {format_allocation(mean_best_alloc)}  "
        f"min={mean_best_vals['_min']:,.0f}  mean={mean_best_vals['_mean']:,.0f}"
    )


if __name__ == "__main__":
    main()

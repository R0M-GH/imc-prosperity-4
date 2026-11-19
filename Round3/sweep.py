"""
Parameter sweep that measures PnL under both scoring regimes:
  - "preview" (first 100k timestamps of day-2 data, matches submission 473067)
  - "full"    (all 1M timestamps of all 3 historical days)

Each variant is applied to round3.py at module level and a fresh backtest is run.
"""
from __future__ import annotations

import importlib
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

import backtest
import round3


def _restrict_csv_to_first_100k(src: str, dst: str) -> None:
    df = pd.read_csv(src, sep=";")
    df = df[df["timestamp"] <= 99900].copy()
    df.to_csv(dst, sep=";", index=False)


PREVIEW_CSV = "/tmp/_preview_day2.csv"
_restrict_csv_to_first_100k(
    os.path.join(os.path.dirname(__file__), "ROUND_3", "prices_round_3_day_2.csv"),
    PREVIEW_CSV,
)


def run_preview() -> float:
    res = backtest.run_backtest(PREVIEW_CSV, start_tte_days=5.0, verbose=False)
    return res["final_pnl"]


def run_full() -> float:
    total = 0.0
    for d, tte in [(0, 8.0), (1, 7.0), (2, 6.0)]:
        res = backtest.run_backtest(
            os.path.join(os.path.dirname(__file__), "ROUND_3", f"prices_round_3_day_{d}.csv"),
            start_tte_days=tte, verbose=False,
        )
        total += res["final_pnl"]
    return total


def apply_overrides(overrides: dict) -> None:
    """Mutate round3 module-level config in place."""
    for key, val in overrides.items():
        # nested dicts get merged into the existing nested config
        if isinstance(val, dict) and key in vars(round3) and isinstance(getattr(round3, key), dict):
            target = getattr(round3, key)
            for k2, v2 in val.items():
                if isinstance(v2, dict) and k2 in target and isinstance(target[k2], dict):
                    target[k2].update(v2)
                else:
                    target[k2] = v2
        else:
            setattr(round3, key, val)


def reset_module() -> None:
    importlib.reload(round3)
    importlib.reload(backtest)


def evaluate(name: str, overrides: dict) -> tuple[float, float]:
    reset_module()
    apply_overrides(overrides)
    pv = run_preview()
    fl = run_full()
    return pv, fl


VARIANTS: list[tuple[str, dict]] = [
    ("baseline (current)", {}),
    ("HYDROGEL larger qty", {
        "PARAMS": {"HYDROGEL_PACK": {"max_quote_size": 80, "soft_cap": 180}},
    }),
    ("HYDROGEL alpha=0.30", {
        "PARAMS": {"HYDROGEL_PACK": {"ema_alpha": 0.30}},
    }),
    ("HYDROGEL tighter clear", {
        "PARAMS": {"HYDROGEL_PACK": {"clear_edge": 0.5}},
    }),
    ("HYDROGEL te=2.5", {
        "PARAMS": {"HYDROGEL_PACK": {"take_edge": 2.5}},
    }),
    ("VEV_4000/4500 aggressive MM", {
        "VEV_PARAMS": {
            "VEV_4000": {"max_quote": 50, "soft_cap": 200, "hard_cap": 280},
            "VEV_4500": {"max_quote": 50, "soft_cap": 200, "hard_cap": 280},
        },
    }),
    ("VEV_4000/4500 take_edge=1", {
        "VEV_PARAMS": {
            "VEV_4000": {"take_edge": 1.0, "clear_edge": 0.5, "max_quote": 40},
            "VEV_4500": {"take_edge": 1.0, "clear_edge": 0.5, "max_quote": 40},
        },
    }),
    ("Enable EXTRACT (R1 PEPPER style)", {
        "PARAMS": {"VELVETFRUIT_EXTRACT": {"enable": True}},
    }),
    ("All VEV hard_cap to 250", {
        "VEV_PARAMS": {f"VEV_{K}": {"hard_cap": 250} for K in (5000,5100,5200,5300,5400,5500)},
    }),
    ("VEV smile take_edge 1.5", {
        "VEV_PARAMS": {f"VEV_{K}": {"take_edge": 1.5, "clear_edge": 0.75} for K in (5000,5100,5200,5300,5400,5500)},
    }),
    ("Combo: HYDROGEL 80 + VEV4000/4500 aggro", {
        "PARAMS": {"HYDROGEL_PACK": {"max_quote_size": 80, "soft_cap": 180}},
        "VEV_PARAMS": {
            "VEV_4000": {"max_quote": 50, "soft_cap": 200, "hard_cap": 280},
            "VEV_4500": {"max_quote": 50, "soft_cap": 200, "hard_cap": 280},
        },
    }),
]


def main() -> int:
    print(f"{'variant':<40s} {'preview':>10s} {'full(3d)':>12s}")
    print("-" * 65)
    base_pv, base_fl = evaluate("baseline", {})
    for name, ov in VARIANTS:
        pv, fl = evaluate(name, ov)
        print(f"{name:<40s} {pv:>+10,.0f} {fl:>+12,.0f}  "
              f"(Δ pv={pv-base_pv:+,.0f}  Δ full={fl-base_fl:+,.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

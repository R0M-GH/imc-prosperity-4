#!/usr/bin/env python3
"""
One-shot allocation optimizer for Research / Scale / Speed.

Usage:
  python optimizer.py --data allocations.csv
  python optimizer.py --data allocations.csv --step 1 --top 10 --ridge 1e-6
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple


REQUIRED_PCT_COLUMNS = ("research_pct", "scale_pct", "speed_pct")
FORECAST_COLUMNS = ("research_forecast", "scale_forecast", "speed_forecast")


@dataclass
class Observation:
    research: float
    scale: float
    speed: float
    objective: float


def parse_float(value: str) -> float:
    value = value.strip()
    if value == "":
        raise ValueError("empty numeric value")
    # Accept common UI/export formatting such as "200,000" or " 50_000 ".
    cleaned = value.replace(",", "").replace("_", "")
    return float(cleaned)


def read_observations(path: str, objective_column: str, budget_total: float) -> List[Observation]:
    rows: List[Observation] = []

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        fieldnames = set(name.strip() for name in reader.fieldnames if name is not None)
        missing_pct = [c for c in REQUIRED_PCT_COLUMNS if c not in fieldnames]
        if missing_pct:
            raise ValueError(f"Missing required columns: {missing_pct}")

        has_objective = objective_column in fieldnames
        available_forecast_columns = [col for col in FORECAST_COLUMNS if col in fieldnames]
        # At minimum we expect the two UI-visible forecasts.
        minimum_default_forecasts = ("research_forecast", "scale_forecast")
        has_min_default_forecasts = all(col in fieldnames for col in minimum_default_forecasts)

        if not has_objective and not has_min_default_forecasts:
            raise ValueError(
                f"Need either '{objective_column}' column or at least "
                f"{minimum_default_forecasts} to define a default objective."
            )

        for idx, row in enumerate(reader, start=2):
            if row is None:
                continue
            try:
                r = parse_float(row["research_pct"])
                s = parse_float(row["scale_pct"])
                p = parse_float(row["speed_pct"])
            except Exception as exc:
                raise ValueError(f"Row {idx}: invalid percentage columns ({exc}).") from exc

            if abs((r + s + p) - 100.0) > 1e-6:
                raise ValueError(f"Row {idx}: percentages must sum to 100, got {r + s + p}.")

            if has_objective and row.get(objective_column, "").strip() != "":
                obj = parse_float(row[objective_column])
            else:
                # Preferred fallback when available:
                # total_score = research_forecast * scale_forecast * speed_multiplier - budget_used
                # where budget_used = budget_total * (r + s + p) / 100.
                has_formula_fields = all(
                    col in fieldnames for col in ("research_forecast", "scale_forecast", "speed_multiplier")
                )
                if has_formula_fields and all(
                    row.get(col, "").strip() != ""
                    for col in ("research_forecast", "scale_forecast", "speed_multiplier")
                ):
                    budget_used = budget_total * (r + s + p) / 100.0
                    obj = (
                        parse_float(row["research_forecast"])
                        * parse_float(row["scale_forecast"])
                        * parse_float(row["speed_multiplier"])
                        - budget_used
                    )
                else:
                    # Legacy fallback: sum forecasts that exist in this CSV.
                    missing_forecast = [
                        c for c in available_forecast_columns if row.get(c, "").strip() == ""
                    ]
                    if missing_forecast:
                        raise ValueError(
                            f"Row {idx}: no {objective_column} and missing forecast values: "
                            f"{missing_forecast}"
                        )
                    obj = sum(parse_float(row[c]) for c in available_forecast_columns)

            rows.append(Observation(research=r, scale=s, speed=p, objective=obj))

    if len(rows) < 10:
        raise ValueError(
            f"Need at least 10 rows to fit quadratic model reliably, got {len(rows)}."
        )
    return rows


def feature_vector(r: float, s: float, p: float) -> List[float]:
    return [1.0, r, s, p, r * r, s * s, p * p, r * s, r * p, s * p]


def transpose(matrix: List[List[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*matrix)]


def matmul(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    out = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for k in range(len(b)):
            aik = a[i][k]
            if aik == 0.0:
                continue
            for j in range(len(b[0])):
                out[i][j] += aik * b[k][j]
    return out


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """
    Solve A x = b using Gauss-Jordan elimination with partial pivoting.
    """
    n = len(a)
    aug = [row[:] + [b_i] for row, b_i in zip(a, b)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        pivot_val = aug[pivot_row][col]
        if abs(pivot_val) < 1e-12:
            raise ValueError("Singular matrix while fitting model. Add more diverse samples.")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot_val = aug[col][col]
        inv_pivot = 1.0 / pivot_val
        for j in range(col, n + 1):
            aug[col][j] *= inv_pivot

        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def fit_quadratic_model(observations: List[Observation], ridge: float) -> List[float]:
    x = [feature_vector(obs.research, obs.scale, obs.speed) for obs in observations]
    y = [obs.objective for obs in observations]

    xt = transpose(x)
    xtx = matmul(xt, x)
    xty = matmul(xt, [[yi] for yi in y])

    for i in range(len(xtx)):
        xtx[i][i] += ridge

    beta = solve_linear_system(xtx, [row[0] for row in xty])
    return beta


def predict(beta: List[float], r: float, s: float, p: float) -> float:
    feats = feature_vector(r, s, p)
    return sum(w * x for w, x in zip(beta, feats))


def r2_score(observations: List[Observation], beta: List[float]) -> float:
    ys = [obs.objective for obs in observations]
    y_mean = sum(ys) / len(ys)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum(
        (obs.objective - predict(beta, obs.research, obs.scale, obs.speed)) ** 2
        for obs in observations
    )
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def search_allocations(beta: List[float], step: int, top_k: int) -> List[Tuple[int, int, int, float]]:
    candidates: List[Tuple[int, int, int, float]] = []
    for r in range(0, 101, step):
        for s in range(0, 101 - r, step):
            p = 100 - r - s
            if p % step != 0:
                continue
            score = predict(beta, float(r), float(s), float(p))
            candidates.append((r, s, p, score))

    candidates.sort(key=lambda t: t[3], reverse=True)
    return candidates[:top_k]


def choose_recommended(top_candidates: List[Tuple[int, int, int, float]]) -> Tuple[int, int, int, float]:
    """
    Pick a recommendation that is robust among top candidates by preferring
    less extreme allocations when scores are close.
    """
    if not top_candidates:
        raise ValueError("No candidates found.")

    best_score = top_candidates[0][3]
    near_best = [c for c in top_candidates if c[3] >= best_score * 0.995]
    near_best.sort(key=lambda c: abs(c[0] - 33) + abs(c[1] - 33) + abs(c[2] - 34))
    return near_best[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize one-shot allocation percentages.")
    parser.add_argument("--data", required=True, help="Path to CSV with tested allocations.")
    parser.add_argument(
        "--objective-column",
        default="total_score",
        help="CSV column to optimize. Defaults to total_score.",
    )
    parser.add_argument(
        "--budget-total",
        type=float,
        default=50_000.0,
        help="Total XIREC budget used in PnL formula (default: 50000).",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-6,
        help="Ridge regularization value for stable fitting (default: 1e-6).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Allocation search granularity in percent (default: 1).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top candidate allocations to print (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.step <= 0 or args.step > 100:
        raise ValueError("--step must be in [1, 100].")

    observations = read_observations(args.data, args.objective_column, budget_total=args.budget_total)
    beta = fit_quadratic_model(observations, ridge=args.ridge)
    train_r2 = r2_score(observations, beta)
    top_candidates = search_allocations(beta, step=args.step, top_k=args.top)
    recommendation = choose_recommended(top_candidates)

    print("Model fit complete")
    print(f"Rows used: {len(observations)}")
    print(f"Training R^2: {train_r2:.4f}")
    print()

    print("Top candidate allocations (Research / Scale / Speed -> predicted objective):")
    for r, s, p, score in top_candidates:
        print(f"  {r:>3}% / {s:>3}% / {p:>3}% -> {score:.6f}")

    rr, rs, rp, rscore = recommendation
    print()
    print("Recommended (robust near-best):")
    print(f"  Research {rr}% | Scale {rs}% | Speed {rp}%")
    print(f"  Predicted objective: {rscore:.6f}")


if __name__ == "__main__":
    main()

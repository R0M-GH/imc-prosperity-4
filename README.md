# IMC Prosperity 4 (2026)

My algorithmic and manual trading code from rounds 2 to 5 of
[IMC Prosperity 4](https://prosperity.imc.com), a team trading competition
with about 20,000 teams.

**Final result:** 798th of ~20K teams overall (211th in the USA), 278th on
the manual challenges.

Round 1 was written with my team and lives in our team repository; this
repo holds the rounds I built.

## Layout

| Folder | What's in it |
| --- | --- |
| `Round2/` | `round2.py`: ASH_COATED_OSMIUM market making around a microprice EMA, and INTARIAN_PEPPER_ROOT trading off fast/slow EMA momentum with a structural long bias (+78,336 in round 2). `InvestmentOptimizer/` scores every Research / Scale / Speed split for the manual challenge, with Monte Carlo over guesses at how other teams split Speed. |
| `Round3/` | `round3.py`: HYDROGEL_PACK market making plus voucher trading off a volatility smile fit live across the strikes each tick. `backtest.py` replays historical order books, `sweep.py` sweeps parameters, `POSTMORTEM.md` covers what rounds 1 and 2 taught us and how it shaped this round. `ManualChallenge/` is the bid optimizer and the simulations behind the manual submission. |
| `Round4/` | `round4.py` carries the round 3 strategy forward with opt-in counterparty signals and gamma scalping (both off after backtests showed they lost money to spread costs). `counterparty_analysis.py` measures each counterparty's predictive alpha. `ManualChallenge/manual_pricer.py` is a Monte Carlo pricer for vanilla, chooser, binary, and knock-out options on IMC's discrete-step GBM. See `Round4/README.md`. |
| `Round5/` | `round5.py`: relative-value models across the round 5 product categories. |

`datamodel.py` is IMC's sandbox data model, included so the traders and
backtesters run locally.

## Running a backtest

IMC's historical price and trade CSVs aren't included. Download them from
the competition and put them in `Round3/ROUND_3/` or `Round4/ROUND_4/`
(or pass `--data-dir`):

```bash
pip install -r requirements.txt
python Round3/backtest.py
python Round4/backtest_r4.py
```

Python 3.10+ with `pip install -r requirements.txt` (jsonpickle, which IMC's
sandbox provides, plus numpy, pandas, and scipy for the backtesters and pricers).

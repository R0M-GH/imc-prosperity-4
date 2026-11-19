# Round 4 — "Hello, I'm Mark" / "Vanilla Just Isn't Exotic Enough"

## File map

| File | Purpose |
|---|---|
| `round4.py` | Main trader. Carries forward the R3 winning logic (HYDROGEL MM, smile-based voucher trading) plus opt-in counterparty signals and gamma scalp infra (both default OFF after backtests showed they regress). |
| `datamodel.py` | IMC sandbox datamodel (copied from R3). |
| `backtest_r4.py` | Round-4 backtester. Critically, this loads `trades_round_4_day_*.csv` and feeds Mark counterparty IDs into `state.market_trades` so signal logic can fire. |
| `counterparty_analysis.py` | Per-Mark predictive-alpha analyser. Outputs `mark_alpha.json` ranking each Mark's directional info. |
| `mark_alpha.json` | Generated; consumed by `round4.py` at runtime. |
| `ROUND_4/` | Historical days 1/2/3 prices + trades CSVs (with counterparty IDs). |
| `ManualChallenge/manual_pricer.py` | Monte Carlo pricer for AETHER_CRYSTAL exotics matching IMC's exact discrete-step model. Handles vanilla calls/puts, choosers, binary puts, knock-out puts. |
| `ManualChallenge/recommend_bids.py` | Fill in market prices + spot, run, get fair values, edges, and a greedy bid recommendation. |

## Algo backtest (3 historical days, full 1M ticks each)

| Config | Day 1 | Day 2 | Day 3 | Total |
|---|---:|---:|---:|---:|
| **R3 carryover (default)** | +$37,755 | +$50,268 | +$47,978 | **+$136,001** |
| Mark signals ON, gamma OFF | +$32,778 | +$45,277 | +$39,624 | +$117,680 |
| Mark signals OFF, gamma ON | +$34,268 | +$45,534 | +$48,382 | +$128,184 |
| Both ON | +$28,836 | +$43,769 | +$33,258 | +$105,863 |

The Mark signals (Mark 67 BUY EXTRACT +$2.24 alpha, Mark 22/49 SELL EXTRACT -$1.5/-$2.1 alpha) are *real* but execution cost (~half-spread $2.50) eats them. Likewise the gamma scalp's expected $5/option/day from realized-vs-implied vol is consumed by hedging spread costs.

We default both to OFF and ship the R3 strategy on R4 data: ~+$45k/day vs ~$58k actual on R3. Production typically beats backtest by 7–19%, so expect **~+$50-55k/day in production**.

## Counterparty intel (informational)

Three actionable Mark signals on EXTRACT:
- **Mark 67 BUY** → +$2.24 in next 1k ticks (smart money long; n=165)
- **Mark 22 SELL** → -$1.56 in next 1k ticks (n=101)
- **Mark 49 SELL** → -$2.14 in next 1k ticks (n=105)

Voucher counterparty patterns are *structural* (Mark 22 always sells OTM, Mark 01 always buys) but offer no short-term alpha because deltas are tiny.

## Manual challenge

Underlying GBM with σ = 251% annualized, 4 steps/day, 1008 steps/year. Products:
- 2-week + 3-week vanilla calls/puts
- Chooser (3 weeks; choose call/put at week 2)
- Binary put
- Knock-out put

To submit: edit the `PRODUCTS = [...]` list in `recommend_bids.py` with whatever IMC's UI displays, then run:

```bash
python ManualChallenge/recommend_bids.py
```

The script will print fair values, edges (mispricing), and a greedy bid recommendation maximising sum of `qty × edge × CONTRACT_SIZE`.

### Things to watch for in the manual challenge
1. **Knock-out puts often mis-priced upward** — if IMC quotes anything close to vanilla put price, sell aggressively (knock-out is asymmetric: ~91% chance of going to zero given barrier ~30% below spot).
2. **Binary puts** — often mis-quoted because traders eyeball them; fair value ≈ payout × P(S_T<K).
3. **Chooser put-call identity** — chooser ≈ Call(T_total) + Put(T_choose). Use this to spot mis-pricing.
4. **Greedy optimizer doesn't account for correlation risk** — if you size into many long-vol positions, your variance multiplies. Cap individual positions or reduce total budget if risk-averse.

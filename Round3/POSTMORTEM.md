# Round 1/2 Submission Postmortem & Round 3 Takeaways

## What we learned from the prior submissions

### Submission `271738.zip` (Round 1, profit = +$89,763)
- Code: `Round1/round1.py` (the simple version with PEPPER + OSMIUM).
- Per-product: ASH_COATED_OSMIUM = +$16,511 (ended +80, max long); INTARIAN_PEPPER_ROOT = +$73,252 (ended +80, max long).
- 1175 total trades in the order book; **156 of them were ours**.
- Steady PnL trajectory from tick 0: +$4.2k by ts=50k, +$8.8k by ts=100k, +$41.4k by ts=500k.
- Max drawdown: only $1,169.
- Result per trade: ~$469. Wide spread + drift bias on PEPPER did most of the work.

### Submission `359686.zip` (Round 2, profit = +$78,336)
- Code: `Round2/round2.py` (the more sophisticated version with fast/slow EMA momentum).
- Per-product: OSMIUM = +$17,211; PEPPER = +$61,124.
- Fired **9,907 of our trades** (8x more than R1!), max drawdown $4,918.
- A `Function not found` Lambda error popped at ts=359686 (transient — trades continued, no impact).
- **PEPPER spent the first 100k ts at -$5,622 PnL** before recovering. Cause: `slow_alpha=0.02` EMA needed thousands of ticks to converge; until then the momentum signal was noisy and we paid spread on bad trades.
- Net: more sophisticated logic delivered **less PnL** than R1 because:
  1. The slow EMA bled before warmup.
  2. The smaller `take_edge`/`clear_edge` rebalanced too often, paying the spread.
  3. The extra activity locked in noise rather than capturing trend.

### Submission `473067.zip` (Round 3, profit = +$3,463)
- Code: an earlier version of `Round3/round3.py` (historical-fixed smile, hardcoded TTE=5).
- Per-product: HYDROGEL_PACK = +$4,228; vouchers net negative (~ -$760).
- **Run length was only 100k ts** (vs. 1M ts for R1/R2). Either the R3 scoring window is shorter, or this was a preview/test run; either way, our strategy must work in the first 100k ticks.
- Bug: the historical smile gave the wrong direction for VEV_5400/5500 because the simulator played day-2 data (TTE=6) but our trader hardcoded TTE=5. We sold what we should have bought.

## Cross-cutting takeaways

| # | Takeaway | Action taken in `round3.py` |
|---|---|---|
| 1 | Strategies must work from **tick 0** — no slow-EMA warmup. | HYDROGEL uses single fast EMA (α=0.20), vouchers use live smile fit (no state). |
| 2 | Over-trading erodes PnL. Each redundant rebalance costs the spread. | Wide-ish `take_edge`/`clear_edge` for the smile-bearing strikes; deep-ITM where parity is exact uses tight 0.5/0.25. |
| 3 | Aggressive position-holding (R1: end at limit) > active rebalancing (R2). | Conviction-scaled hard caps; let strong signals build to soft cap before throttling. |
| 4 | Wide-spread products are gold mines. VEV_4000 (spread=21) and VEV_4500 (spread=16) were under-exploited at `take_edge=2`. | Tightened deep-ITM to `take_edge=0.5, clear_edge=0.25, max_quote=40, hard_cap=250`. ~+$9k full-day. |
| 5 | TTE/regime mistakes are toxic for any fixed-smile strategy. | Live smile fit each tick: market IV and the fitted curve share T, so deviations are invariant. |
| 6 | Slow random walks (PEPPER R2 first-100k, VELVETFRUIT_EXTRACT) bleed on directional MM. | EXTRACT disabled. Confirmed: enabling adds -$28k to backtest. |
| 7 | The PnL leader is whichever single product has the widest spread + best fair-value model. | HYDROGEL (spread=16) is the workhorse, ~$4-5k per 100k ts. |

## Final backtest (after applying takeaways)

| Mode      | TTE  | Preview (100k of day 2) | Full (3 historical days) |
|-----------|-----:|------------------------:|-------------------------:|
| smile     | hist |                  +4,170 |                +145,154 |
| smile     | r3=5 |                  +4,147 |                +144,985 |
| smile     | r3=6 |                  +4,170 |                +144,921 |
| per_strike| hist |                  +4,356 |                +149,422 |
| per_strike| r3=5 |                  +3,464 |                +125,022 |
| per_strike| r3=6 |                  +4,356 |                +140,920 |

Smile mode (default) is **TTE-invariant** — preview varies by ≤ $25 regardless of T assumption. Per-strike mode is fragile: $900 swing on preview when TTE is wrong.

## Open risks

1. We don't know if the R3 production scoring is 100k ts or full 1M ts; the strategy aims to be near-optimal on either.
2. The smile fit relies on at least 3 of {VEV_5000…VEV_5500} having valid live IVs each tick. If a strike has missing data, we fall back to historical constants.
3. EXTRACT is a sleeping product; if competitors find an alpha there we leave that money on the table.

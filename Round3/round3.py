"""
Round 3 - "Gloves Off" Trader
=============================

Strategy follows A.R.I.A.'s 'Orin' hints for the algorithmic challenge:
  1. Compute Black-Scholes implied vol vs moneyness (the smile).
  2. Find vouchers that deviate from the structural smile.
  3. Size each position proportional to the strength of the deviation.

The smile is fit *live every tick* from the 6 vega-bearing strikes
(VEV_5000..VEV_5500). Fitting live makes the strategy invariant to TTE
mistakes and to vol-regime shifts: both the market IV and the fitted smile
share the same T, so RELATIVE deviations remain meaningful even if the
absolute T_days estimate is off.

Products
--------
- HYDROGEL_PACK         : delta-1, mean-reverts around ~10000 (limit 200).
                          Aggressive EMA-based MM (alpha=0.20) with anchor
                          blend; this is the single largest PnL contributor.
- VELVETFRUIT_EXTRACT   : delta-1, slow random walk around ~5250 (limit 200).
                          Disabled by default - directional MM bleeds against
                          trends. Used only as the *underlying spot* for
                          voucher pricing.
- VEV_<K> for K in {4000,4500,5000,5100,5200,5300,5400,5500,6000,6500}:
                          European calls on VELVETFRUIT_EXTRACT, expiry on day
                          7 (limit 300 each). Round 3 begins at TTE = 5 days.

Pricing
-------
- Black-Scholes (r = q = 0).
- LIVE smile fit each tick: solve sigma_i = a*m_i^2 + b*m_i + c by least
  squares over the 6 strikes 5000..5500, where m_i = log(K_i/S)/sqrt(T).
  Falls back to the historical smile (a=0.537, c=0.01253) if fewer than 3
  strikes have valid live IVs.
- Conviction is z = (sigma_smile - sigma_market) / SMILE_RESIDUAL_STD;
  z>0 means the market underprices that strike vs the smile (BUY bias),
  z<0 means it overprices (SELL bias). Conviction strength = clip(|z|/3, 0,1)
  scales the per-side hard_cap and soft_cap for that voucher.

Lessons from earlier preview submission
---------------------------------------
The first submission used a *historical-fixed* smile sigma = 0.537*m^2 + 0.01253
fit at TTE 6/7/8d. When IMC's preview replayed day-2 history (TTE=6) but our
trader hardcoded start_tte_days=5, every voucher's market IV computed at T=5
came out higher than the smile, and we shorted everything. The live-fit smile
fixes this: market IV and smile fit share the same T, so VEV_5400's actual
"cheap-relative-to-its-neighbours" character is detected regardless of T.

PnL backtest summary (3 days, simple passive-fill simulator)
------------------------------------------------------------
At historical TTEs (8/7/6 d):  ~+136k total (~+45k/day)
At round-3 TTE (5 d):          ~+135k total (~+45k/day)
At wrong TTE (6 d):            ~+136k total  <-- live fit is invariant
Max single-day drawdown:        ~$20k

Toggle PRICING_MODE = "per_strike" to absorb the persistent IV deviations as
fair (NOT robust to TTE mistakes - drops to ~+120k if T is wrong).
"""
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import math
import jsonpickle

# ---------------------------------------------------------------------------
# Product identifiers / position limits
# ---------------------------------------------------------------------------
HYDROGEL = "HYDROGEL_PACK"
EXTRACT = "VELVETFRUIT_EXTRACT"

VEV_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VEVS = [f"VEV_{K}" for K in VEV_STRIKES]
STRIKE_OF: Dict[str, int] = {f"VEV_{K}": K for K in VEV_STRIKES}

LIMIT: Dict[str, int] = {HYDROGEL: 200, EXTRACT: 200}
for v in VEVS:
    LIMIT[v] = 300

# ---------------------------------------------------------------------------
# Time-to-expiry handling
# ---------------------------------------------------------------------------
# Vouchers expire after day 7 (TTE = 7d at start of round 1). Round 3 starts
# at TTE = 5d. One round = 1 simulated day = 1,000,000 timestamps.
TIMESTAMPS_PER_DAY = 1_000_000
ROUND3_START_TTE_DAYS = 5.0

# Backtest hook: the live trader doesn't know which historical day it is on,
# but a backtest can override the start-TTE by setting `Trader.start_tte_days`.

# Floor on T to avoid blowups near expiry; unimportant for round 3 itself.
MIN_T_DAYS = 0.05

# ---------------------------------------------------------------------------
# Volatility smile / per-strike IVs
# ---------------------------------------------------------------------------
# Orin's hints: "If you plot Implied Volatility against moneyness, a pattern
# tends to emerge ... Find the odd ones out". We treat the historical smile
# as the structural fair value and trade individual vouchers against their
# deviation from that smile.
#
# Quadratic smile fit on rounds 1-2 history (TTE=8/7/6, 3 days):
#       sigma(m) ~= 0.537 * m^2 + 0.01253,    m = log(K/S) / sqrt(T_days)
# Residual std around the smile is ~0.00039 vol points across all strikes.
# At round-3 vega (~4500/contract), 1 std of residual = ~$1.8 of price edge.
SMILE_A = 0.537       # quadratic coefficient on m
SMILE_C = 0.01253     # ATM (m=0) implied vol per sqrt(day)
SMILE_FLOOR = 0.005   # absolute minimum sigma
SMILE_RESIDUAL_STD = 0.00039   # noise floor for deviation; scale conviction

# Empirical per-strike sigma (per sqrt(day)) - the *market-consistent* fair
# (kept for reference / fallback / hybrid mode).
PER_STRIKE_SIGMA = {
    4000: 0.01253,
    4500: 0.01253,
    5000: 0.01266,
    5100: 0.01253,
    5200: 0.01267,
    5300: 0.01285,
    5400: 0.01200,   # market consistently below the quadratic smile
    5500: 0.01310,   # market consistently above the quadratic smile
    6000: 0.01310,
    6500: 0.01310,
}

# Pricing mode for vouchers:
#   "smile"      - trade against the historical smile (Orin's recommendation)
#   "per_strike" - trade against the empirical per-strike sigma
#   "hybrid"     - smile fair value, but only act on deviations > threshold
PRICING_MODE = "smile"

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
PARAMS: Dict[str, dict] = {
    HYDROGEL: {
        # Mean-reverts around 10000 with std ~30. EMA alpha=0.20 + 30%-anchor
        # blend tracks short-term noise tightly while still being pulled toward
        # the long-run mean. take_edge=2 captures most book mispricings.
        #
        # Sweep result history (full 3-day backtest):
        #   * tightening take/cap or panic-unwinding always *regresses* PnL.
        #   * The drawdown in submission 476806 (peak +$11k -> end +$4k) is
        #     just inventory risk; the position correctly mean-reverts.
        # Therefore unwind_frac / panic_frac default to 1.01 (disabled).
        "anchor": 10000.0,
        "ema_alpha": 0.20,
        "take_edge": 2.0,
        "clear_edge": 4.0,        # 482594 sweep: cl=4 vs cl=2.5 = +$168 preview; safe, well-tested
        "quote_offset": 1,
        "max_quote_size": 50,
        "soft_cap": 140,
        "unwind_frac": 1.01,     # disabled
        "panic_frac": 1.01,      # disabled
    },
    EXTRACT: {
        # VELVETFRUIT_EXTRACT is a slow random walk; PnL on directional MM
        # bleeds against trends. We *only* take on really obvious mispricings
        # and otherwise stay flat. We rely on it as the underlying for voucher
        # pricing rather than as a primary income stream.
        "enable": False,            # set True to re-enable directional MM
        "fast_alpha": 0.20,
        "slow_alpha": 0.02,
        "take_edge": 4.0,
        "clear_edge": 1.0,
        "quote_offset": 2,
        "max_quote_size": 15,
        "soft_cap": 60,
        "momentum_dead_zone": 0.5,
        "momentum_scale": 4.0,
        "max_target": 40,
    },
}

# Per-voucher params. For each voucher we trade against BS-smile theo.
# `take_edge` is in voucher-price units (cash).
VEV_PARAMS: Dict[str, dict] = {
    # Deep ITM: theo = S - K with ~$0.8 of noise; spread is wide (16-21).
    # Tight take/clear edges harvest the spread aggressively because the
    # underlying parity arg keeps theo locked at intrinsic. Backtest sweep
    # found this tightening adds ~+$9k full-day with no preview cost; the
    # 476806 submission realised +$162 / +$127 from these legs in 100k ts.
    "VEV_4000": {"take_edge": 0.5, "clear_edge": 0.25, "max_quote": 40, "soft_cap": 200, "hard_cap": 250, "make": True},
    "VEV_4500": {"take_edge": 0.5, "clear_edge": 0.25, "max_quote": 40, "soft_cap": 200, "hard_cap": 250, "make": True},
    # Mid-OTM smile strikes: in submission 476806, the smile-deviation trades
    # on K=5200/5300/5400 each lost ~$100 because the deviations did NOT
    # revert within 100k ts (it took the full day for the smile to drag the
    # prices back). We disable these strikes; sweep shows +$153 preview and
    # +$774 full-day improvement vs. trading them. Keep K=5000/5100/5500
    # active because they don't suffer from the 5400-style structural lag.
    "VEV_5000": {"take_edge": 2.0, "clear_edge": 1.0,  "max_quote": 25, "soft_cap": 80,  "hard_cap": 200, "make": True},
    "VEV_5100": {"take_edge": 2.0, "clear_edge": 1.0,  "max_quote": 25, "soft_cap": 80,  "hard_cap": 200, "make": True},
    "VEV_5200": {"take_edge": 999.0, "clear_edge": 1.0, "max_quote": 25, "soft_cap": 80, "hard_cap": 200, "make": False},
    "VEV_5300": {"take_edge": 999.0, "clear_edge": 1.5, "max_quote": 20, "soft_cap": 60, "hard_cap": 150, "make": False},
    "VEV_5400": {"take_edge": 999.0, "clear_edge": 1.0, "max_quote": 25, "soft_cap": 80, "hard_cap": 200, "make": False},
    "VEV_5500": {"take_edge": 1.5, "clear_edge": 0.75, "max_quote": 25, "soft_cap": 80,  "hard_cap": 200, "make": True},
    # Deep OTM: theoretical << 1; passive sells at >=1 only.
    "VEV_6000": {"take_edge": 0.5, "clear_edge": 0.5,  "max_quote": 20, "soft_cap": 200, "hard_cap": 250, "make": False},
    "VEV_6500": {"take_edge": 0.5, "clear_edge": 0.5,  "max_quote": 20, "soft_cap": 200, "hard_cap": 250, "make": False},
}

# Cap on total absolute portfolio delta (in units of underlying VELVETFRUIT_EXTRACT).
# Vouchers contribute their BS deltas; the spot itself contributes 1.
DELTA_CAP = 220.0


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
SQRT_2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def _npdf(x: float) -> float:
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if S <= 0 or K <= 0:
        return 0.0
    if sigma <= 0 or T <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _ncdf(d1) - K * _ncdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if S <= 0 or K <= 0:
        return 0.0
    if sigma <= 0 or T <= 0:
        return 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    return _ncdf(d1)


def smile_sigma(S: float, K: float, T: float) -> float:
    """Quadratic smile in log-moneyness/sqrt(T). The structural fair vol."""
    if T <= 0 or S <= 0 or K <= 0:
        return SMILE_C
    m = math.log(K / S) / math.sqrt(T)
    return max(SMILE_FLOOR, SMILE_A * m * m + SMILE_C)


def _det3(M: list) -> float:
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def fit_smile_live(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    """
    Closed-form least-squares fit of sigma = a*m^2 + b*m + c over (m, sigma)
    points. Returns (a, b, c). Falls back to historical smile constants if
    the system is degenerate or too few points are supplied.
    """
    n = len(points)
    if n < 3:
        return SMILE_A, 0.0, SMILE_C
    sm = ssm = sssm = ssssm = 0.0
    sy = sym = symm = 0.0
    for m, y in points:
        m2 = m * m
        sm += m
        ssm += m2
        sssm += m2 * m
        ssssm += m2 * m2
        sy += y
        sym += y * m
        symm += y * m2
    M = [[ssssm, sssm, ssm],
         [sssm,  ssm,  sm],
         [ssm,   sm,   float(n)]]
    Y = [symm, sym, sy]
    D = _det3(M)
    if abs(D) < 1e-15:
        return SMILE_A, 0.0, SMILE_C
    Ma = [[Y[0], M[0][1], M[0][2]],
          [Y[1], M[1][1], M[1][2]],
          [Y[2], M[2][1], M[2][2]]]
    Mb = [[M[0][0], Y[0], M[0][2]],
          [M[1][0], Y[1], M[1][2]],
          [M[2][0], Y[2], M[2][2]]]
    Mc = [[M[0][0], M[0][1], Y[0]],
          [M[1][0], M[1][1], Y[1]],
          [M[2][0], M[2][1], Y[2]]]
    a = _det3(Ma) / D
    b = _det3(Mb) / D
    c = _det3(Mc) / D
    return a, b, c


def per_strike_sigma(K: int, S: float, T: float) -> float:
    sig = PER_STRIKE_SIGMA.get(int(K))
    if sig is not None:
        return sig
    return smile_sigma(S, K, T)


def voucher_sigma(K: int, S: float, T: float) -> float:
    """
    Sigma used to compute *fair value* of the voucher.
    Mode "smile" follows Orin's hint - trade against the structural smile.
    Mode "per_strike" treats each strike's empirical IV as fair.
    """
    if PRICING_MODE == "smile":
        return smile_sigma(S, K, T)
    elif PRICING_MODE == "per_strike":
        return per_strike_sigma(int(K), S, T)
    else:  # hybrid
        return smile_sigma(S, K, T)


def voucher_theo(S: float, K: float, T: float) -> float:
    sig = voucher_sigma(int(K), S, T)
    return bs_call(S, K, T, sig)


def voucher_delta(S: float, K: float, T: float) -> float:
    sig = voucher_sigma(int(K), S, T)
    return bs_delta(S, K, T, sig)


def implied_vol(C: float, S: float, K: float, T: float,
                lo: float = 1e-5, hi: float = 5.0) -> float:
    """Bisection-invert BS for sigma given call mid C. Returns 0 if unsolvable."""
    if T <= 0 or S <= 0 or K <= 0:
        return 0.0
    intrinsic = max(S - K, 0.0)
    if C <= intrinsic + 1e-9:
        return 0.0
    if C >= S:
        return hi
    a, b = lo, hi
    for _ in range(40):
        m = 0.5 * (a + b)
        if bs_call(S, K, T, m) > C:
            b = m
        else:
            a = m
    return 0.5 * (a + b)


def voucher_vega(S: float, K: float, T: float, sigma: float) -> float:
    """Vega = ∂C/∂σ. Used to convert IV deviations to price deviations."""
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    return S * sqrtT * _npdf(d1)


# ---------------------------------------------------------------------------
# Order-book helpers
# ---------------------------------------------------------------------------
def _top_of_book(d: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    bb = max(d.buy_orders.keys()) if d.buy_orders else None
    ba = min(d.sell_orders.keys()) if d.sell_orders else None
    return bb, ba


def _mid(bb: Optional[int], ba: Optional[int]) -> Optional[float]:
    if bb is not None and ba is not None:
        return 0.5 * (bb + ba)
    if bb is not None:
        return float(bb)
    if ba is not None:
        return float(ba)
    return None


def _microprice(d: OrderDepth) -> Optional[float]:
    if not d.buy_orders or not d.sell_orders:
        return None
    bb = max(d.buy_orders.keys())
    ba = min(d.sell_orders.keys())
    bv = d.buy_orders[bb]
    av = -d.sell_orders[ba]
    tot = bv + av
    if tot <= 0:
        return 0.5 * (bb + ba)
    return (bb * av + ba * bv) / tot


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------
class Trader:
    def __init__(self) -> None:
        self.hydrogel_ema: Optional[float] = None
        self.extract_fast: Optional[float] = None
        self.extract_slow: Optional[float] = None
        # Last observed underlying mid - falls back to anchor on first tick.
        self.last_extract_mid: Optional[float] = None
        # Override-able TTE (days) at timestamp = 0 for backtests.
        self.start_tte_days: float = ROUND3_START_TTE_DAYS

    # --------------------- delta-1 market making -------------------------
    def _trade_delta1(
        self,
        product: str,
        depth: OrderDepth,
        fair: float,
        cfg: dict,
        position: int,
    ) -> List[Order]:
        """
        Standard take/clear/make MM with one *targeted* enhancement: when the
        running position has crossed `unwind_frac * limit`, our passive
        unwind quote drops from `fair + offset` to `fair` (or even `bb` at
        `panic_frac`). This addresses the 474614/476806 trap where we held
        +200 HYDROGEL through a 70-tick downtrend because our 9971 sells
        were stranded above the moving best bid.
        """
        orders: List[Order] = []
        limit = LIMIT[product]
        soft_cap = cfg["soft_cap"]
        max_size = cfg["max_quote_size"]
        offset = cfg["quote_offset"]

        buy_used = 0
        sell_used = 0

        # 1) Take aggressively cheap asks below fair - take_edge.
        if depth.sell_orders:
            for ask, sz in sorted(depth.sell_orders.items()):
                if ask > fair - cfg["take_edge"]:
                    break
                cap = limit - (position + buy_used)
                if cap <= 0:
                    break
                qty = min(-sz, cap)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_used += qty

        # 2) Hit rich bids above fair + take_edge.
        if depth.buy_orders:
            for bid, sz in sorted(depth.buy_orders.items(), reverse=True):
                if bid < fair + cfg["take_edge"]:
                    break
                cap = limit + (position - sell_used)
                if cap <= 0:
                    break
                qty = min(sz, cap)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_used += qty

        # 3) Inventory clearing at fair + small edge.
        net = position + buy_used - sell_used
        if net > 0:
            target = int(round(fair + cfg["clear_edge"]))
            interest = sum(v for p, v in depth.buy_orders.items() if p >= target)
            cap = limit + (position - sell_used)
            qty = min(net, interest, cap)
            if qty > 0:
                orders.append(Order(product, target, -qty))
                sell_used += qty
        elif net < 0:
            target = int(round(fair - cfg["clear_edge"]))
            interest = sum(-v for p, v in depth.sell_orders.items() if p <= target)
            cap = limit - (position + buy_used)
            qty = min(-net, interest, cap)
            if qty > 0:
                orders.append(Order(product, target, qty))
                buy_used += qty

        # 4) Passive market making with inventory-aware unwind.
        bb, ba = _top_of_book(depth)
        if bb is not None and ba is not None:
            fair_int = int(round(fair))
            bid_px = min(bb + 1, fair_int - offset)
            ask_px = max(ba - 1, fair_int + offset)
            if bid_px >= ask_px:
                bid_px = fair_int - offset
                ask_px = fair_int + offset

            buy_cap = limit - (position + buy_used)
            sell_cap = limit + (position - sell_used)

            buy_size = max(0, min(max_size, buy_cap))
            sell_size = max(0, min(max_size, sell_cap))

            if position > soft_cap:
                buy_size = 0
            elif position > soft_cap // 2:
                buy_size = min(buy_size, max_size // 2)

            if position < -soft_cap:
                sell_size = 0
            elif position < -soft_cap // 2:
                sell_size = min(sell_size, max_size // 2)

            # Aggressive unwind: when we're stuck near the limit, lower the
            # passive ask (or raise the passive bid) so we actually get filled.
            unwind_frac = cfg.get("unwind_frac", 1.01)
            panic_frac = cfg.get("panic_frac", 1.01)
            if position >= int(limit * unwind_frac):
                ask_px = min(ask_px, fair_int)
            if position >= int(limit * panic_frac):
                ask_px = min(ask_px, bb)
            if position <= -int(limit * unwind_frac):
                bid_px = max(bid_px, fair_int)
            if position <= -int(limit * panic_frac):
                bid_px = max(bid_px, ba)

            if buy_size > 0 and bid_px > 0:
                orders.append(Order(product, bid_px, buy_size))
            if sell_size > 0 and ask_px > 0:
                orders.append(Order(product, ask_px, -sell_size))

        return orders

    # --------------------- voucher trading -------------------------------
    def _trade_voucher(
        self,
        product: str,
        depth: OrderDepth,
        theo: float,
        cfg: dict,
        position: int,
        delta_budget_buy: float,
        delta_budget_sell: float,
        delta_per_unit: float,
        conviction_long: float = 1.0,
        conviction_short: float = 1.0,
    ) -> Tuple[List[Order], int, int]:
        """
        Trade a single voucher around `theo` (the BS-smile theoretical price).

        delta_budget_buy:  how much (positive) net delta the portfolio can still
                           absorb if we BUY this voucher.
        delta_budget_sell: how much (positive) net delta we can REMOVE if we
                           SELL this voucher (i.e. how negative we can go).
        delta_per_unit:    BS delta of one voucher contract.
        """
        orders: List[Order] = []
        limit = LIMIT[product]
        soft_cap_base = cfg["soft_cap"]
        hard_cap_base = cfg.get("hard_cap", limit)

        buy_used = 0
        sell_used = 0

        def _max_buy_units_by_delta() -> int:
            if delta_per_unit <= 1e-6:
                return 10**9
            return max(0, int(delta_budget_buy / delta_per_unit))

        def _max_sell_units_by_delta() -> int:
            if delta_per_unit <= 1e-6:
                return 10**9
            return max(0, int(delta_budget_sell / delta_per_unit))

        # Volume per Orin: scale capacity by conviction strength on each side.
        max_long_total = max(0, int(min(limit, hard_cap_base) * conviction_long))
        max_short_total = max(0, int(min(limit, hard_cap_base) * conviction_short))
        soft_cap_long = max(1, int(soft_cap_base * conviction_long))
        soft_cap_short = max(1, int(soft_cap_base * conviction_short))

        # 1) Take cheap asks (price + take_edge < theo).
        if depth.sell_orders:
            for ask, sz in sorted(depth.sell_orders.items()):
                if ask > theo - cfg["take_edge"]:
                    break
                cap = max_long_total - (position + buy_used)
                cap = min(cap, _max_buy_units_by_delta() - buy_used)
                if cap <= 0:
                    break
                qty = min(-sz, cap)
                if qty > 0:
                    orders.append(Order(product, ask, qty))
                    buy_used += qty

        # 2) Hit rich bids (price - take_edge > theo).
        if depth.buy_orders:
            for bid, sz in sorted(depth.buy_orders.items(), reverse=True):
                if bid < theo + cfg["take_edge"]:
                    break
                cap = max_short_total + (position - sell_used)
                cap = min(cap, _max_sell_units_by_delta() - sell_used)
                if cap <= 0:
                    break
                qty = min(sz, cap)
                if qty > 0:
                    orders.append(Order(product, bid, -qty))
                    sell_used += qty

        # 3) Inventory clearing at theo + tighter edge.
        net = position + buy_used - sell_used
        if net > 0:
            target_px = max(1, int(math.ceil(theo + cfg["clear_edge"])))
            interest = sum(v for p, v in depth.buy_orders.items() if p >= target_px)
            cap = limit + (position - sell_used)
            qty = min(net, interest, cap)
            if qty > 0:
                orders.append(Order(product, target_px, -qty))
                sell_used += qty
        elif net < 0:
            target_px = max(1, int(math.floor(theo - cfg["clear_edge"])))
            interest = sum(-v for p, v in depth.sell_orders.items() if p <= target_px)
            cap = limit - (position + buy_used)
            qty = min(-net, interest, cap)
            if qty > 0:
                orders.append(Order(product, target_px, qty))
                buy_used += qty

        # 4) Passive making at theo +/- (round-to-tick + 1).
        if cfg.get("make", True):
            bb, ba = _top_of_book(depth)
            if bb is not None and ba is not None:
                # Use a 1-tick offset; for very small theo (deep OTM) ensure >= 1.
                bid_px = max(1, int(math.floor(theo - 1)))
                ask_px = max(bid_px + 1, int(math.ceil(theo + 1)))

                # Stay inside the prevailing book if possible.
                bid_px = min(bid_px, bb + 1) if bb + 1 < ask_px else bid_px
                ask_px = max(ask_px, ba - 1) if ba - 1 > bid_px else ask_px
                if bid_px >= ask_px:
                    bid_px = max(1, int(math.floor(theo - 1)))
                    ask_px = max(bid_px + 1, int(math.ceil(theo + 1)))

                max_q = cfg["max_quote"]

                buy_cap = max_long_total - (position + buy_used)
                sell_cap = max_short_total + (position - sell_used)

                buy_size = max(0, min(max_q, buy_cap, _max_buy_units_by_delta() - buy_used))
                sell_size = max(0, min(max_q, sell_cap, _max_sell_units_by_delta() - sell_used))

                if position >= soft_cap_long:
                    buy_size = 0
                elif position >= soft_cap_long // 2:
                    buy_size = min(buy_size, max(1, max_q // 3))

                if position <= -soft_cap_short:
                    sell_size = 0
                elif position <= -soft_cap_short // 2:
                    sell_size = min(sell_size, max(1, max_q // 3))

                if buy_size > 0 and bid_px > 0:
                    orders.append(Order(product, bid_px, buy_size))
                if sell_size > 0 and ask_px > 0:
                    orders.append(Order(product, ask_px, -sell_size))

        return orders, buy_used, sell_used

    # --------------------- delta-1 fair value ----------------------------
    def _fair_hydrogel(self, depth: OrderDepth) -> Optional[float]:
        bb, ba = _top_of_book(depth)
        mid = _mid(bb, ba)
        if mid is None:
            return None
        mp = _microprice(depth)
        x = mp if mp is not None else mid
        cfg = PARAMS[HYDROGEL]
        if self.hydrogel_ema is None:
            self.hydrogel_ema = cfg["anchor"]
        a = cfg["ema_alpha"]
        # Mean-revert: shrink slightly toward anchor.
        self.hydrogel_ema = a * x + (1 - a) * self.hydrogel_ema
        # Blend EMA with anchor to keep it pinned around 10000.
        return 0.7 * self.hydrogel_ema + 0.3 * cfg["anchor"]

    def _trade_extract(
        self,
        depth: OrderDepth,
        position: int,
    ) -> List[Order]:
        """Momentum-aware MM for VELVETFRUIT_EXTRACT (slow random walk)."""
        cfg = PARAMS[EXTRACT]
        bb, ba = _top_of_book(depth)
        mid = _mid(bb, ba)
        if mid is None:
            return []
        mp = _microprice(depth)
        x = mp if mp is not None else mid

        if self.extract_fast is None:
            self.extract_fast = x
        if self.extract_slow is None:
            self.extract_slow = x
        self.extract_fast = cfg["fast_alpha"] * x + (1 - cfg["fast_alpha"]) * self.extract_fast
        self.extract_slow = cfg["slow_alpha"] * x + (1 - cfg["slow_alpha"]) * self.extract_slow

        momentum_raw = self.extract_fast - self.extract_slow
        if abs(momentum_raw) < cfg["momentum_dead_zone"]:
            momentum = 0.0
        else:
            sign = 1 if momentum_raw > 0 else -1
            momentum = sign * (abs(momentum_raw) - cfg["momentum_dead_zone"])

        fair = self.extract_fast + 0.5 * momentum

        # Momentum-driven target position (skew long when trending up, short when down).
        target = int(cfg["momentum_scale"] * momentum)
        target = max(-cfg["max_target"], min(cfg["max_target"], target))

        orders: List[Order] = []
        limit = LIMIT[EXTRACT]
        soft_cap = cfg["soft_cap"]
        offset = cfg["quote_offset"]
        max_size = cfg["max_quote_size"]

        buy_used = 0
        sell_used = 0

        # 1) Take if there's a *big* mispricing.
        if depth.sell_orders:
            for ask, sz in sorted(depth.sell_orders.items()):
                if ask > fair - cfg["take_edge"]:
                    break
                cap = limit - (position + buy_used)
                if cap <= 0:
                    break
                qty = min(-sz, cap)
                if qty > 0:
                    orders.append(Order(EXTRACT, ask, qty))
                    buy_used += qty
        if depth.buy_orders:
            for bid, sz in sorted(depth.buy_orders.items(), reverse=True):
                if bid < fair + cfg["take_edge"]:
                    break
                cap = limit + (position - sell_used)
                if cap <= 0:
                    break
                qty = min(sz, cap)
                if qty > 0:
                    orders.append(Order(EXTRACT, bid, -qty))
                    sell_used += qty

        # 2) Drive net position toward `target`.
        net = position + buy_used - sell_used
        if net > target:
            reduce_px = int(round(fair + cfg["clear_edge"]))
            interest = sum(v for p, v in depth.buy_orders.items() if p >= reduce_px)
            cap = limit + (position - sell_used)
            qty = min(net - target, interest, cap)
            if qty > 0:
                orders.append(Order(EXTRACT, reduce_px, -qty))
                sell_used += qty
        elif net < target:
            add_px = int(round(fair - cfg["clear_edge"]))
            interest = sum(-v for p, v in depth.sell_orders.items() if p <= add_px)
            cap = limit - (position + buy_used)
            qty = min(target - net, interest, cap)
            if qty > 0:
                orders.append(Order(EXTRACT, add_px, qty))
                buy_used += qty

        # 3) Passive MM. Wide spreads -> place 2 inside.
        if bb is not None and ba is not None:
            fair_int = int(round(fair))
            bid_px = min(bb + 1, fair_int - offset)
            ask_px = max(ba - 1, fair_int + offset)
            if bid_px >= ask_px:
                bid_px = fair_int - offset
                ask_px = fair_int + offset

            buy_cap = limit - (position + buy_used)
            sell_cap = limit + (position - sell_used)

            buy_size = max(0, min(max_size, buy_cap))
            sell_size = max(0, min(max_size, sell_cap))

            if position > soft_cap:
                buy_size = 0
            elif position > soft_cap // 2:
                buy_size = min(buy_size, max_size // 2)
            if position < -soft_cap:
                sell_size = 0
            elif position < -soft_cap // 2:
                sell_size = min(sell_size, max_size // 2)

            if buy_size > 0 and bid_px > 0:
                orders.append(Order(EXTRACT, bid_px, buy_size))
            if sell_size > 0 and ask_px > 0:
                orders.append(Order(EXTRACT, ask_px, -sell_size))

        return orders

    # --------------------- main entry ------------------------------------
    def run(self, state: TradingState):
        # Restore persistent state.
        if state.traderData:
            try:
                saved = jsonpickle.decode(state.traderData)
                self.hydrogel_ema = saved.get("hydrogel_ema", None)
                self.extract_fast = saved.get("extract_fast", None)
                self.extract_slow = saved.get("extract_slow", None)
                self.last_extract_mid = saved.get("last_extract_mid", None)
            except Exception:
                pass

        result: Dict[str, List[Order]] = {}
        positions = state.position

        # ---- Underlying spot first (we need it for voucher theos) ----
        extract_depth = state.order_depths.get(EXTRACT)
        spot: Optional[float] = None
        if extract_depth is not None:
            ebb, eba = _top_of_book(extract_depth)
            mid_extract = _mid(ebb, eba)
            mp_extract = _microprice(extract_depth)
            spot = mp_extract if mp_extract is not None else mid_extract
            if spot is not None:
                self.last_extract_mid = float(spot)
        if spot is None:
            spot = self.last_extract_mid if self.last_extract_mid is not None else PARAMS[EXTRACT]["anchor"]

        # ---- TTE in days ----
        T_days = max(MIN_T_DAYS, self.start_tte_days - state.timestamp / TIMESTAMPS_PER_DAY)

        # ---- Live smile fit ----
        # Compute live IV for each "smile-bearing" strike (5000-5500), then
        # fit a quadratic in log-moneyness/sqrt(T). This makes the strategy
        # invariant to TTE/regime mistakes - both market IV and the fitted
        # smile use the same T, so RELATIVE deviations remain meaningful.
        smile_a = SMILE_A
        smile_b = 0.0
        smile_c = SMILE_C
        live_iv: Dict[int, float] = {}  # K -> iv_market
        live_m: Dict[int, float] = {}   # K -> moneyness
        smile_K_set = (5000, 5100, 5200, 5300, 5400, 5500)
        if T_days > 0 and spot > 0:
            pts: List[Tuple[float, float]] = []
            for K in smile_K_set:
                v = f"VEV_{K}"
                d_v = state.order_depths.get(v)
                if d_v is None:
                    continue
                bb_v, ba_v = _top_of_book(d_v)
                vmid = _mid(bb_v, ba_v)
                if vmid is None or vmid <= 0:
                    continue
                iv = implied_vol(vmid, spot, float(K), T_days)
                if iv <= 0:
                    continue
                m = math.log(K / spot) / math.sqrt(T_days)
                live_iv[K] = iv
                live_m[K] = m
                pts.append((m, iv))
            if len(pts) >= 3:
                smile_a, smile_b, smile_c = fit_smile_live(pts)

        def _live_smile_sigma(K: int) -> float:
            if K in live_m:
                m = live_m[K]
            elif T_days > 0:
                m = math.log(K / spot) / math.sqrt(T_days)
            else:
                m = 0.0
            return max(SMILE_FLOOR, smile_a * m * m + smile_b * m + smile_c)

        def _live_voucher_theo(K: int) -> float:
            return bs_call(spot, float(K), T_days, _live_smile_sigma(K))

        def _live_voucher_delta(K: int) -> float:
            return bs_delta(spot, float(K), T_days, _live_smile_sigma(K))

        # ---- Compute current portfolio delta from existing voucher positions ----
        current_delta = float(positions.get(EXTRACT, 0))
        for v in VEVS:
            K = STRIKE_OF[v]
            d = (_live_voucher_delta(K)
                 if PRICING_MODE in ("smile", "hybrid") else
                 voucher_delta(spot, K, T_days))
            current_delta += d * positions.get(v, 0)

        # Allowable additional delta we can take this tick (before sending orders)
        delta_budget_long = max(0.0, DELTA_CAP - current_delta)   # how much more positive we can go
        delta_budget_short = max(0.0, DELTA_CAP + current_delta)  # how much more negative we can go

        # ---- Trade vouchers (priced off `spot`) ----
        # Per Orin's "Volume" hint, conviction (and thus size) scales with the
        # *magnitude* of the IV deviation from the smile. Strong deviation =>
        # bigger position; small deviation => stay closer to flat.
        # Helper that returns the active fair price + delta per voucher.
        def _theo_for(K: int) -> float:
            if PRICING_MODE in ("smile", "hybrid"):
                return _live_voucher_theo(K)
            return voucher_theo(spot, K, T_days)

        def _delta_for(K: int) -> float:
            if PRICING_MODE in ("smile", "hybrid"):
                return _live_voucher_delta(K)
            return voucher_delta(spot, K, T_days)

        vev_order = sorted(VEVS, key=lambda v: -_delta_for(STRIKE_OF[v]))
        for v in vev_order:
            depth = state.order_depths.get(v)
            if depth is None:
                continue
            K = STRIKE_OF[v]
            theo = _theo_for(K)
            d = _delta_for(K)
            cfg = VEV_PARAMS[v]
            pos = positions.get(v, 0)

            # ----- Conviction from live IV vs LIVE smile -----
            conv_long, conv_short = 1.0, 1.0
            if (PRICING_MODE in ("smile", "hybrid")
                    and K in smile_K_set
                    and K in live_iv):
                iv_market = live_iv[K]
                iv_smile = _live_smile_sigma(K)
                z = (iv_smile - iv_market) / max(SMILE_RESIDUAL_STD, 1e-6)
                # z > 0 => market vol BELOW smile => underpriced => long bias
                # z < 0 => market vol ABOVE smile => overpriced  => short bias
                strength = max(0.0, min(1.0, abs(z) / 3.0))
                if z > 0:    # buy this voucher
                    conv_long = max(0.2, strength)
                    conv_short = max(0.2, 1.0 - strength)
                elif z < 0:  # sell this voucher
                    conv_short = max(0.2, strength)
                    conv_long = max(0.2, 1.0 - strength)

            orders, buy_used, sell_used = self._trade_voucher(
                v, depth, theo, cfg, pos,
                delta_budget_buy=delta_budget_long,
                delta_budget_sell=delta_budget_short,
                delta_per_unit=d,
                conviction_long=conv_long,
                conviction_short=conv_short,
            )
            if orders:
                result[v] = orders

            # Decrement budgets by net delta consumed this tick.
            net_units = buy_used - sell_used
            if net_units > 0:
                delta_budget_long = max(0.0, delta_budget_long - d * net_units)
                delta_budget_short = delta_budget_short + d * net_units
            elif net_units < 0:
                delta_budget_short = max(0.0, delta_budget_short + d * net_units)  # net_units<0
                delta_budget_long = delta_budget_long - d * net_units              # adds back

        # ---- VELVETFRUIT_EXTRACT ----
        # Default OFF: backtests show ~$45k/day bleed from MM-ing this slow
        # random walk. The HYDROGEL+voucher legs alone net ~+30k/day. If
        # PARAMS[EXTRACT]["enable"] is set True, the momentum-aware MM runs.
        if extract_depth is not None and PARAMS[EXTRACT].get("enable", False):
            ext_orders = self._trade_extract(extract_depth, positions.get(EXTRACT, 0))
            if ext_orders:
                result[EXTRACT] = ext_orders

        # ---- HYDROGEL_PACK (independent delta-1 MM) ----
        hydro_depth = state.order_depths.get(HYDROGEL)
        if hydro_depth is not None:
            fair = self._fair_hydrogel(hydro_depth)
            if fair is not None:
                cfg = PARAMS[HYDROGEL]
                pos = positions.get(HYDROGEL, 0)
                hyd_orders = self._trade_delta1(HYDROGEL, hydro_depth, fair, cfg, pos)
                if hyd_orders:
                    result[HYDROGEL] = hyd_orders

        # ---- Persist state ----
        trader_data = jsonpickle.encode({
            "hydrogel_ema": self.hydrogel_ema,
            "extract_fast": self.extract_fast,
            "extract_slow": self.extract_slow,
            "last_extract_mid": self.last_extract_mid,
        })

        conversions = 0
        return result, conversions, trader_data

"""
Round 4 - "Hello, I'm Mark" Trader
==================================

**Production path (default):** Round-3-class **HYDROGEL_PACK** delta-1 MM plus
**voucher** market-making with a **live parabolic smile** fit and residual
conviction. Historical replay PnL is dominated here.

**Round-4-specific machinery (defaults OFF after replay):**

  * **Counterparty signals** — Trade tape exposes Mark IDs; values used at
    runtime are the embedded `DEFAULT_MARK_ALPHA` below. (IMC accepts a
    **single** `round4.py` upload, so you cannot ship `mark_alpha.json` with
    the bot; the optional `load_mark_alpha_config()` only helps **local** runs
    when that file lives next to this module. After `counterparty_analysis.py`,
    copy any new numbers into `DEFAULT_MARK_ALPHA` before uploading.)
  * **Passive / signal EXTRACT** paths stay off unless `PARAMS[EXTRACT]`
    toggles are enabled (spread vs alpha).
  * **Gamma scalp + EXTRACT delta hedge** — Implemented; `GAMMA_SCALP_ENABLED`
    is False (theta and hedge frictions hurt in replay).

Tunables for one-axis backtests: `imb_gated_skew` (HYDROGEL), smile fit gates
below, and EXTRACT `enable_*` flags.
"""
from datamodel import OrderDepth, TradingState, Order, Trade
from typing import Dict, List, Optional, Tuple
import copy
import json
import math
from pathlib import Path
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
# Time-to-expiry
# ---------------------------------------------------------------------------
TIMESTAMPS_PER_DAY = 1_000_000
ROUND4_START_TTE_DAYS = 4.0   # round 4 = day 4 in vouchers' 7-day life
MIN_T_DAYS = 0.05

# ---------------------------------------------------------------------------
# Volatility smile
# ---------------------------------------------------------------------------
SMILE_A = 0.537
SMILE_C = 0.01253
SMILE_FLOOR = 0.005
SMILE_RESIDUAL_STD = 0.00039

# Realized daily vol of EXTRACT in historical data (used for gamma sizing).
REALIZED_VOL_DAILY = 0.0216

PER_STRIKE_SIGMA = {
    4000: 0.01253, 4500: 0.01253,
    5000: 0.01266, 5100: 0.01253, 5200: 0.01267,
    5300: 0.01285, 5400: 0.01200, 5500: 0.01310,
    6000: 0.01310, 6500: 0.01310,
}

PRICING_MODE = "smile"

# ---------------------------------------------------------------------------
# Live smile fit: optional wing strikes (quality-filtered; see _smile_depth_ok)
# ---------------------------------------------------------------------------
SMILE_FIT_STRIKES = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500)
SMILE_FIT_MIN_DEPTH = 3          # min contracts on best bid and best ask
SMILE_FIT_MAX_SPREAD = 40        # max (ask - bid) in price ticks; skip illiquid

# ---------------------------------------------------------------------------
# Counterparty signal config
# ---------------------------------------------------------------------------
# IMC production: this dict is the only source (no sidecar file on upload).
# Local: `mark_alpha.json` can merge in via load_mark_alpha_config() if you keep it in repo.
DEFAULT_MARK_ALPHA = {
    EXTRACT: {
        "Mark 67": {"BUY":  {"alpha_1k": +2.24, "alpha_5k": +1.92, "n": 165}},
        "Mark 22": {"SELL": {"alpha_1k": -1.56, "alpha_5k": -1.87, "n": 101}},
        "Mark 49": {"SELL": {"alpha_1k": -2.14, "alpha_5k": -1.99, "n": 105}},
    },
}
SIGNAL_HALF_LIFE_TICKS = 5000        # alpha exponentially decays
SIGNAL_INITIAL_FRAC = 0.6            # we use 60% of the measured 1k alpha as our shift
                                     # (realistic vs idealised; tuneable)

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
PARAMS: Dict[str, dict] = {
    HYDROGEL: {
        # R4 data analysis discovered a strong threshold signal in the order
        # book volume imbalance:
        #     imb > +0.2  =>  mid moves +$3.67 in next 10 ticks (n=303)
        #     imb < -0.2  =>  mid moves -$3.46 in next 10 ticks (n=297)
        #     |imb| < 0.2 =>  ~0 (noise)
        # We add a fair-value tilt of `imb_scale * imb` whenever |imb| > thr.
        "anchor": 10000.0,
        # fair = (1-w)*ema(microprice) + w*anchor.  R3 used w=0.3 (0.7/0.3 blend).
        # R4 3-day backtest: w=0.3 -> 153.1K; w=0.45 -> 156.4K; w=0 (no anchor) -> 15K.
        "anchor_blend": 0.45,
        "ema_alpha": 0.20,
        "take_edge": 2.0,
        "clear_edge": 0.25,        # R4 sweep: 0.25 vs 4.0 = +$17K total
        "quote_offset": 1,
        "max_quote_size": 50,
        "soft_cap": 140,
        "unwind_frac": 1.01,
        "panic_frac": 1.01,
        "imb_threshold": 0.20,
        "imb_scale": 0.0,        # fair-tilt approach FAILED in backtest (-$400)
        "imb_quote_skew": False,  # always-on size skew: ~flat vs off in replay
        # Gated skew: same one-sided pull as imb_quote_skew, only when spread
        # is wide and |imb| is extreme (avoids blunt global tilt). Default off.
        "imb_gated_skew": False,
        "imb_gate_min_spread": 4,
        "imb_gate_extreme": 0.28,
        # Optional AS-style skew: fair -= inv_kappa * pos.  Tuned 0.01–0.04 on
        # R4 data: all regressed PnL vs 0 (inventory reduction costs edge).
        "inv_kappa": 0.0,
    },
    EXTRACT: {
        # Per Orin's hint we built a Mark-avoidance passive MM (withdraw
        # quotes when an informed Mark is active). Backtest result: -$103
        # across 3 days vs. baseline (essentially break-even). Defaulted
        # OFF because the EXTRACT spread (~$5) is too tight for any MM
        # strategy to consistently beat once execution friction is paid.
        "enable_signal_trades": False,
        "enable_passive_mm": False,
        "mm_quote_size": 15,
        "mm_take_edge": 3.5,
        "mm_soft_cap": 60,
        "max_position_per_signal": 60,
        "signal_lean_fair": 1.0,
        "soft_cap": 120,
    },
}

# Per-voucher params.  Round-3 winning: 5200/5300/5400 disabled as MM legs.
# In round 4 we keep them disabled as MM, but re-enable as **long-only gamma
# legs** through the gamma_scalp module below.
VEV_PARAMS: Dict[str, dict] = {
    "VEV_4000": {"take_edge": 0.5, "clear_edge": 0.25, "max_quote": 40, "soft_cap": 200, "hard_cap": 250, "make": True},
    "VEV_4500": {"take_edge": 0.5, "clear_edge": 0.25, "max_quote": 40, "soft_cap": 200, "hard_cap": 250, "make": True},
    "VEV_5000": {"take_edge": 2.0, "clear_edge": 1.0,  "max_quote": 25, "soft_cap": 80,  "hard_cap": 200, "make": True},
    "VEV_5100": {"take_edge": 2.0, "clear_edge": 1.0,  "max_quote": 25, "soft_cap": 80,  "hard_cap": 200, "make": True},
    "VEV_5200": {"take_edge": 999.0, "clear_edge": 1.0, "max_quote": 25, "soft_cap": 80, "hard_cap": 200, "make": False},
    "VEV_5300": {"take_edge": 999.0, "clear_edge": 1.5, "max_quote": 20, "soft_cap": 60, "hard_cap": 150, "make": False},
    "VEV_5400": {"take_edge": 999.0, "clear_edge": 1.0, "max_quote": 25, "soft_cap": 80, "hard_cap": 200, "make": False},
    "VEV_5500": {"take_edge": 1.5, "clear_edge": 0.75, "max_quote": 25, "soft_cap": 80,  "hard_cap": 200, "make": True},
    "VEV_6000": {"take_edge": 0.5, "clear_edge": 0.5,  "max_quote": 20, "soft_cap": 200, "hard_cap": 250, "make": False},
    "VEV_6500": {"take_edge": 0.5, "clear_edge": 0.5,  "max_quote": 20, "soft_cap": 200, "hard_cap": 250, "make": False},
}

# ---------------------------------------------------------------------------
# Gamma scalping
# ---------------------------------------------------------------------------
# Long-gamma target positions on the disabled smile strikes. The gamma scalp
# *should* harvest realized vol > implied vol but in 3-day backtests it lost
# ~$8k vs disabled. Likely cause: theta drag on long positions exceeds the
# delta-hedge gamma capture given our hedge spread costs. Default OFF.
GAMMA_SCALP_ENABLED = False
GAMMA_TARGET_PER_STRIKE = {
    5200: 50,
    5300: 60,
    5400: 60,
    5500: 30,
}
# Maximum portfolio absolute delta (in EXTRACT units) - hard cap for hedger.
DELTA_HEDGE_CAP = 180

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
    if T <= 0 or S <= 0 or K <= 0:
        return SMILE_C
    m = math.log(K / S) / math.sqrt(T)
    return max(SMILE_FLOOR, SMILE_A * m * m + SMILE_C)


def per_strike_sigma(K: int, S: float, T: float) -> float:
    sig = PER_STRIKE_SIGMA.get(int(K))
    return sig if sig is not None else smile_sigma(S, K, T)


def voucher_sigma(K: int, S: float, T: float) -> float:
    if PRICING_MODE == "smile":
        return smile_sigma(S, K, T)
    elif PRICING_MODE == "per_strike":
        return per_strike_sigma(int(K), S, T)
    return smile_sigma(S, K, T)


def voucher_theo(S: float, K: float, T: float) -> float:
    return bs_call(S, K, T, voucher_sigma(int(K), S, T))


def voucher_delta(S: float, K: float, T: float) -> float:
    return bs_delta(S, K, T, voucher_sigma(int(K), S, T))


def implied_vol(C: float, S: float, K: float, T: float,
                lo: float = 1e-5, hi: float = 5.0) -> float:
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


def _det3(M: list) -> float:
    return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))


def fit_smile_live(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
    n = len(points)
    if n < 3:
        return SMILE_A, 0.0, SMILE_C
    sm = ssm = sssm = ssssm = 0.0
    sy = sym = symm = 0.0
    for m, y in points:
        m2 = m * m
        sm += m;  ssm += m2
        sssm += m2 * m;  ssssm += m2 * m2
        sy += y;  sym += y * m;  symm += y * m2
    M = [[ssssm, sssm, ssm], [sssm, ssm, sm], [ssm, sm, float(n)]]
    Y = [symm, sym, sy]
    D = _det3(M)
    if abs(D) < 1e-15:
        return SMILE_A, 0.0, SMILE_C
    Ma = [[Y[0], M[0][1], M[0][2]], [Y[1], M[1][1], M[1][2]], [Y[2], M[2][1], M[2][2]]]
    Mb = [[M[0][0], Y[0], M[0][2]], [M[1][0], Y[1], M[1][2]], [M[2][0], Y[2], M[2][2]]]
    Mc = [[M[0][0], M[0][1], Y[0]], [M[1][0], M[1][1], Y[1]], [M[2][0], M[2][1], Y[2]]]
    return _det3(Ma) / D, _det3(Mb) / D, _det3(Mc) / D


# ---------------------------------------------------------------------------
# Order-book helpers
# ---------------------------------------------------------------------------
def _top_of_book(d: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    bb = max(d.buy_orders.keys()) if d.buy_orders else None
    ba = min(d.sell_orders.keys()) if d.sell_orders else None
    return bb, ba


def _mid(bb, ba) -> Optional[float]:
    if bb is not None and ba is not None: return 0.5 * (bb + ba)
    if bb is not None: return float(bb)
    if ba is not None: return float(ba)
    return None


def _microprice(d: OrderDepth) -> Optional[float]:
    if not d.buy_orders or not d.sell_orders: return None
    bb = max(d.buy_orders.keys()); ba = min(d.sell_orders.keys())
    bv = d.buy_orders[bb]; av = -d.sell_orders[ba]
    tot = bv + av
    if tot <= 0: return 0.5 * (bb + ba)
    return (bb * av + ba * bv) / tot


def _smile_depth_ok(d: OrderDepth, bb: int, ba: int, min_depth: int, max_spread: int) -> bool:
    """Skip strikes with a wide or empty book so wing IVs don't pollute the parabolic fit."""
    if ba - bb > max_spread:
        return False
    bv = d.buy_orders.get(bb, 0)
    av = -d.sell_orders.get(ba, 0)
    return bv >= min_depth and av >= min_depth


def load_mark_alpha_config() -> Dict:
    """
    For **local** dev only: if `mark_alpha.json` exists beside this file or
    in CWD, merge it into a copy of `DEFAULT_MARK_ALPHA`. On a single-file
    IMC upload, no file is present; this returns a deep copy of
    `DEFAULT_MARK_ALPHA` unchanged. Keep production numbers in
    `DEFAULT_MARK_ALPHA` in the source you submit.
    """
    paths: List[str] = []
    try:
        paths.append(str(Path(__file__).resolve().parent / "mark_alpha.json"))
    except NameError:
        pass
    paths.append(str(Path.cwd() / "mark_alpha.json"))

    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        merged = copy.deepcopy(DEFAULT_MARK_ALPHA)
        for prod, marks in data.items():
            if not isinstance(marks, dict):
                continue
            if prod not in merged:
                merged[prod] = {}
            for mk, sides in marks.items():
                if isinstance(sides, dict):
                    merged[prod][mk] = sides
        return merged
    return copy.deepcopy(DEFAULT_MARK_ALPHA)


# ---------------------------------------------------------------------------
# Counterparty signal tracker
# ---------------------------------------------------------------------------
class CounterpartySignal:
    """
    Maintains a per-product alpha estimate driven by recent Mark trades.
    Each tick, decay existing signal, then absorb any newly-arrived Mark
    trades whose buyer/seller is in our alpha map.
    """

    def __init__(self, mark_alpha_map: Dict):
        self.mark_alpha = mark_alpha_map
        self.signal: Dict[str, float] = {}
        self.last_ts: Dict[str, int] = {}
        self.processed_trades: set = set()  # dedup across ticks

    def _decay(self, prod: str, current_ts: int) -> None:
        if prod not in self.signal:
            return
        last = self.last_ts.get(prod, current_ts)
        elapsed = current_ts - last
        if elapsed <= 0:
            return
        decay = math.exp(-elapsed / SIGNAL_HALF_LIFE_TICKS)
        self.signal[prod] = self.signal[prod] * decay
        self.last_ts[prod] = current_ts

    def update(self, market_trades: Dict[str, List[Trade]], current_ts: int) -> None:
        # Decay all signals to current ts first.
        for prod in list(self.signal.keys()):
            self._decay(prod, current_ts)

        # Absorb new trades.
        for prod, mtrades in market_trades.items():
            if prod not in self.mark_alpha:
                continue
            alpha_map = self.mark_alpha[prod]
            for tr in mtrades:
                key = (prod, tr.timestamp, tr.buyer, tr.seller, tr.price, tr.quantity)
                if key in self.processed_trades:
                    continue
                self.processed_trades.add(key)

                # Look up alpha for this Mark+side
                cand = []
                if tr.buyer and tr.buyer in alpha_map and "BUY" in alpha_map[tr.buyer]:
                    cand.append(alpha_map[tr.buyer]["BUY"]["alpha_1k"])
                if tr.seller and tr.seller in alpha_map and "SELL" in alpha_map[tr.seller]:
                    cand.append(alpha_map[tr.seller]["SELL"]["alpha_1k"])
                if not cand:
                    continue

                # Take the maximum-magnitude alpha; combine with existing signal
                # (keeping the larger absolute value).
                new_alpha = SIGNAL_INITIAL_FRAC * max(cand, key=abs)
                cur = self.signal.get(prod, 0.0)
                if abs(new_alpha) > abs(cur):
                    self.signal[prod] = new_alpha
                else:
                    # Don't overwrite a still-strong signal with a weaker fresh one.
                    self.signal[prod] = cur

                self.last_ts[prod] = current_ts

        # Cap memory on processed_trades to avoid unbounded growth.
        if len(self.processed_trades) > 5000:
            # keep only most recent 2000
            self.processed_trades = set(list(self.processed_trades)[-2000:])

    def get(self, prod: str) -> float:
        return self.signal.get(prod, 0.0)


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------
class Trader:
    def __init__(self) -> None:
        self.hydrogel_ema: Optional[float] = None
        self.last_extract_mid: Optional[float] = None
        self.start_tte_days: float = ROUND4_START_TTE_DAYS

        # Offline scores from counterparty_analysis.py; optional JSON merge.
        self.mark_alpha_cfg = load_mark_alpha_config()
        self.signal = CounterpartySignal(self.mark_alpha_cfg)

    # ---------------- HYDROGEL fair value (R4: + imbalance tilt) ----------
    def _fair_hydrogel(self, depth: OrderDepth) -> Optional[float]:
        bb, ba = _top_of_book(depth)
        mid = _mid(bb, ba)
        if mid is None: return None
        mp = _microprice(depth)
        x = mp if mp is not None else mid
        cfg = PARAMS[HYDROGEL]
        w = float(cfg.get("anchor_blend", 0.3))
        if self.hydrogel_ema is None:
            # Pure-EMA mode: seed from first print so we are not stuck starting at 10000.
            self.hydrogel_ema = float(x) if w <= 0.0 else float(cfg["anchor"])
        a = cfg["ema_alpha"]
        self.hydrogel_ema = a * x + (1 - a) * self.hydrogel_ema
        base = (1.0 - w) * self.hydrogel_ema + w * float(cfg["anchor"])

        # Imbalance tilt: add `imb_scale * imb` to fair when |imb| exceeds threshold.
        # Backtested R4 signal: |imb| > 0.2 -> ~$3 mid move in next 10 ticks.
        if depth.buy_orders and depth.sell_orders:
            bv = depth.buy_orders[bb]
            av = -depth.sell_orders[ba]
            tot = bv + av
            if tot > 0:
                imb = (bv - av) / tot
                thr = cfg.get("imb_threshold", 0.20)
                scale = cfg.get("imb_scale", 0.0)
                if abs(imb) >= thr and scale > 0:
                    base += scale * imb
        return base

    # ---------------- Generic delta-1 MM (HYDROGEL) -----------------------
    def _trade_delta1(self, product: str, depth: OrderDepth, fair: float,
                      cfg: dict, position: int) -> List[Order]:
        orders: List[Order] = []
        limit = LIMIT[product]
        soft_cap = cfg["soft_cap"]
        max_size = cfg["max_quote_size"]
        offset = cfg["quote_offset"]
        buy_used = sell_used = 0

        if depth.sell_orders:
            for ask, sz in sorted(depth.sell_orders.items()):
                if ask > fair - cfg["take_edge"]: break
                cap = limit - (position + buy_used)
                if cap <= 0: break
                qty = min(-sz, cap)
                if qty > 0:
                    orders.append(Order(product, ask, qty)); buy_used += qty
        if depth.buy_orders:
            for bid, sz in sorted(depth.buy_orders.items(), reverse=True):
                if bid < fair + cfg["take_edge"]: break
                cap = limit + (position - sell_used)
                if cap <= 0: break
                qty = min(sz, cap)
                if qty > 0:
                    orders.append(Order(product, bid, -qty)); sell_used += qty

        net = position + buy_used - sell_used
        if net > 0:
            target = int(round(fair + cfg["clear_edge"]))
            interest = sum(v for p, v in depth.buy_orders.items() if p >= target)
            cap = limit + (position - sell_used)
            qty = min(net, interest, cap)
            if qty > 0:
                orders.append(Order(product, target, -qty)); sell_used += qty
        elif net < 0:
            target = int(round(fair - cfg["clear_edge"]))
            interest = sum(-v for p, v in depth.sell_orders.items() if p <= target)
            cap = limit - (position + buy_used)
            qty = min(-net, interest, cap)
            if qty > 0:
                orders.append(Order(product, target, qty)); buy_used += qty

        bb, ba = _top_of_book(depth)
        if bb is not None and ba is not None:
            fair_int = int(round(fair))
            bid_px = min(bb + 1, fair_int - offset)
            ask_px = max(ba - 1, fair_int + offset)
            if bid_px >= ask_px:
                bid_px = fair_int - offset; ask_px = fair_int + offset

            buy_cap = limit - (position + buy_used)
            sell_cap = limit + (position - sell_used)
            buy_size = max(0, min(max_size, buy_cap))
            sell_size = max(0, min(max_size, sell_cap))

            if position > soft_cap: buy_size = 0
            elif position > soft_cap // 2: buy_size = min(buy_size, max_size // 2)
            if position < -soft_cap: sell_size = 0
            elif position < -soft_cap // 2: sell_size = min(sell_size, max_size // 2)

            # Skew quote sizes against imbalance (adverse selection). Either
            # always on (`imb_quote_skew`) or gated: wide spread + extreme |imb|.
            imb_skew_active = bool(cfg.get("imb_quote_skew", False))
            if not imb_skew_active and cfg.get("imb_gated_skew", False):
                gate_spread = int(cfg.get("imb_gate_min_spread", 4))
                gate_ext = float(cfg.get("imb_gate_extreme", 0.28))
                if (ba - bb) >= gate_spread and depth.buy_orders and depth.sell_orders:
                    bv0 = depth.buy_orders[bb]
                    av0 = -depth.sell_orders[ba]
                    tot0 = bv0 + av0
                    if tot0 > 0 and abs((bv0 - av0) / tot0) >= gate_ext:
                        imb_skew_active = True

            if imb_skew_active and depth.buy_orders and depth.sell_orders:
                bv = depth.buy_orders[bb]
                av = -depth.sell_orders[ba]
                tot = bv + av
                if tot > 0:
                    imb = (bv - av) / tot
                    thr = cfg.get("imb_threshold", 0.20)
                    if imb >= thr:
                        sell_size = 0      # don't sell into a rising market
                    elif imb <= -thr:
                        buy_size = 0       # don't buy into a falling market

            if buy_size > 0 and bid_px > 0:
                orders.append(Order(product, bid_px, buy_size))
            if sell_size > 0 and ask_px > 0:
                orders.append(Order(product, ask_px, -sell_size))

        return orders

    # ---------------- Passive Mark-avoidance MM on EXTRACT ----------------
    def _trade_extract_mm(self, depth: OrderDepth, position: int) -> List[Order]:
        """
        Passive MM on EXTRACT with informed-Mark avoidance:
          * Quote 1 inside the BBO on each side, small size.
          * If Mark 67 is currently bullish (recent BUY), DROP our ask.
          * If Mark 22/49 is currently bearish (recent SELL), DROP our bid.
          * Take asks/bids only if extremely off (>3.5 from microprice).
          * Inventory clears at mid via small orders.

        Avoids the "execution cost > alpha" trap of trading WITH signals,
        and avoids adverse selection by withdrawing quotes when an informed
        counterparty is active.
        """
        cfg = PARAMS[EXTRACT]
        if not cfg.get("enable_passive_mm", False):
            return []
        bb, ba = _top_of_book(depth)
        mid = _mid(bb, ba)
        if mid is None or bb is None or ba is None:
            return []
        mp = _microprice(depth)
        x = mp if mp is not None else mid

        signal = self.signal.get(EXTRACT)
        avoid_buy = signal < -0.5    # Mark 22/49 currently selling -> price falling -> don't buy
        avoid_sell = signal > +0.5   # Mark 67 currently buying -> price rising -> don't sell

        orders: List[Order] = []
        soft_cap = cfg["mm_soft_cap"]
        max_q = cfg["mm_quote_size"]
        take_edge = cfg["mm_take_edge"]

        buy_used = sell_used = 0

        # 1) Take obvious mispricings (rare but free money).
        if not avoid_buy and depth.sell_orders:
            for ask, sz in sorted(depth.sell_orders.items()):
                if ask > x - take_edge:
                    break
                cap = LIMIT[EXTRACT] - (position + buy_used)
                if cap <= 0: break
                qty = min(-sz, cap, max_q)
                if qty > 0:
                    orders.append(Order(EXTRACT, ask, qty)); buy_used += qty
        if not avoid_sell and depth.buy_orders:
            for bid, sz in sorted(depth.buy_orders.items(), reverse=True):
                if bid < x + take_edge:
                    break
                cap = LIMIT[EXTRACT] + (position - sell_used)
                if cap <= 0: break
                qty = min(sz, cap, max_q)
                if qty > 0:
                    orders.append(Order(EXTRACT, bid, -qty)); sell_used += qty

        # 2) Passive MM 1 inside the BBO. Skip the side that informed Mark trades.
        bid_px = bb + 1
        ask_px = ba - 1
        if bid_px >= ask_px:
            # spread too tight; sit at BBO
            bid_px = bb
            ask_px = ba

        net = position + buy_used - sell_used
        buy_cap = LIMIT[EXTRACT] - (position + buy_used)
        sell_cap = LIMIT[EXTRACT] + (position - sell_used)
        buy_size = max(0, min(max_q, buy_cap))
        sell_size = max(0, min(max_q, sell_cap))

        if net >= soft_cap:
            buy_size = 0
        if net <= -soft_cap:
            sell_size = 0
        if avoid_buy:
            buy_size = 0
        if avoid_sell:
            sell_size = 0

        if buy_size > 0 and bid_px > 0:
            orders.append(Order(EXTRACT, bid_px, buy_size))
        if sell_size > 0 and ask_px > 0:
            orders.append(Order(EXTRACT, ask_px, -sell_size))

        return orders

    # ---------------- Counterparty-driven EXTRACT trader -----------------
    def _trade_extract(self, depth: OrderDepth, position: int) -> List[Order]:
        """
        Counterparty-signal-aware EXTRACT trading.

        Key insight from backtest debug: crossing the spread (~$5 wide) to
        take a Mark 67 BUY signal of +$2.24 alpha is *negative EV* because
        execution cost > alpha. We must trade only *passively*:
          * If signal > 0 (bullish): post a passive BUY at bb+1 (or BUY at
            best-ask only if it's below mid + signal/2, ie. we skip the
            half-spread cost).
          * If signal < 0 (bearish): symmetric.
          * If signal magnitude is small, we don't quote at all.

        This converts the alpha into a "free option" - we only fill when
        someone else crosses to us, capturing close-to-the-spread gain.
        """
        cfg = PARAMS[EXTRACT]
        signal = self.signal.get(EXTRACT)            # +ve = bullish, -ve = bearish
        if abs(signal) < 0.5:
            return []  # signal too weak to act on

        bb, ba = _top_of_book(depth)
        mid = _mid(bb, ba)
        if mid is None or bb is None or ba is None:
            return []
        mp = _microprice(depth)
        x = mp if mp is not None else mid
        fair = x + cfg["signal_lean_fair"] * signal

        orders: List[Order] = []
        max_pos = cfg["max_position_per_signal"]

        # Target position scaled by signal strength.
        target = int(round(max_pos * max(-1.0, min(1.0, signal / 2.0))))
        delta_to_target = target - position
        if delta_to_target == 0:
            return []

        if delta_to_target > 0:
            # Bullish - want long. ONLY take asks that are below fair (i.e.
            # already cheap given the signal); otherwise post a passive bid.
            if depth.sell_orders:
                for ask, sz in sorted(depth.sell_orders.items()):
                    if ask >= fair:    # require ask below our shifted fair
                        break
                    qty = min(-sz, delta_to_target)
                    if qty > 0:
                        orders.append(Order(EXTRACT, ask, qty))
                        delta_to_target -= qty
                    if delta_to_target <= 0:
                        break
            if delta_to_target > 0:
                bid_px = bb + 1 if bb + 1 < ba else bb
                qty = min(delta_to_target, 25)
                if qty > 0:
                    orders.append(Order(EXTRACT, bid_px, qty))

        else:
            # Bearish - want short.
            if depth.buy_orders:
                for bid, sz in sorted(depth.buy_orders.items(), reverse=True):
                    if bid <= fair:
                        break
                    qty = min(sz, -delta_to_target)
                    if qty > 0:
                        orders.append(Order(EXTRACT, bid, -qty))
                        delta_to_target += qty
                    if delta_to_target >= 0:
                        break
            if delta_to_target < 0:
                ask_px = ba - 1 if ba - 1 > bb else ba
                qty = min(-delta_to_target, 25)
                if qty > 0:
                    orders.append(Order(EXTRACT, ask_px, -qty))

        return orders

    # ---------------- Voucher trading (R3 logic, unchanged for MM legs) ---
    def _trade_voucher(self, product: str, depth: OrderDepth, theo: float,
                       cfg: dict, position: int,
                       conviction_long: float = 1.0,
                       conviction_short: float = 1.0) -> List[Order]:
        if not (cfg.get("make", True) or cfg["take_edge"] < 100):
            return []

        orders: List[Order] = []
        limit = LIMIT[product]
        soft_cap_base = cfg["soft_cap"]
        hard_cap_base = cfg.get("hard_cap", limit)
        max_long_total = max(0, int(min(limit, hard_cap_base) * conviction_long))
        max_short_total = max(0, int(min(limit, hard_cap_base) * conviction_short))
        soft_cap_long = max(1, int(soft_cap_base * conviction_long))
        soft_cap_short = max(1, int(soft_cap_base * conviction_short))

        buy_used = sell_used = 0

        # 1) Take cheap asks
        if depth.sell_orders:
            for ask, sz in sorted(depth.sell_orders.items()):
                if ask > theo - cfg["take_edge"]: break
                cap = max_long_total - (position + buy_used)
                if cap <= 0: break
                qty = min(-sz, cap)
                if qty > 0:
                    orders.append(Order(product, ask, qty)); buy_used += qty
        # 2) Hit rich bids
        if depth.buy_orders:
            for bid, sz in sorted(depth.buy_orders.items(), reverse=True):
                if bid < theo + cfg["take_edge"]: break
                cap = max_short_total + (position - sell_used)
                if cap <= 0: break
                qty = min(sz, cap)
                if qty > 0:
                    orders.append(Order(product, bid, -qty)); sell_used += qty
        # 3) Clear inventory
        net = position + buy_used - sell_used
        if net > 0:
            target_px = max(1, int(math.ceil(theo + cfg["clear_edge"])))
            interest = sum(v for p, v in depth.buy_orders.items() if p >= target_px)
            cap = limit + (position - sell_used)
            qty = min(net, interest, cap)
            if qty > 0:
                orders.append(Order(product, target_px, -qty)); sell_used += qty
        elif net < 0:
            target_px = max(1, int(math.floor(theo - cfg["clear_edge"])))
            interest = sum(-v for p, v in depth.sell_orders.items() if p <= target_px)
            cap = limit - (position + buy_used)
            qty = min(-net, interest, cap)
            if qty > 0:
                orders.append(Order(product, target_px, qty)); buy_used += qty
        # 4) Make
        if cfg.get("make", True):
            bb, ba = _top_of_book(depth)
            if bb is not None and ba is not None:
                bid_px = max(1, int(math.floor(theo - 1)))
                ask_px = max(bid_px + 1, int(math.ceil(theo + 1)))
                bid_px = min(bid_px, bb + 1) if bb + 1 < ask_px else bid_px
                ask_px = max(ask_px, ba - 1) if ba - 1 > bid_px else ask_px
                if bid_px >= ask_px:
                    bid_px = max(1, int(math.floor(theo - 1)))
                    ask_px = max(bid_px + 1, int(math.ceil(theo + 1)))
                max_q = cfg["max_quote"]
                buy_cap = max_long_total - (position + buy_used)
                sell_cap = max_short_total + (position - sell_used)
                buy_size = max(0, min(max_q, buy_cap))
                sell_size = max(0, min(max_q, sell_cap))
                if position >= soft_cap_long: buy_size = 0
                elif position >= soft_cap_long // 2: buy_size = min(buy_size, max(1, max_q // 3))
                if position <= -soft_cap_short: sell_size = 0
                elif position <= -soft_cap_short // 2: sell_size = min(sell_size, max(1, max_q // 3))
                if buy_size > 0 and bid_px > 0:
                    orders.append(Order(product, bid_px, buy_size))
                if sell_size > 0 and ask_px > 0:
                    orders.append(Order(product, ask_px, -sell_size))
        return orders

    # ---------------- Gamma scalp -----------------------------------------
    def _gamma_scalp_orders(self, state: TradingState, spot: float,
                            T_days: float, positions: Dict[str, int]) -> List[Order]:
        """
        Maintain target long positions on a basket of vouchers.  Returns
        orders to move toward the target per-strike position; each strike is
        traded only via passive quotes 1 tick inside the spread.
        """
        if not GAMMA_SCALP_ENABLED:
            return []
        out: List[Order] = []
        for K, target in GAMMA_TARGET_PER_STRIKE.items():
            sym = f"VEV_{K}"
            if sym not in state.order_depths:
                continue
            depth = state.order_depths[sym]
            bb, ba = _top_of_book(depth)
            if bb is None or ba is None: continue
            cur = positions.get(sym, 0)
            need = target - cur
            if need == 0: continue
            if need > 0:
                # buy passively at bb+1 (or theo-1, whichever lower)
                theo = voucher_theo(spot, K, T_days)
                buy_px = min(bb + 1, max(1, int(math.floor(theo - 0.5))))
                size = min(need, 10)
                if size > 0 and buy_px > 0:
                    out.append(Order(sym, buy_px, size))
            else:
                theo = voucher_theo(spot, K, T_days)
                sell_px = max(ba - 1, int(math.ceil(theo + 0.5)))
                size = min(-need, 10)
                if size > 0:
                    out.append(Order(sym, sell_px, -size))
        return out

    def _delta_hedge_orders(self, state: TradingState, spot: float,
                            T_days: float, positions: Dict[str, int],
                            extract_signal_orders_qty: int) -> List[Order]:
        """
        Compute net portfolio delta from voucher positions and return EXTRACT
        orders to neutralize (subject to extract limit + DELTA_HEDGE_CAP).

        `extract_signal_orders_qty` is the *signed* quantity already committed
        to EXTRACT this tick by the counterparty-signal path; we don't double-
        spend that capacity here.
        """
        # Net delta from voucher positions (BS deltas)
        delta = 0.0
        for sym in VEVS:
            pos = positions.get(sym, 0)
            if pos == 0: continue
            K = STRIKE_OF[sym]
            d = voucher_delta(spot, K, T_days)
            delta += d * pos
        # We want EXTRACT position == -delta (clamped to limit & hedge cap)
        target_extract = -delta
        target_extract = max(-DELTA_HEDGE_CAP, min(DELTA_HEDGE_CAP, target_extract))

        cur_extract = positions.get(EXTRACT, 0) + extract_signal_orders_qty
        gap = target_extract - cur_extract
        if abs(gap) < 5:
            return []

        depth = state.order_depths.get(EXTRACT)
        if depth is None: return []
        bb, ba = _top_of_book(depth)
        if bb is None or ba is None: return []

        out: List[Order] = []
        if gap > 0:
            # Buy EXTRACT passively
            buy_px = bb + 1 if bb + 1 < ba else bb
            size = int(min(gap, 30))
            if size > 0:
                out.append(Order(EXTRACT, buy_px, size))
        else:
            sell_px = ba - 1 if ba - 1 > bb else ba
            size = int(min(-gap, 30))
            if size > 0:
                out.append(Order(EXTRACT, sell_px, -size))
        return out

    # ---------------- Main entry ------------------------------------------
    def run(self, state: TradingState):
        # Restore persistent state.
        if state.traderData:
            try:
                saved = jsonpickle.decode(state.traderData)
                self.hydrogel_ema = saved.get("hydrogel_ema", None)
                self.last_extract_mid = saved.get("last_extract_mid", None)
                # Restore signal state too
                if "signal_state" in saved:
                    s = saved["signal_state"]
                    self.signal.signal = s.get("signal", {})
                    self.signal.last_ts = s.get("last_ts", {})
            except Exception:
                pass

        result: Dict[str, List[Order]] = {}
        positions = state.position
        ts = state.timestamp

        # Update counterparty signals from this tick's market trades.
        self.signal.update(state.market_trades, ts)

        # ----- Underlying spot (we need it for vouchers) -----
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
            spot = self.last_extract_mid if self.last_extract_mid is not None else 5250.0

        T_days = max(MIN_T_DAYS, self.start_tte_days - ts / TIMESTAMPS_PER_DAY)

        # ----- Vouchers (smile-based, R3 winning config) -----
        # Live smile fit
        smile_a = SMILE_A; smile_b = 0.0; smile_c = SMILE_C
        live_iv: Dict[int, float] = {}
        live_m: Dict[int, float] = {}
        smile_K_set = SMILE_FIT_STRIKES
        if T_days > 0 and spot > 0:
            pts: List[Tuple[float, float]] = []
            for K in smile_K_set:
                v = f"VEV_{K}"
                d_v = state.order_depths.get(v)
                if d_v is None: continue
                bb_v, ba_v = _top_of_book(d_v)
                if bb_v is None or ba_v is None: continue
                if not _smile_depth_ok(d_v, bb_v, ba_v, SMILE_FIT_MIN_DEPTH, SMILE_FIT_MAX_SPREAD):
                    continue
                vmid = _mid(bb_v, ba_v)
                if vmid is None or vmid <= 0: continue
                iv = implied_vol(vmid, spot, float(K), T_days)
                if iv <= 0: continue
                m = math.log(K / spot) / math.sqrt(T_days)
                live_iv[K] = iv; live_m[K] = m
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

        def _theo_for(K: int) -> float:
            return _live_voucher_theo(K) if PRICING_MODE in ("smile","hybrid") else voucher_theo(spot, K, T_days)

        for v in VEVS:
            depth = state.order_depths.get(v)
            if depth is None: continue
            K = STRIKE_OF[v]
            cfg = VEV_PARAMS[v]
            theo = _theo_for(K)
            pos = positions.get(v, 0)

            # Conviction from smile residual (R3 logic preserved)
            conv_long = conv_short = 1.0
            if K in smile_K_set and K in live_iv:
                iv_market = live_iv[K]
                iv_smile = _live_smile_sigma(K)
                z = (iv_smile - iv_market) / max(SMILE_RESIDUAL_STD, 1e-6)
                strength = max(0.0, min(1.0, abs(z) / 3.0))
                if z > 0:
                    conv_long = max(0.2, strength); conv_short = max(0.2, 1.0 - strength)
                elif z < 0:
                    conv_short = max(0.2, strength); conv_long = max(0.2, 1.0 - strength)

            voucher_orders = self._trade_voucher(v, depth, theo, cfg, pos, conv_long, conv_short)
            if voucher_orders:
                result.setdefault(v, []).extend(voucher_orders)

        # ----- Gamma scalp (target longs on disabled strikes) -----
        if GAMMA_SCALP_ENABLED:
            gamma_orders = self._gamma_scalp_orders(state, spot, T_days, positions)
            for o in gamma_orders:
                result.setdefault(o.symbol, []).append(o)

        # ----- EXTRACT counterparty-signal trading (legacy; default off) -----
        ext_signal_qty = 0
        if extract_depth is not None and PARAMS[EXTRACT].get("enable_signal_trades", False):
            ext_orders = self._trade_extract(extract_depth, positions.get(EXTRACT, 0))
            for o in ext_orders:
                ext_signal_qty += o.quantity
                result.setdefault(EXTRACT, []).append(o)

        # ----- Passive EXTRACT MM with Mark-avoidance (Orin's hint) -----
        if extract_depth is not None and PARAMS[EXTRACT].get("enable_passive_mm", False):
            mm_orders = self._trade_extract_mm(extract_depth, positions.get(EXTRACT, 0))
            for o in mm_orders:
                ext_signal_qty += o.quantity
                result.setdefault(EXTRACT, []).append(o)

        # ----- EXTRACT delta hedge (only if needed AFTER signal trades) -----
        if GAMMA_SCALP_ENABLED:
            hedge_orders = self._delta_hedge_orders(state, spot, T_days, positions, ext_signal_qty)
            for o in hedge_orders:
                result.setdefault(EXTRACT, []).append(o)

        # ----- HYDROGEL_PACK (R3 winner) -----
        hydro_depth = state.order_depths.get(HYDROGEL)
        if hydro_depth is not None:
            fair = self._fair_hydrogel(hydro_depth)
            if fair is not None:
                hpos = positions.get(HYDROGEL, 0)
                kappa = float(PARAMS[HYDROGEL].get("inv_kappa", 0.0))
                fair = fair - kappa * hpos
                hyd_orders = self._trade_delta1(HYDROGEL, hydro_depth, fair,
                                                 PARAMS[HYDROGEL], hpos)
                if hyd_orders:
                    result.setdefault(HYDROGEL, []).extend(hyd_orders)

        # ----- Persist state -----
        trader_data = jsonpickle.encode({
            "hydrogel_ema": self.hydrogel_ema,
            "last_extract_mid": self.last_extract_mid,
            "signal_state": {
                "signal": self.signal.signal,
                "last_ts": self.signal.last_ts,
            },
        })

        return result, 0, trader_data

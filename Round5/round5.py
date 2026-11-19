from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math


LIMIT = 10
RET_SCALE = 2.0
MAX_RET_HISTORY = 400
ENABLE_RV_MODELS = False

MR_ALIAS = {
    "ROBOT_DISHES": "D",
    "ROBOT_IRONING": "I",
    "OXYGEN_SHAKE_CHOCOLATE": "C",
    "OXYGEN_SHAKE_EVENING_BREATH": "E",
    "UV_VISOR_ORANGE": "UO",
    "TRANSLATOR_ASTRO_BLACK": "TAB",
    "PEBBLES_S": "PS",
    "UV_VISOR_MAGENTA": "UM",
    "GALAXY_SOUNDS_SOLAR_WINDS": "GSW",
    "GALAXY_SOUNDS_SOLAR_FLAMES": "GSF",
    "PANEL_2X2": "P22",
}
ALIAS_TO_MR = {v: k for k, v in MR_ALIAS.items()}

PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
PEBBLE_TRADE_PRODUCTS = {"PEBBLES_XS", "PEBBLES_L", "PEBBLES_XL"}
SNACKS = [
    "SNACKPACK_VANILLA",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY",
]
SNACK_IMBALANCE_SCALE = 120.0

CATEGORY_SUM_MODELS = {
    # Learn the category sum from the opening prints, then trade deviations
    # against the other four legs. This adapts to the short evaluation window
    # without hard-coding a day-specific anchor.
    "MICRO": {
        "products": ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE"],
        "trade_products": ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE"],
        "warmup": 50,
        "edge": 200.0,
        "max_qty": 1,
    },
    "SLEEP": {
        "products": ["SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER", "SLEEP_POD_NYLON", "SLEEP_POD_COTTON"],
        "trade_products": ["SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER"],
        "warmup": 50,
        "edge": 300.0,
        "max_qty": 1,
    },
    "ROBOT": {
        "products": ["ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES", "ROBOT_LAUNDRY", "ROBOT_IRONING"],
        "trade_products": ["ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_LAUNDRY", "ROBOT_IRONING"],
        "warmup": 200,
        "edge": 100.0,
        "max_qty": 1,
    },
    "OXYGEN": {
        "products": ["OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH", "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC"],
        "trade_products": ["OXYGEN_SHAKE_EVENING_BREATH", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC"],
        "warmup": 200,
        "edge": 150.0,
        "max_qty": 1,
    },
}
CATEGORY_SUM_MODELS = {}
CATEGORY_SUM_PRODUCTS = {
    product
    for model in CATEGORY_SUM_MODELS.values()
    for product in model["products"]
}

MEAN_REVERSION = {
    # These are live regime detectors, not day-specific switches. The strategy
    # only trades after recent returns show sufficiently negative autocorrelation.
    "ROBOT_DISHES": {"k": 0.25, "max_qty": 1, "window": 400, "corr_threshold": -0.08},
    "ROBOT_IRONING": {"k": 0.25, "max_qty": 1, "window": 400, "corr_threshold": -0.12},
    "OXYGEN_SHAKE_CHOCOLATE": {"k": 0.25, "max_qty": 1, "window": 400, "corr_threshold": -0.12},
    "OXYGEN_SHAKE_EVENING_BREATH": {"k": 0.25, "max_qty": 1, "window": 400, "corr_threshold": -0.12},
}
MEAN_REVERSION = {}

SECONDARY_MEAN_REVERSION = {
    # Small non-overlapping add-ons. They end flat in replay and do not consume
    # capacity from the category-sum legs.
    "UV_VISOR_ORANGE": {"k": 0.25, "max_qty": 1},
    "TRANSLATOR_ASTRO_BLACK": {"k": 0.25, "max_qty": 1},
    "PEBBLES_S": {"k": 0.25, "max_qty": 1},
    "UV_VISOR_MAGENTA": {"k": 0.35, "max_qty": 1},
    "GALAXY_SOUNDS_SOLAR_WINDS": {"k": 0.25, "max_qty": 1},
    "GALAXY_SOUNDS_SOLAR_FLAMES": {"k": 0.25, "max_qty": 1},
    "PANEL_2X2": {"k": 0.25, "max_qty": 1},
}
SECONDARY_MEAN_REVERSION = {}

RV_MODELS = {
    # Fitted on days 2-4. SLEEP was the most stable out-of-sample category in
    # the simple crossing backtest, so it gets active relative-value trades.
    "SLEEP_POD_SUEDE": {
        "others": ["SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER", "SLEEP_POD_NYLON", "SLEEP_POD_COTTON"],
        "coef": [-2067.33760313, -0.01011782, 0.98380601, 0.18421800, 0.03205572],
        "sigma": 478.513827,
        "edge_mult": 4.0,
        "max_qty": 1,
    },
    "SLEEP_POD_LAMB_WOOL": {
        "others": ["SLEEP_POD_SUEDE", "SLEEP_POD_POLYESTER", "SLEEP_POD_NYLON", "SLEEP_POD_COTTON"],
        "coef": [6849.96985652, -0.00626498, -0.11786737, 0.47949813, 0.06236700],
        "sigma": 376.539664,
        "edge_mult": 6.0,
        "max_qty": 1,
    },
    "SLEEP_POD_POLYESTER": {
        "others": ["SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON", "SLEEP_POD_COTTON"],
        "coef": [4433.98073491, 0.46765625, -0.09048537, -0.39227442, 0.57756173],
        "sigma": 329.915809,
        "edge_mult": 4.0,
        "max_qty": 1,
    },
    "SLEEP_POD_NYLON": {
        "others": ["SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER", "SLEEP_POD_COTTON"],
        "coef": [7616.02520361, 0.05835479, 0.24530074, -0.26140696, 0.13295756],
        "sigma": 269.318822,
        "edge_mult": 6.0,
        "max_qty": 1,
    },
    "SLEEP_POD_COTTON": {
        "others": ["SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER", "SLEEP_POD_NYLON"],
        "coef": [2413.20892219, 0.01444139, 0.04537597, 0.54737463, 0.18909155],
        "sigma": 321.178343,
        "edge_mult": 4.0,
        "max_qty": 1,
    },
    # Snacks are noisier than sleep pods. High edge multiplier keeps this to
    # rare extremes only.
    "SNACKPACK_CHOCOLATE": {
        "others": ["SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY"],
        "coef": [19500.95713546, -0.92208396, 0.11201971, -0.14232698, 0.01122195],
        "sigma": 45.615569,
        "edge_mult": 5.0,
        "max_qty": 1,
    },
    "SNACKPACK_VANILLA": {
        "others": ["SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY"],
        "coef": [20483.25471028, -0.99691686, 0.08894212, -0.13700221, 0.00489387],
        "sigma": 47.430460,
        "edge_mult": 5.0,
        "max_qty": 1,
    },
    "SNACKPACK_PISTACHIO": {
        "others": ["SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY"],
        "coef": [13853.26654776, 0.43521960, 0.31961932, -0.30672563, -0.85184686],
        "sigma": 89.912507,
        "edge_mult": 5.0,
        "max_qty": 1,
    },
    "SNACKPACK_RASPBERRY": {
        "others": ["SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO", "SNACKPACK_STRAWBERRY"],
        "coef": [20719.31264794, 0.03957740, 0.01596403, -0.77326131, -0.35954434],
        "sigma": 85.664816,
        "edge_mult": 5.0,
        "max_qty": 1,
    },
}


def top_of_book(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    best_bid = max(depth.buy_orders) if depth.buy_orders else None
    best_ask = min(depth.sell_orders) if depth.sell_orders else None
    return best_bid, best_ask


def mid_price(depth: OrderDepth) -> Optional[float]:
    best_bid, best_ask = top_of_book(depth)
    if best_bid is None or best_ask is None:
        return None
    return (best_bid + best_ask) / 2.0


def rolling_autocorr(returns: List[int]) -> float:
    n = len(returns) - 1
    if n < 20:
        return 0.0
    xs = returns[:-1]
    ys = returns[1:]
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    syy = sum(y * y for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    if vx <= 1e-9 or vy <= 1e-9:
        return 0.0
    return (sxy - sx * sy / n) / math.sqrt(vx * vy)


class Trader:
    def _add_order(
        self,
        result: Dict[str, List[Order]],
        working_pos: Dict[str, int],
        product: str,
        price: int,
        quantity: int,
    ) -> None:
        if price <= 0 or quantity == 0:
            return
        pos = working_pos.get(product, 0)
        if quantity > 0:
            quantity = min(quantity, LIMIT - pos)
        else:
            quantity = -min(-quantity, LIMIT + pos)
        if quantity == 0:
            return
        result.setdefault(product, []).append(Order(product, int(price), int(quantity)))
        working_pos[product] = pos + quantity

    def _take_against_fair(
        self,
        result: Dict[str, List[Order]],
        working_pos: Dict[str, int],
        product: str,
        depth: OrderDepth,
        fair: float,
        edge: float,
        max_qty: int,
    ) -> None:
        for ask, volume in sorted(depth.sell_orders.items()):
            if ask > fair - edge:
                break
            qty = min(-volume, max_qty)
            self._add_order(result, working_pos, product, ask, qty)
            if working_pos.get(product, 0) >= LIMIT:
                break

        for bid, volume in sorted(depth.buy_orders.items(), reverse=True):
            if bid < fair + edge:
                break
            qty = min(volume, max_qty)
            self._add_order(result, working_pos, product, bid, -qty)
            if working_pos.get(product, 0) <= -LIMIT:
                break

    def _flatten_near_fair(
        self,
        result: Dict[str, List[Order]],
        working_pos: Dict[str, int],
        product: str,
        depth: OrderDepth,
        fair: float,
        max_qty: int,
    ) -> None:
        best_bid, best_ask = top_of_book(depth)
        pos = working_pos.get(product, 0)
        if pos > 0 and best_bid is not None and best_bid >= fair:
            self._add_order(result, working_pos, product, best_bid, -min(pos, max_qty))
        elif pos < 0 and best_ask is not None and best_ask <= fair:
            self._add_order(result, working_pos, product, best_ask, min(-pos, max_qty))

    def _quote_around_fair(
        self,
        result: Dict[str, List[Order]],
        working_pos: Dict[str, int],
        product: str,
        depth: OrderDepth,
        fair: float,
        quote_edge: float,
        size: int,
    ) -> None:
        best_bid, best_ask = top_of_book(depth)
        if best_bid is None or best_ask is None:
            return

        bid = min(best_bid + 1, int(math.floor(fair - quote_edge)))
        ask = max(best_ask - 1, int(math.ceil(fair + quote_edge)))
        if bid >= ask:
            bid = int(math.floor(fair - quote_edge))
            ask = int(math.ceil(fair + quote_edge))
        if bid >= ask:
            return

        pos = working_pos.get(product, 0)
        buy_size = size if pos < 6 else 0
        sell_size = size if pos > -6 else 0
        self._add_order(result, working_pos, product, bid, buy_size)
        self._add_order(result, working_pos, product, ask, -sell_size)

    def _trade_relative_value(
        self,
        result: Dict[str, List[Order]],
        working_pos: Dict[str, int],
        mids: Dict[str, float],
        product: str,
        depth: OrderDepth,
    ) -> None:
        cfg = RV_MODELS[product]
        if any(other not in mids for other in cfg["others"]):
            return
        coef = cfg["coef"]
        fair = coef[0]
        for i, other in enumerate(cfg["others"]):
            fair += coef[i + 1] * mids[other]
        best_bid, best_ask = top_of_book(depth)
        if best_bid is None or best_ask is None:
            return
        spread = best_ask - best_bid
        edge = max(0.5 * spread, cfg["edge_mult"] * cfg["sigma"])
        max_qty = int(cfg["max_qty"])
        self._take_against_fair(result, working_pos, product, depth, fair, edge, max_qty)
        self._flatten_near_fair(result, working_pos, product, depth, fair, max_qty)

    def _trade_pebbles(
        self,
        result: Dict[str, List[Order]],
        working_pos: Dict[str, int],
        mids: Dict[str, float],
        product: str,
        depth: OrderDepth,
    ) -> None:
        if product not in PEBBLE_TRADE_PRODUCTS:
            return
        if any(other not in mids for other in PEBBLES if other != product):
            return
        fair = 50000.0 - sum(mids[other] for other in PEBBLES if other != product)
        best_bid, best_ask = top_of_book(depth)
        if best_bid is None or best_ask is None:
            return
        spread = best_ask - best_bid
        self._take_against_fair(result, working_pos, product, depth, fair, max(12.0, 0.75 * spread), 1)
        self._flatten_near_fair(result, working_pos, product, depth, fair, 1)

    def _trade_snack_imbalance(
        self,
        result: Dict[str, List[Order]],
        working_pos: Dict[str, int],
        product: str,
        depth: OrderDepth,
    ) -> None:
        current_mid = mid_price(depth)
        best_bid, best_ask = top_of_book(depth)
        if current_mid is None or best_bid is None or best_ask is None:
            return
        bid_volume = depth.buy_orders[best_bid]
        ask_volume = -depth.sell_orders[best_ask]
        total_volume = bid_volume + ask_volume
        if total_volume <= 0:
            return
        imbalance = (bid_volume - ask_volume) / total_volume
        fair = current_mid + SNACK_IMBALANCE_SCALE * imbalance
        edge = max(0.5 * (best_ask - best_bid), 1.0)
        self._take_against_fair(result, working_pos, product, depth, fair, edge, 1)
        self._flatten_near_fair(result, working_pos, product, depth, fair, 1)

    def _trade_category_sum(
        self,
        result: Dict[str, List[Order]],
        working_pos: Dict[str, int],
        mids: Dict[str, float],
        category_state: Dict[str, dict],
        model_name: str,
    ) -> None:
        cfg = CATEGORY_SUM_MODELS[model_name]
        products = cfg["products"]
        if any(product not in mids for product in products):
            return

        state = category_state.setdefault(model_name, {"n": 0, "sum": 0.0})
        current_sum = sum(mids[product] for product in products)
        n = int(state.get("n", 0)) + 1
        if n <= int(cfg["warmup"]):
            state["n"] = n
            state["sum"] = float(state.get("sum", 0.0)) + current_sum
            state["target"] = state["sum"] / n
            return
        state["n"] = n
        target_sum = float(state.get("target", current_sum))
        edge = float(cfg["edge"])
        max_qty = int(cfg["max_qty"])

        for product in cfg.get("trade_products", products):
            depth = self._current_depths.get(product)
            if depth is None:
                continue
            fair = target_sum - sum(mids[other] for other in products if other != product)
            self._take_against_fair(result, working_pos, product, depth, fair, edge, max_qty)
            self._flatten_near_fair(result, working_pos, product, depth, fair, max_qty)

    def _trade_mean_reversion(
        self,
        result: Dict[str, List[Order]],
        working_pos: Dict[str, int],
        last_mid: Dict[str, float],
        recent_returns: Dict[str, List[int]],
        product: str,
        depth: OrderDepth,
    ) -> None:
        current_mid = mid_price(depth)
        if current_mid is None or product not in last_mid:
            return
        cfg = MEAN_REVERSION[product]
        move = current_mid - last_mid[product]
        ret_hist = recent_returns.get(product, [])
        if rolling_autocorr(ret_hist[-int(cfg["window"]):]) > float(cfg["corr_threshold"]):
            return
        fair = current_mid - cfg["k"] * move
        best_bid, best_ask = top_of_book(depth)
        if best_bid is None or best_ask is None:
            return
        spread = best_ask - best_bid
        max_qty = int(cfg["max_qty"])
        self._take_against_fair(result, working_pos, product, depth, fair, max(1.0, 0.5 * spread), max_qty)
        self._flatten_near_fair(result, working_pos, product, depth, fair, max_qty)

    def _trade_secondary_mean_reversion(
        self,
        result: Dict[str, List[Order]],
        working_pos: Dict[str, int],
        last_mid: Dict[str, float],
        product: str,
        depth: OrderDepth,
    ) -> None:
        current_mid = mid_price(depth)
        if current_mid is None or product not in last_mid:
            return
        cfg = SECONDARY_MEAN_REVERSION[product]
        move = current_mid - last_mid[product]
        fair = current_mid - cfg["k"] * move
        best_bid, best_ask = top_of_book(depth)
        if best_bid is None or best_ask is None:
            return
        spread = best_ask - best_bid
        max_qty = int(cfg["max_qty"])
        self._take_against_fair(result, working_pos, product, depth, fair, max(1.0, 0.5 * spread), max_qty)
        self._flatten_near_fair(result, working_pos, product, depth, fair, max_qty)

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        working_pos = dict(state.position)

        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}
        if "m" in saved:
            last_mid = {ALIAS_TO_MR.get(k, k): v for k, v in saved.get("m", {}).items()}
        else:
            last_mid = saved.get("last_mid", {})
        if "r" in saved:
            recent_returns = {ALIAS_TO_MR.get(k, k): v for k, v in saved.get("r", {}).items()}
        else:
            recent_returns = saved.get("recent_returns", {})
        if not isinstance(recent_returns, dict):
            recent_returns = {}
        category_state = saved.get("c", {})
        if not isinstance(category_state, dict):
            category_state = {}
        if int(saved.get("ts", -1)) > state.timestamp:
            recent_returns = {}
            category_state = {}

        mids: Dict[str, float] = {}
        for product, depth in state.order_depths.items():
            mid = mid_price(depth)
            if mid is not None:
                mids[product] = mid
        self._current_depths = state.order_depths

        tracked_mr = set(MEAN_REVERSION) | set(SECONDARY_MEAN_REVERSION)
        for product in MEAN_REVERSION:
            if product in mids and product in last_mid:
                hist = recent_returns.get(product, [])
                if not isinstance(hist, list):
                    hist = []
                move = int(round((mids[product] - float(last_mid[product])) * RET_SCALE))
                hist.append(move)
                recent_returns[product] = hist[-MAX_RET_HISTORY:]

        for product in PEBBLES:
            depth = state.order_depths.get(product)
            if depth is not None:
                self._trade_pebbles(result, working_pos, mids, product, depth)

        for product in RV_MODELS if ENABLE_RV_MODELS else ():
            if product in CATEGORY_SUM_PRODUCTS:
                continue
            depth = state.order_depths.get(product)
            if depth is not None:
                self._trade_relative_value(result, working_pos, mids, product, depth)

        for model_name in CATEGORY_SUM_MODELS:
            self._trade_category_sum(result, working_pos, mids, category_state, model_name)

        for product in SNACKS:
            depth = state.order_depths.get(product)
            if depth is not None:
                self._trade_snack_imbalance(result, working_pos, product, depth)

        for product in MEAN_REVERSION:
            if product in CATEGORY_SUM_PRODUCTS:
                continue
            depth = state.order_depths.get(product)
            if depth is not None:
                self._trade_mean_reversion(result, working_pos, last_mid, recent_returns, product, depth)

        for product in SECONDARY_MEAN_REVERSION:
            depth = state.order_depths.get(product)
            if depth is not None:
                self._trade_secondary_mean_reversion(result, working_pos, last_mid, product, depth)

        saved_mids = {MR_ALIAS[p]: mids[p] for p in tracked_mr if p in mids}
        saved_returns = {MR_ALIAS[p]: recent_returns.get(p, []) for p in MEAN_REVERSION}
        trader_data = json.dumps({"m": saved_mids, "r": saved_returns, "c": category_state, "ts": state.timestamp}, separators=(",", ":"))
        return result, 0, trader_data

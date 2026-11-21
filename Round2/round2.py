from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import jsonpickle


MARKET_ACCESS_FEE = 50
PEPPER_STRUCTURAL_LONG = 40

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

LIMIT = {OSMIUM: 80, PEPPER: 80}

PARAMS = {
    OSMIUM: {
        "ema_alpha": 0.06,
        "take_edge": 2.0,
        "clear_edge": 0.5,
        "quote_offset": 1,
        "max_quote_size": 40,
        "soft_cap": 60,
    },
    PEPPER: {
        "fast_alpha": 0.40,
        "slow_alpha": 0.02,
        "take_edge": 1.0,
        "clear_edge": 2.0,
        "quote_offset": 1,
        "max_quote_size": 40,
        "soft_cap": 79,
        "momentum_dead_zone": 0.3,
        "momentum_scale": 8.0,
        "max_target": 60,
    },
}


class Trader:
    def __init__(self):
        self.osmium_ema: Optional[float] = None
        self.pepper_fast_ema: Optional[float] = None
        self.pepper_slow_ema: Optional[float] = None

    def bid(self) -> int:
        return max(0, int(MARKET_ACCESS_FEE))

    @staticmethod
    def _top_of_book(order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    @staticmethod
    def _mid(best_bid: Optional[int], best_ask: Optional[int]) -> Optional[float]:
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    @staticmethod
    def _microprice(order_depth: OrderDepth) -> Optional[float]:
        """Size-weighted mid: pulls toward the side with more resting volume."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        bid_sz = order_depth.buy_orders[best_bid]
        ask_sz = -order_depth.sell_orders[best_ask]
        total = bid_sz + ask_sz
        if total <= 0:
            return (best_bid + best_ask) / 2
        return (best_bid * ask_sz + best_ask * bid_sz) / total

    def _take(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_edge: float,
        position: int,
        orders: List[Order],
    ) -> Tuple[int, int]:
        limit = LIMIT[product]
        buy_used = 0
        sell_used = 0

        # Aggressive take of every ask strictly below fair - edge.
        if order_depth.sell_orders:
            asks = sorted(order_depth.sell_orders.items())
            for ask_price, ask_size in asks:
                if ask_price > fair_value - take_edge:
                    break
                available = limit - (position + buy_used)
                if available <= 0:
                    break
                qty = min(-ask_size, available)
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_used += qty

        # Aggressive hit of every bid strictly above fair + edge.
        if order_depth.buy_orders:
            bids = sorted(order_depth.buy_orders.items(), reverse=True)
            for bid_price, bid_size in bids:
                if bid_price < fair_value + take_edge:
                    break
                available = limit + (position - sell_used)
                if available <= 0:
                    break
                qty = min(bid_size, available)
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    sell_used += qty

        return buy_used, sell_used

    def _clear(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_edge: float,
        position: int,
        buy_used: int,
        sell_used: int,
        orders: List[Order],
    ) -> Tuple[int, int]:
        limit = LIMIT[product]
        net_position = position + buy_used - sell_used

        if net_position > 0:
            target_price = int(round(fair_value + clear_edge))
            interest = sum(
                v for p, v in order_depth.buy_orders.items() if p >= target_price
            )
            capacity = limit + (position - sell_used)
            qty = min(net_position, interest, capacity)
            if qty > 0:
                orders.append(Order(product, target_price, -qty))
                sell_used += qty

        elif net_position < 0:
            target_price = int(round(fair_value - clear_edge))
            interest = sum(
                -v for p, v in order_depth.sell_orders.items() if p <= target_price
            )
            capacity = limit - (position + buy_used)
            qty = min(-net_position, interest, capacity)
            if qty > 0:
                orders.append(Order(product, target_price, qty))
                buy_used += qty

        return buy_used, sell_used

    def _make(
        self,
        product: str,
        best_bid: int,
        best_ask: int,
        fair_value: float,
        position: int,
        buy_used: int,
        sell_used: int,
        orders: List[Order],
        cfg: dict,
    ):
        limit = LIMIT[product]
        soft_cap = cfg["soft_cap"]
        offset = cfg["quote_offset"]
        max_size = cfg["max_quote_size"]

        fair_px = int(round(fair_value))
        # Place at 1 inside best when there is room, otherwise at fair ± offset.
        bid_px = min(best_bid + 1, fair_px - offset)
        ask_px = max(best_ask - 1, fair_px + offset)
        if bid_px >= ask_px:
            bid_px = fair_px - offset
            ask_px = fair_px + offset

        buy_capacity = limit - (position + buy_used)
        sell_capacity = limit + (position - sell_used)

        # Inventory-aware size: cut the side that grows inventory once past soft cap.
        buy_size = max(0, min(max_size, buy_capacity))
        sell_size = max(0, min(max_size, sell_capacity))

        if position > soft_cap:
            buy_size = 0
        elif position > soft_cap / 2:
            buy_size = min(buy_size, max_size // 2)

        if position < -soft_cap:
            sell_size = 0
        elif position < -soft_cap / 2:
            sell_size = min(sell_size, max_size // 2)

        if buy_size > 0 and bid_px > 0:
            orders.append(Order(product, bid_px, buy_size))
        if sell_size > 0 and ask_px > 0:
            orders.append(Order(product, ask_px, -sell_size))

    def _osmium(self, order_depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        best_bid, best_ask = self._top_of_book(order_depth)
        mid = self._mid(best_bid, best_ask)
        if mid is None:
            return orders

        # Prefer microprice (size-weighted) when both sides exist; fall back to mid.
        mp = self._microprice(order_depth)
        price_input = mp if mp is not None else mid

        cfg = PARAMS[OSMIUM]
        if self.osmium_ema is None:
            self.osmium_ema = 10001.0
        self.osmium_ema = cfg["ema_alpha"] * price_input + (1 - cfg["ema_alpha"]) * self.osmium_ema
        fair_value = self.osmium_ema

        buy_used, sell_used = self._take(
            OSMIUM, order_depth, fair_value, cfg["take_edge"], position, orders
        )
        buy_used, sell_used = self._clear(
            OSMIUM, order_depth, fair_value, cfg["clear_edge"],
            position, buy_used, sell_used, orders,
        )

        if best_bid is not None and best_ask is not None:
            self._make(
                OSMIUM, best_bid, best_ask, fair_value,
                position, buy_used, sell_used, orders, cfg,
            )
        return orders

    def _pepper(self, order_depth: OrderDepth, position: int) -> List[Order]:
        orders: List[Order] = []
        best_bid, best_ask = self._top_of_book(order_depth)
        mid = self._mid(best_bid, best_ask)
        if mid is None:
            return orders

        # Microprice when both sides exist; else mid.
        mp = self._microprice(order_depth)
        price_input = mp if mp is not None else mid

        cfg = PARAMS[PEPPER]
        if self.pepper_fast_ema is None:
            self.pepper_fast_ema = price_input
        if self.pepper_slow_ema is None:
            self.pepper_slow_ema = price_input
        self.pepper_fast_ema = cfg["fast_alpha"] * price_input + (1 - cfg["fast_alpha"]) * self.pepper_fast_ema
        self.pepper_slow_ema = cfg["slow_alpha"] * price_input + (1 - cfg["slow_alpha"]) * self.pepper_slow_ema

        momentum_raw = self.pepper_fast_ema - self.pepper_slow_ema
        if abs(momentum_raw) < cfg["momentum_dead_zone"]:
            effective_momentum = 0.0
        else:
            sign = 1 if momentum_raw > 0 else -1
            effective_momentum = sign * (abs(momentum_raw) - cfg["momentum_dead_zone"])

        fair_value = self.pepper_fast_ema + 0.5 * effective_momentum

        buy_used, sell_used = self._take(
            PEPPER, order_depth, fair_value, cfg["take_edge"], position, orders
        )

        # Option B: structural long bias on top of momentum-driven target.
        target_position = PEPPER_STRUCTURAL_LONG + int(cfg["momentum_scale"] * effective_momentum)
        target_position = max(-cfg["max_target"], min(cfg["max_target"], target_position))

        # Clear drives us toward target_position gently.
        net_position = position + buy_used - sell_used
        if net_position > target_position:
            reduce_price = int(round(fair_value + cfg["clear_edge"]))
            interest = sum(
                v for p, v in order_depth.buy_orders.items() if p >= reduce_price
            )
            capacity = LIMIT[PEPPER] + (position - sell_used)
            qty = min(net_position - target_position, interest, capacity)
            if qty > 0:
                orders.append(Order(PEPPER, reduce_price, -qty))
                sell_used += qty
        elif net_position < target_position:
            add_price = int(round(fair_value - cfg["clear_edge"]))
            interest = sum(
                -v for p, v in order_depth.sell_orders.items() if p <= add_price
            )
            capacity = LIMIT[PEPPER] - (position + buy_used)
            qty = min(target_position - net_position, interest, capacity)
            if qty > 0:
                orders.append(Order(PEPPER, add_price, qty))
                buy_used += qty

        if best_bid is not None and best_ask is not None:
            self._make(
                PEPPER, best_bid, best_ask, fair_value,
                position, buy_used, sell_used, orders, cfg,
            )

        return orders

    def run(self, state: TradingState):
        if state.traderData:
            try:
                saved = jsonpickle.decode(state.traderData)
                self.osmium_ema = saved.get("osmium_ema", None)
                self.pepper_fast_ema = saved.get("pepper_fast_ema", None)
                self.pepper_slow_ema = saved.get("pepper_slow_ema", None)
            except Exception:
                pass

        result: Dict[str, List[Order]] = {}

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._osmium(
                state.order_depths[OSMIUM],
                state.position.get(OSMIUM, 0),
            )

        if PEPPER in state.order_depths:
            result[PEPPER] = self._pepper(
                state.order_depths[PEPPER],
                state.position.get(PEPPER, 0),
            )

        trader_data = jsonpickle.encode({
            "osmium_ema": self.osmium_ema,
            "pepper_fast_ema": self.pepper_fast_ema,
            "pepper_slow_ema": self.pepper_slow_ema,
        })

        conversions = 0
        return result, conversions, trader_data

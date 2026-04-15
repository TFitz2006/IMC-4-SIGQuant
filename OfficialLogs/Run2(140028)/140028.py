import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from datamodel import Order, OrderDepth, TradingState


POSITION_LIMITS = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}


@dataclass(frozen=True)
class ProductConfig:
    static_fair: Optional[float]
    model_weight: float
    drift_per_timestamp: float
    fair_offset: float
    take_buy_edge: float
    take_sell_edge: float
    rebalance_buy_edge: float
    rebalance_sell_edge: float
    quote_half_spread: float
    quote_clear_edge: float
    inventory_penalty: float
    quote_size: int
    target_position: int


@dataclass(frozen=True)
class BookStats:
    best_bid: Optional[int]
    best_bid_volume: int
    best_ask: Optional[int]
    best_ask_volume: int
    bid_wall: Optional[int]
    ask_wall: Optional[int]
    wall_mid: Optional[float]
    mid_price: Optional[float]


PRODUCT_CONFIGS = {
    # Resembles Frankfurt Hedgehogs' Rainforest Resin playbook:
    # stable around a known fair value with wide enough spread to market make.
    "ASH_COATED_OSMIUM": ProductConfig(
        static_fair=10000.0,
        model_weight=0.9,
        drift_per_timestamp=0.0,
        fair_offset=0.0,
        take_buy_edge=1.0,
        take_sell_edge=1.0,
        rebalance_buy_edge=0.0,
        rebalance_sell_edge=0.0,
        quote_half_spread=4.0,
        quote_clear_edge=1.0,
        inventory_penalty=0.08,
        quote_size=14,
        target_position=0,
    ),
    # Resembles a drift-aware version of their dynamic-fair-value Kelp approach:
    # same micro mean reversion, but on top of a strong deterministic uptrend.
    "INTARIAN_PEPPER_ROOT": ProductConfig(
        static_fair=None,
        model_weight=0.8,
        drift_per_timestamp=0.001,
        fair_offset=1.0,
        take_buy_edge=0.5,
        take_sell_edge=3.0,
        rebalance_buy_edge=0.5,
        rebalance_sell_edge=2.0,
        quote_half_spread=3.0,
        quote_clear_edge=1.0,
        inventory_penalty=0.05,
        quote_size=14,
        target_position=12,
    ),
}


class Trader:
    def run(self, state: TradingState):
        memory = self._load_memory(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            if product not in PRODUCT_CONFIGS:
                result[product] = []
                continue

            orders, product_memory = self._trade_product(
                product=product,
                order_depth=order_depth,
                position=state.position.get(product, 0),
                timestamp=state.timestamp,
                product_memory=memory.get(product, {}),
            )
            memory[product] = product_memory
            result[product] = orders

        conversions = 0
        trader_data = json.dumps(memory, separators=(",", ":"))
        return result, conversions, trader_data

    def _trade_product(
        self,
        product: str,
        order_depth: OrderDepth,
        position: int,
        timestamp: int,
        product_memory: Dict[str, float],
    ) -> Tuple[List[Order], Dict[str, float]]:
        config = PRODUCT_CONFIGS[product]
        book = self._book_stats(order_depth)
        observed_reference = book.wall_mid if book.wall_mid is not None else book.mid_price

        if observed_reference is None:
            return [], product_memory

        fair_value, updated_memory = self._fair_value(
            product=product,
            timestamp=timestamp,
            observed_reference=observed_reference,
            product_memory=product_memory,
        )
        target_position = self._target_position(product, fair_value, observed_reference, config)

        buy_headroom = POSITION_LIMITS[product] - position
        sell_headroom = POSITION_LIMITS[product] + position
        working_position = position
        orders: List[Order] = []

        asks = sorted(order_depth.sell_orders.items())
        bids = sorted(order_depth.buy_orders.items(), reverse=True)

        for ask_price, ask_volume in asks:
            available = abs(int(ask_volume))
            if available <= 0 or buy_headroom <= 0:
                continue
            if ask_price <= fair_value - config.take_buy_edge:
                quantity = min(buy_headroom, available)
                orders.append(Order(product, int(ask_price), int(quantity)))
                buy_headroom -= quantity
                working_position += quantity
            elif ask_price <= fair_value - config.rebalance_buy_edge and working_position < target_position:
                quantity = min(buy_headroom, available, target_position - working_position)
                if quantity > 0:
                    orders.append(Order(product, int(ask_price), int(quantity)))
                    buy_headroom -= quantity
                    working_position += quantity

        for bid_price, bid_volume in bids:
            available = abs(int(bid_volume))
            if available <= 0 or sell_headroom <= 0:
                continue
            if bid_price >= fair_value + config.take_sell_edge:
                quantity = min(sell_headroom, available)
                orders.append(Order(product, int(bid_price), -int(quantity)))
                sell_headroom -= quantity
                working_position -= quantity
            elif bid_price >= fair_value + config.rebalance_sell_edge and working_position > target_position:
                quantity = min(sell_headroom, available, working_position - target_position)
                if quantity > 0:
                    orders.append(Order(product, int(bid_price), -int(quantity)))
                    sell_headroom -= quantity
                    working_position -= quantity

        reservation_price = fair_value - config.inventory_penalty * (working_position - target_position)
        bid_quote, ask_quote = self._quote_prices(
            product=product,
            config=config,
            fair_value=fair_value,
            reservation_price=reservation_price,
            book=book,
            working_position=working_position,
            target_position=target_position,
        )

        inventory_gap = target_position - working_position
        buy_quote_size = min(buy_headroom, config.quote_size + max(0, inventory_gap // 10))
        sell_quote_size = min(sell_headroom, config.quote_size + max(0, (-inventory_gap) // 10))

        if buy_quote_size > 0 and working_position >= POSITION_LIMITS[product] - 5:
            buy_quote_size = 0
        if sell_quote_size > 0 and working_position <= -POSITION_LIMITS[product] + 5:
            sell_quote_size = 0

        if buy_quote_size > 0 and bid_quote is not None and (book.best_ask is None or bid_quote < book.best_ask):
            orders.append(Order(product, bid_quote, int(buy_quote_size)))
        if sell_quote_size > 0 and ask_quote is not None and (book.best_bid is None or ask_quote > book.best_bid):
            orders.append(Order(product, ask_quote, -int(sell_quote_size)))

        return orders, updated_memory

    def _fair_value(
        self,
        product: str,
        timestamp: int,
        observed_reference: float,
        product_memory: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        config = PRODUCT_CONFIGS[product]
        memory = dict(product_memory)
        last_timestamp = memory.get("last_timestamp")
        if last_timestamp is not None and timestamp < last_timestamp:
            memory = {}

        if config.static_fair is not None:
            model_fair = config.static_fair
        else:
            intercept = memory.get("trend_intercept")
            fresh_intercept = observed_reference - config.drift_per_timestamp * timestamp
            if intercept is None:
                intercept = fresh_intercept
            else:
                intercept = 0.9 * intercept + 0.1 * fresh_intercept
            memory["trend_intercept"] = float(intercept)
            model_fair = intercept + config.drift_per_timestamp * timestamp + config.fair_offset

        fair_value = config.model_weight * model_fair + (1.0 - config.model_weight) * observed_reference

        memory["last_timestamp"] = int(timestamp)
        memory["last_reference"] = float(observed_reference)
        return fair_value, memory

    @staticmethod
    def _target_position(
        product: str,
        fair_value: float,
        observed_reference: float,
        config: ProductConfig,
    ) -> int:
        if product != "INTARIAN_PEPPER_ROOT":
            return config.target_position

        deviation = fair_value - observed_reference
        if deviation >= 2.0:
            return 28
        if deviation >= 0.75:
            return 20
        if deviation <= -2.0:
            return 2
        if deviation <= -0.75:
            return 6
        return config.target_position

    @staticmethod
    def _book_stats(order_depth: OrderDepth) -> BookStats:
        best_bid = None
        best_bid_volume = 0
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = abs(int(order_depth.buy_orders[best_bid]))

        best_ask = None
        best_ask_volume = 0
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = abs(int(order_depth.sell_orders[best_ask]))

        bid_wall = min(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        ask_wall = max(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        wall_mid = None
        if bid_wall is not None and ask_wall is not None:
            wall_mid = (bid_wall + ask_wall) / 2.0

        mid_price = Trader._mid_price(best_bid, best_ask)
        return BookStats(
            best_bid=best_bid,
            best_bid_volume=best_bid_volume,
            best_ask=best_ask,
            best_ask_volume=best_ask_volume,
            bid_wall=bid_wall,
            ask_wall=ask_wall,
            wall_mid=wall_mid,
            mid_price=mid_price,
        )

    @staticmethod
    def _mid_price(best_bid: Optional[int], best_ask: Optional[int]) -> Optional[float]:
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return None

    @staticmethod
    def _quote_prices(
        product: str,
        config: ProductConfig,
        fair_value: float,
        reservation_price: float,
        book: BookStats,
        working_position: int,
        target_position: int,
    ) -> Tuple[Optional[int], Optional[int]]:
        bid_quote = math.floor(reservation_price - config.quote_half_spread)
        ask_quote = math.ceil(reservation_price + config.quote_half_spread)

        if product == "INTARIAN_PEPPER_ROOT":
            ask_buffer = 2.0 if working_position <= target_position + 4 else 1.0
            bid_buffer = 0.5 if working_position < target_position else config.quote_clear_edge
        else:
            ask_buffer = config.quote_clear_edge
            bid_buffer = config.quote_clear_edge

        bid_ceiling = math.floor(fair_value - bid_buffer)
        ask_floor = math.ceil(fair_value + ask_buffer)

        if book.bid_wall is not None:
            bid_quote = max(bid_quote, book.bid_wall + 1)
        if book.ask_wall is not None:
            ask_quote = min(ask_quote, book.ask_wall - 1)

        if book.best_bid is not None:
            improved_bid = book.best_bid + 1
            if improved_bid <= bid_ceiling:
                bid_quote = max(bid_quote, improved_bid)
            elif book.best_bid <= bid_ceiling:
                bid_quote = max(bid_quote, book.best_bid)

        if book.best_ask is not None:
            improved_ask = book.best_ask - 1
            if product == "INTARIAN_PEPPER_ROOT" and working_position <= target_position:
                pass
            elif improved_ask >= ask_floor:
                ask_quote = min(ask_quote, improved_ask)
            elif book.best_ask >= ask_floor:
                ask_quote = min(ask_quote, book.best_ask)

        bid_quote = min(bid_quote, bid_ceiling)
        ask_quote = max(ask_quote, ask_floor)

        if product == "INTARIAN_PEPPER_ROOT" and working_position <= target_position:
            if book.best_ask is not None:
                ask_quote = max(ask_quote, min(int(book.best_ask), int(math.ceil(fair_value + 2.0))))
            else:
                ask_quote = max(ask_quote, int(math.ceil(fair_value + 2.0)))

        if book.best_ask is not None:
            bid_quote = min(bid_quote, book.best_ask - 1)
        if book.best_bid is not None:
            ask_quote = max(ask_quote, book.best_bid + 1)

        if book.best_bid is not None and book.best_ask is not None and bid_quote >= ask_quote:
            inside_mid = (book.best_bid + book.best_ask) / 2.0
            bid_quote = min(int(math.floor(inside_mid)), book.best_ask - 1, bid_ceiling)
            ask_quote = max(int(math.ceil(inside_mid)), book.best_bid + 1, ask_floor)

        if bid_quote >= ask_quote:
            if product == "ASH_COATED_OSMIUM":
                bid_quote = math.floor(fair_value - 1)
                ask_quote = math.ceil(fair_value + 1)
            else:
                bid_quote = math.floor(reservation_price)
                ask_quote = math.ceil(reservation_price + 1)

        return bid_quote, ask_quote

    @staticmethod
    def _load_memory(trader_data: str) -> Dict[str, Dict[str, float]]:
        if not trader_data:
            return {}
        try:
            parsed = json.loads(trader_data)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {}
"""现货回测执行器.

采用目标仓位权重模型:
- `target_weight=0.0` 表示空仓
- `target_weight=1.0` 表示满仓持有 BTC
- 仅支持现货做多, 自动禁止负仓位
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Trade:
    datetime: str
    symbol: str
    side: str
    price: float
    quantity: float
    notional: float
    fee: float
    pnl: float
    position_after: float
    target_weight_after: float


class SpotExecutor:
    def __init__(
        self,
        *,
        init_balance: float = 1_000_000.0,
        fee_rate: float = 0.001,
        min_qty: float = 1e-8,
        min_notional: float = 0.0,
        min_weight_change: float = 0.0,
    ) -> None:
        self.init_balance = float(init_balance)
        self.fee_rate = float(fee_rate)
        self.min_qty = float(min_qty)
        self.min_notional = float(min_notional)
        self.min_weight_change = float(min_weight_change)

        self.cash = float(init_balance)
        self.position_qty = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.total_fee = 0.0
        self.last_price = 0.0
        self.last_target_weight = 0.0
        self._trades: List[Trade] = []

    def equity_at_price(self, price: float) -> float:
        return self.cash + self.position_qty * float(price)

    def get_net_position(self, _symbol: str) -> float:
        return self.position_qty

    def get_position_weight(self, price: float) -> float:
        eq = self.equity_at_price(price)
        if eq <= 0:
            return 0.0
        return (self.position_qty * price) / eq

    def set_target_weight(self, symbol: str, target_weight: float, price: float, dt: str = "") -> None:
        price = float(price)
        self.last_price = price
        target_weight = max(0.0, min(float(target_weight), 1.0))
        current_weight = self.get_position_weight(price)
        if abs(target_weight - current_weight) < self.min_weight_change:
            self.last_target_weight = current_weight
            return

        equity_before = self.equity_at_price(price)
        target_notional = equity_before * target_weight
        target_qty = target_notional / price if price > 0 else 0.0
        delta_qty = target_qty - self.position_qty
        delta_notional = abs(delta_qty) * price
        if abs(delta_qty) < self.min_qty:
            self.last_target_weight = target_weight
            return
        if delta_notional < self.min_notional:
            self.last_target_weight = current_weight
            return

        if delta_qty > 0:
            gross_cost = delta_qty * price
            gross_with_fee = gross_cost * (1.0 + self.fee_rate)
            if gross_with_fee > self.cash:
                affordable_qty = self.cash / (price * (1.0 + self.fee_rate))
                delta_qty = max(0.0, affordable_qty)
                gross_cost = delta_qty * price
            fee = gross_cost * self.fee_rate
            if delta_qty < self.min_qty:
                self.last_target_weight = self.get_position_weight(price)
                return

            old_cost = self.position_qty * self.avg_entry_price
            new_cost = delta_qty * price
            self.position_qty += delta_qty
            self.avg_entry_price = (old_cost + new_cost) / self.position_qty if self.position_qty > 0 else 0.0
            self.cash -= gross_cost + fee
            self.total_fee += fee
            self._trades.append(
                Trade(
                    datetime=dt,
                    symbol=symbol,
                    side="BUY",
                    price=price,
                    quantity=delta_qty,
                    notional=gross_cost,
                    fee=fee,
                    pnl=0.0,
                    position_after=self.position_qty,
                    target_weight_after=target_weight,
                )
            )
        else:
            sell_qty = min(self.position_qty, abs(delta_qty))
            if sell_qty < self.min_qty:
                self.last_target_weight = self.get_position_weight(price)
                return
            gross_proceeds = sell_qty * price
            fee = gross_proceeds * self.fee_rate
            pnl = sell_qty * (price - self.avg_entry_price)
            self.position_qty -= sell_qty
            if self.position_qty < self.min_qty:
                self.position_qty = 0.0
                self.avg_entry_price = 0.0
            self.cash += gross_proceeds - fee
            self.realized_pnl += pnl
            self.total_fee += fee
            self._trades.append(
                Trade(
                    datetime=dt,
                    symbol=symbol,
                    side="SELL",
                    price=price,
                    quantity=sell_qty,
                    notional=gross_proceeds,
                    fee=fee,
                    pnl=pnl,
                    position_after=self.position_qty,
                    target_weight_after=target_weight,
                )
            )

        self.last_target_weight = self.get_position_weight(price)

    def mark_to_market(self, _symbol: str, price: float) -> None:
        self.last_price = float(price)

    def get_equity(self) -> Dict[str, Any]:
        holdings_value = self.position_qty * self.last_price
        equity = self.cash + holdings_value
        unrealized = self.position_qty * (self.last_price - self.avg_entry_price)
        return {
            "balance": round(self.cash, 6),
            "equity": round(equity, 6),
            "holdings_value": round(holdings_value, 6),
            "position_qty": round(self.position_qty, 10),
            "position_weight": round(self.get_position_weight(self.last_price) if self.last_price > 0 else 0.0, 6),
            "avg_entry_price": round(self.avg_entry_price, 6),
            "unrealized_pnl": round(unrealized, 6),
            "realized_pnl": round(self.realized_pnl, 6),
            "total_fee": round(self.total_fee, 6),
        }

    def get_trades(self) -> List[Trade]:
        return list(self._trades)

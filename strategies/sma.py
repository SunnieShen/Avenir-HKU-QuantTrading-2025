"""简单均线策略示例."""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

from strategies.base import BaseStrategy, StrategyContext


class Strategy(BaseStrategy):
    def __init__(self, ctx: StrategyContext):
        super().__init__(ctx)
        self.short_n = self.param_int("short", 10)
        self.long_n = self.param_int("long", 40)
        self.max_weight = self.param_float("max_weight", 0.95)
        self._closes = deque(maxlen=self.long_n + 2)

    def on_bar(self, bar: Dict[str, Any]) -> Optional[float]:
        self._closes.append(float(bar["close"]))
        if len(self._closes) < self.long_n:
            return None
        closes = list(self._closes)
        ma_s = sum(closes[-self.short_n:]) / self.short_n
        ma_l = sum(closes[-self.long_n:]) / self.long_n
        if ma_s > ma_l:
            return self.max_weight
        if ma_s < ma_l:
            return 0.0
        return None

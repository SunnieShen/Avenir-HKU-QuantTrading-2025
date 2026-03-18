"""策略基类."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class StrategyContext:
    symbol: str
    bar_seconds: int
    params: Dict[str, Any] = field(default_factory=dict)


class BaseStrategy:
    def __init__(self, ctx: StrategyContext):
        self.ctx = ctx

    def param(self, key: str, default: Any = None) -> Any:
        return self.ctx.params.get(key, default)

    def param_int(self, key: str, default: int = 0) -> int:
        return int(self.ctx.params.get(key, default))

    def param_float(self, key: str, default: float = 0.0) -> float:
        return float(self.ctx.params.get(key, default))

    def param_bool(self, key: str, default: bool = False) -> bool:
        return bool(self.ctx.params.get(key, default))

    def on_start(self) -> None:
        pass

    def on_bar(self, bar: Dict[str, Any]) -> Optional[float]:
        raise NotImplementedError

    def on_finish(self) -> None:
        pass

    def generate_signal_frame(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        return None

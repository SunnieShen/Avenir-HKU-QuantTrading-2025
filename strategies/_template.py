"""策略模板.

复制这个文件并重命名，然后实现 `on_bar()` 或 `generate_signal_frame()`。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from strategies.base import BaseStrategy, StrategyContext


class Strategy(BaseStrategy):
    def __init__(self, ctx: StrategyContext):
        super().__init__(ctx)
        self.max_weight = self.param_float("max_weight", 0.95)

    def on_bar(self, bar: Dict[str, Any]) -> Optional[float]:
        # 返回目标仓位权重:
        # 1.0 = 满仓, 0.0 = 空仓, None = 保持
        return None

    def generate_signal_frame(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        # 如果是机器学习策略, 返回包含 prediction/target_weight 的 DataFrame
        return None

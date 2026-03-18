"""BTC 单标的 LightGBM 信号策略."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.lightgbm_signal import SignalConfig, walk_forward_signal_frame
from strategies.base import BaseStrategy, StrategyContext


class Strategy(BaseStrategy):
    def __init__(self, ctx: StrategyContext):
        super().__init__(ctx)
        self.signal_meta = {}

    def on_bar(self, bar):
        raise NotImplementedError("lightgbm_btc uses generate_signal_frame(), not on_bar().")

    def generate_signal_frame(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        cfg = SignalConfig(
            prediction_horizon_bars=self.param_int("prediction_horizon_bars", 24),
            train_window_bars=self.param_int("train_window_bars", 24 * 180),
            min_train_bars=self.param_int("min_train_bars", 24 * 90),
            retrain_every_bars=self.param_int("retrain_every_bars", 24),
            valid_bars=self.param_int("valid_bars", 24 * 14),
            learning_rate=self.param_float("learning_rate", 0.05),
            n_estimators=self.param_int("n_estimators", 400),
            num_leaves=self.param_int("num_leaves", 63),
            feature_fraction=self.param_float("feature_fraction", 0.9),
            bagging_fraction=self.param_float("bagging_fraction", 0.8),
            bagging_freq=self.param_int("bagging_freq", 1),
            min_data_in_leaf=self.param_int("min_data_in_leaf", 30),
            lambda_l2=self.param_float("lambda_l2", 1.0),
            random_state=self.param_int("random_state", 42),
            use_time_decay=self.param_bool("use_time_decay", True),
            halflife_bars=self.param_int("halflife_bars", 24 * 90),
            long_threshold=self.param_float("long_threshold", 0.0015),
            flat_threshold=self.param_float("flat_threshold", 0.0),
            max_weight=self.param_float("max_weight", 0.95),
        )
        signal_df, meta = walk_forward_signal_frame(df, cfg)
        self.signal_meta = meta
        return signal_df

"""单标的 BTC LightGBM 信号生成.

优先使用 LightGBM；若本机缺少 `libomp` 等动态库，则自动回退到
`HistGradientBoostingRegressor`，保证本地框架仍可运行。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
    LIGHTGBM_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local runtime
    lgb = None
    LIGHTGBM_AVAILABLE = False
    LIGHTGBM_IMPORT_ERROR = str(exc)


@dataclass
class SignalConfig:
    windows: Tuple[int, ...] = (8, 24, 48, 96, 288)
    prediction_horizon_bars: int = 24
    train_window_bars: int = 24 * 180
    min_train_bars: int = 24 * 90
    retrain_every_bars: int = 24
    valid_bars: int = 24 * 14
    learning_rate: float = 0.05
    n_estimators: int = 400
    num_leaves: int = 63
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    min_data_in_leaf: int = 30
    lambda_l2: float = 1.0
    random_state: int = 42
    use_time_decay: bool = True
    halflife_bars: int = 24 * 90
    long_threshold: float = 0.0015
    flat_threshold: float = 0.0
    max_weight: float = 0.95


def _vwap(df: pd.DataFrame) -> pd.Series:
    vwap = df["quote_volume"] / df["volume"].replace(0, np.nan)
    return vwap.replace([np.inf, -np.inf], np.nan).fillna(df["close"])


def build_feature_frame(df: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    price = _vwap(df)
    ret1 = price.pct_change()
    hlp = (np.log(df["high"] / df["low"]).replace([np.inf, -np.inf], np.nan) ** 2) / (4.0 * np.log(2.0))
    turnover = (df["quote_volume"] / price).replace([np.inf, -np.inf], np.nan)

    feats: Dict[str, pd.Series] = {}
    for w in cfg.windows:
        ema = price.ewm(span=w, min_periods=w).mean()
        feats[f"mom_{w}"] = price / price.shift(w) - 1.0
        feats[f"emagap_{w}"] = price / ema - 1.0
        feats[f"vol_{w}"] = ret1.rolling(w).std()
        feats[f"pk_{w}"] = hlp.rolling(w).mean()
        feats[f"tostd_{w}"] = turnover.pct_change(fill_method=None).rolling(w).std()
        feats[f"ret_mean_{w}"] = ret1.rolling(w).mean()
        feats[f"price_z_{w}"] = (price - price.rolling(w).mean()) / price.rolling(w).std()

    hours = pd.Index(df.index).hour.values
    dows = pd.Index(df.index).dayofweek.values
    feats["sin_h"] = pd.Series(np.sin(2 * np.pi * hours / 24.0), index=df.index)
    feats["cos_h"] = pd.Series(np.cos(2 * np.pi * hours / 24.0), index=df.index)
    feats["sin_d"] = pd.Series(np.sin(2 * np.pi * dows / 7.0), index=df.index)
    feats["cos_d"] = pd.Series(np.cos(2 * np.pi * dows / 7.0), index=df.index)
    feats["volume_z_24"] = (df["volume"] - df["volume"].rolling(24).mean()) / df["volume"].rolling(24).std()
    feats["quote_volume_z_24"] = (df["quote_volume"] - df["quote_volume"].rolling(24).mean()) / df["quote_volume"].rolling(24).std()
    feats["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    out = pd.DataFrame(feats, index=df.index)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_target(price: pd.Series, horizon_bars: int) -> pd.Series:
    return price.shift(-horizon_bars) / price - 1.0


def _build_sample_weights(n_rows: int, cfg: SignalConfig) -> np.ndarray:
    if not cfg.use_time_decay or n_rows <= 1:
        return np.ones(n_rows, dtype=np.float32)
    idx = np.arange(n_rows, dtype=np.float64)
    ages = idx.max() - idx
    weights = np.exp(-np.log(2.0) * ages / max(float(cfg.halflife_bars), 1.0))
    return weights.astype(np.float32)


def _fit_model(X_train: pd.DataFrame, y_train: pd.Series, cfg: SignalConfig):
    sample_weight = _build_sample_weights(len(X_train), cfg)
    if LIGHTGBM_AVAILABLE:
        model = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            num_leaves=cfg.num_leaves,
            feature_fraction=cfg.feature_fraction,
            bagging_fraction=cfg.bagging_fraction,
            bagging_freq=cfg.bagging_freq,
            min_data_in_leaf=cfg.min_data_in_leaf,
            lambda_l2=cfg.lambda_l2,
            random_state=cfg.random_state,
            verbose=-1,
        )

        valid_bars = min(cfg.valid_bars, max(0, len(X_train) // 5))
        if valid_bars >= 50 and len(X_train) > valid_bars + 100:
            split = len(X_train) - valid_bars
            model.fit(
                X_train.iloc[:split],
                y_train.iloc[:split],
                sample_weight=sample_weight[:split],
                eval_set=[(X_train.iloc[split:], y_train.iloc[split:])],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        return model, "lightgbm"

    model = HistGradientBoostingRegressor(
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.num_leaves,
        max_iter=cfg.n_estimators,
        min_samples_leaf=cfg.min_data_in_leaf,
        l2_regularization=cfg.lambda_l2,
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model, "hist_gradient_boosting"


def prediction_to_weight(pred: float, cfg: SignalConfig) -> float:
    if not np.isfinite(pred) or pred <= cfg.flat_threshold:
        return 0.0
    if pred >= cfg.long_threshold:
        return cfg.max_weight
    width = cfg.long_threshold - cfg.flat_threshold
    if width <= 0:
        return cfg.max_weight
    scaled = (pred - cfg.flat_threshold) / width
    return float(np.clip(scaled, 0.0, 1.0) * cfg.max_weight)


def walk_forward_signal_frame(df: pd.DataFrame, cfg: SignalConfig) -> tuple[pd.DataFrame, Dict[str, object]]:
    price = _vwap(df)
    X = build_feature_frame(df, cfg)
    y = build_target(price, cfg.prediction_horizon_bars)

    valid_train_mask = X.notna().all(axis=1) & y.notna()
    predictions = pd.Series(np.nan, index=df.index, dtype=np.float64)
    model_ids = pd.Series("", index=df.index, dtype=object)
    importances: List[pd.Series] = []
    fit_ranges: List[Dict[str, object]] = []

    n_rows = len(df)
    start_bar = max(cfg.min_train_bars, max(cfg.windows))
    model_counter = 0

    for pred_start in range(start_bar, n_rows - cfg.prediction_horizon_bars, cfg.retrain_every_bars):
        train_end = pred_start - cfg.prediction_horizon_bars
        if train_end <= 0:
            continue
        train_start = max(0, train_end - cfg.train_window_bars)
        train_idx = np.arange(train_start, train_end)
        train_mask = valid_train_mask.iloc[train_idx].to_numpy()
        if int(train_mask.sum()) < cfg.min_train_bars:
            continue

        train_positions = train_idx[train_mask]
        X_train = X.iloc[train_positions]
        y_train = y.iloc[train_positions]
        if len(X_train) < cfg.min_train_bars:
            continue

        model, model_type = _fit_model(X_train, y_train, cfg)
        model_counter += 1
        pred_end = min(pred_start + cfg.retrain_every_bars, n_rows - cfg.prediction_horizon_bars)
        X_pred = X.iloc[pred_start:pred_end]
        pred_values = model.predict(X_pred)
        pred_mask = X_pred.notna().all(axis=1).to_numpy()
        pred_index = X_pred.index[pred_mask]
        predictions.loc[pred_index] = pred_values[pred_mask]
        model_ids.loc[pred_index] = f"model_{model_counter:03d}"

        if hasattr(model, "feature_importances_"):
            importances.append(pd.Series(model.feature_importances_, index=X_train.columns))
        fit_ranges.append(
            {
                "model_id": f"model_{model_counter:03d}",
                "model_type": model_type,
                "train_start": X_train.index.min().isoformat(),
                "train_end": X_train.index.max().isoformat(),
                "pred_start": X_pred.index.min().isoformat(),
                "pred_end": X_pred.index.max().isoformat(),
                "train_rows": int(len(X_train)),
            }
        )

    signal_df = pd.DataFrame(index=df.index)
    signal_df["prediction"] = predictions.astype(float)
    signal_df["target_return"] = y.astype(float)
    signal_df["target_weight"] = signal_df["prediction"].apply(lambda x: prediction_to_weight(float(x), cfg) if pd.notna(x) else 0.0)
    signal_df["model_id"] = model_ids

    avg_importance = (
        pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        if importances
        else pd.Series(dtype=np.float64)
    )
    meta = {
        "config": asdict(cfg),
        "preferred_model": "lightgbm",
        "runtime_model": fit_ranges[-1]["model_type"] if fit_ranges else ("lightgbm" if LIGHTGBM_AVAILABLE else "hist_gradient_boosting"),
        "lightgbm_available": LIGHTGBM_AVAILABLE,
        "lightgbm_import_error": LIGHTGBM_IMPORT_ERROR,
        "n_models": model_counter,
        "fit_ranges": fit_ranges,
        "feature_importance": avg_importance.to_dict(),
    }
    return signal_df, meta

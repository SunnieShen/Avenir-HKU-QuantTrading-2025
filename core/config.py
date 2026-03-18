"""配置系统.

优先级: CLI 参数 > YAML 文件 > 默认值。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_kv(kvs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in kvs:
        if "=" not in item:
            raise ValueError(f"Invalid param '{item}', expected key=value")
        key, raw_val = item.split("=", 1)
        val = raw_val.strip()
        if val.lower() in {"true", "false"}:
            out[key.strip()] = val.lower() == "true"
            continue
        try:
            out[key.strip()] = float(val) if "." in val else int(val)
        except ValueError:
            out[key.strip()] = val
    return out


def apply_cli_overrides(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    if getattr(args, "start", ""):
        cfg.setdefault("backtest", {})["start"] = args.start
    if getattr(args, "end", ""):
        cfg.setdefault("backtest", {})["end"] = args.end
    if getattr(args, "bar", ""):
        cfg.setdefault("backtest", {})["bar_seconds"] = int(args.bar)
    if getattr(args, "symbol", ""):
        cfg.setdefault("backtest", {})["symbol"] = args.symbol
    if getattr(args, "data_dir", ""):
        cfg.setdefault("backtest", {})["data_dir"] = args.data_dir
    if getattr(args, "balance", ""):
        cfg.setdefault("backtest", {})["init_balance"] = float(args.balance)
    if getattr(args, "fee", ""):
        cfg.setdefault("backtest", {})["fee_rate"] = float(args.fee)
    if getattr(args, "strategy", ""):
        cfg.setdefault("strategy", {})["name"] = args.strategy
    if getattr(args, "param", None):
        cfg.setdefault("strategy", {}).setdefault("params", {})
        cfg["strategy"]["params"].update(_parse_kv(args.param))
    return cfg


def defaults() -> Dict[str, Any]:
    return {
        "backtest": {
            "data_dir": str(PROJECT_ROOT / "data"),
            "symbol": "BTCUSDT",
            "start": "2020-01-01",
            "end": "2025-01-01",
            "bar_seconds": 3600,
            "init_balance": 1_000_000.0,
            "fee_rate": 0.001,
            "min_trade_notional": 1_000.0,
            "min_weight_change": 0.02,
        },
        "strategy": {
            "name": "lightgbm_btc",
            "params": {
                "prediction_horizon_bars": 24,
                "train_window_bars": 24 * 180,
                "min_train_bars": 24 * 90,
                "retrain_every_bars": 24,
                "long_threshold": 0.0015,
                "flat_threshold": 0.0,
                "max_weight": 0.95,
            },
        },
        "output": {
            "save_bars": True,
            "save_signal": True,
            "save_equity": True,
            "save_trades": True,
            "save_predictions": True,
        },
        "experiment": {
            "root_dir": str(PROJECT_ROOT / "experiments"),
            "overwrite": True,
        },
        "scan": {
            "filters": {
                "require_competition_ready": True,
                "forbid_short": True,
                "max_avg_trades_per_day": 24,
            }
        },
    }

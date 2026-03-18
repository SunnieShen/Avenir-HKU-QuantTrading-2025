"""回测运行器."""
from __future__ import annotations

import csv
import importlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
import yaml

from core.competition import build_competition_report
from core.config import defaults
from core.data_loader import DEFAULT_DATA_DIR, load, to_bar_list
from core.executor import SpotExecutor
from core.metrics import calc_bucketed_metrics, calc_metrics, calc_rolling_window_metrics, calc_trade_metrics
from core.plot import (
    plot_bucket_metrics,
    plot_monthly_returns,
    plot_price_signals,
    plot_return_drawdown,
    plot_rolling_window_vs_buy_hold,
    plot_trade_analysis,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunResult:
    symbol: str
    strategy: str
    ok: bool
    error: str = ""
    metrics: Optional[Dict[str, Any]] = None
    trade_metrics: Optional[Dict[str, Any]] = None
    bucket_metrics: Optional[List[Dict[str, Any]]] = None
    rolling_metrics: Optional[List[Dict[str, Any]]] = None
    competition_report: Optional[Dict[str, Any]] = None
    signal_meta: Optional[Dict[str, Any]] = None
    exp_dir: str = ""


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_strategy_class(name: str):
    try:
        return importlib.import_module(f"strategies.{name}").Strategy
    except ModuleNotFoundError:
        return importlib.import_module(name).Strategy


def _compute_preload_start(start: str, bar_seconds: int, strategy_params: Dict[str, Any]) -> str:
    start_ts = pd.Timestamp(start, tz="UTC")
    extra_bars = max(
        int(strategy_params.get("train_window_bars", 0)),
        int(strategy_params.get("min_train_bars", 0)),
    )
    extra_bars += int(strategy_params.get("prediction_horizon_bars", 0))
    extra_bars += 400
    if extra_bars <= 0:
        return start
    preload_start = start_ts - pd.Timedelta(seconds=bar_seconds * extra_bars)
    return preload_start.strftime("%Y-%m-%d")


def backtest(
    strategy: str = "lightgbm_btc",
    *,
    symbol: str = "BTCUSDT",
    start: str = "2020-01-01",
    end: str = "2025-01-01",
    bar_seconds: int = 3600,
    init_balance: float = 1_000_000.0,
    fee_rate: float = 0.001,
    params: Optional[Dict[str, Any]] = None,
    data_dir: Optional[str] = None,
    exp_name: str = "",
    strategy_cls: Optional[Type] = None,
) -> RunResult:
    if params is None:
        params = {}
    if data_dir is None:
        data_dir = str(DEFAULT_DATA_DIR)
    cfg = defaults()
    cfg["backtest"].update(
        {
            "data_dir": data_dir,
            "symbol": symbol,
            "start": start,
            "end": end,
            "bar_seconds": bar_seconds,
            "init_balance": init_balance,
            "fee_rate": fee_rate,
        }
    )
    cfg["strategy"] = {"name": strategy, "params": params}
    if exp_name:
        cfg.setdefault("experiment", {})["name"] = exp_name
    return run_backtest(cfg, strategy_cls=strategy_cls)


def run_backtest(cfg: Dict[str, Any], strategy_cls: Optional[Type] = None) -> RunResult:
    bt = cfg.get("backtest", {})
    sc = cfg.get("strategy", {})
    oc = cfg.get("output", {})
    ec = cfg.get("experiment", {})

    data_dir = Path(bt.get("data_dir", str(DEFAULT_DATA_DIR)))
    symbol = bt.get("symbol", "BTCUSDT")
    start = bt.get("start", "2020-01-01")
    end = bt.get("end", "2025-01-01")
    bar_seconds = int(bt.get("bar_seconds", 3600))
    init_balance = float(bt.get("init_balance", 1_000_000.0))
    fee_rate = float(bt.get("fee_rate", 0.001))
    min_trade_notional = float(bt.get("min_trade_notional", 0.0))
    min_weight_change = float(bt.get("min_weight_change", 0.0))

    strategy_name = sc.get("name", "lightgbm_btc")
    strategy_params = sc.get("params", {})
    preload_start = _compute_preload_start(start, bar_seconds, strategy_params)
    actual_start_ts = pd.Timestamp(start, tz="UTC")

    exp_root = Path(ec.get("root_dir", PROJECT_ROOT / "experiments"))
    exp_name = ec.get("name", "")
    if not exp_name:
        from datetime import datetime as _dt

        exp_name = _dt.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = exp_root / strategy_name / exp_name
    data_out = exp_dir / "data"
    figs_out = exp_dir / "figs"
    exp_dir.mkdir(parents=True, exist_ok=True)
    data_out.mkdir(exist_ok=True)
    figs_out.mkdir(exist_ok=True)

    with (exp_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    t0 = time.perf_counter()
    try:
        print(f"[data] {symbol}  {start} -> {end}  bar={bar_seconds}s")
        df = load(symbol=symbol, start=preload_start, end=end, bar_seconds=bar_seconds, data_dir=data_dir)
        bars = to_bar_list(df)
        print(f"[data] loaded {len(bars)} bars in {time.perf_counter() - t0:.1f}s")
    except Exception as exc:
        return RunResult(symbol=symbol, strategy=strategy_name, ok=False, error=f"Data: {exc}")

    if len(bars) < 100:
        return RunResult(symbol=symbol, strategy=strategy_name, ok=False, error="Too few bars (<100)")

    if strategy_cls is None:
        strategy_cls = _load_strategy_class(strategy_name)

    from strategies.base import StrategyContext

    strategy = strategy_cls(StrategyContext(symbol=symbol, bar_seconds=bar_seconds, params=strategy_params))
    strategy.on_start()

    signal_frame = strategy.generate_signal_frame(df)
    signal_rows: List[Dict[str, Any]] = []
    if signal_frame is not None:
        signal_frame = signal_frame.copy()
        signal_frame.index = signal_frame.index.map(lambda x: x.isoformat())

    executor = SpotExecutor(
        init_balance=init_balance,
        fee_rate=fee_rate,
        min_notional=min_trade_notional,
        min_weight_change=min_weight_change,
    )
    equity_rows: List[Dict[str, Any]] = []
    bar_records: List[Dict[str, Any]] = []
    t1 = time.perf_counter()

    for bar in bars:
        bar_ts = pd.Timestamp(bar["datetime"])
        target_weight = None
        prediction = ""
        target_return = ""
        model_id = ""

        if signal_frame is not None and bar["datetime"] in signal_frame.index:
            row = signal_frame.loc[bar["datetime"]]
            target_weight = float(row.get("target_weight", 0.0))
            pred_val = row.get("prediction", np.nan) if hasattr(row, "get") else row["prediction"]
            target_val = row.get("target_return", np.nan) if hasattr(row, "get") else row["target_return"]
            prediction = "" if pd.isna(pred_val) else float(pred_val)
            target_return = "" if pd.isna(target_val) else float(target_val)
            model_id = row.get("model_id", "") if hasattr(row, "get") else ""
        else:
            target = strategy.on_bar(bar)
            if target is not None:
                target_weight = float(target)

        close = float(bar["close"])
        if target_weight is not None and bar_ts >= actual_start_ts:
            executor.set_target_weight(symbol, target_weight, close, dt=bar["datetime"])
        executor.mark_to_market(symbol, close)
        if bar_ts < actual_start_ts:
            continue

        eq = executor.get_equity()
        eq["datetime"] = bar["datetime"]
        equity_rows.append(eq)

        signal_rows.append(
            {
                "datetime": bar["datetime"],
                "prediction": prediction,
                "target_return": target_return,
                "target_weight": round(float(target_weight), 6) if target_weight is not None else "",
                "current_weight": round(eq["position_weight"], 6),
                "model_id": model_id,
            }
        )
        if oc.get("save_bars", True):
            bar_records.append(bar)

    strategy.on_finish()
    elapsed = time.perf_counter() - t1
    print(f"[run] strategy={strategy_name} bars={len(bars)} elapsed={elapsed:.1f}s")

    metrics = calc_metrics(equity_rows, bar_seconds)
    trades = executor.get_trades()
    trade_metrics = calc_trade_metrics(trades)
    bucket_metrics = calc_bucketed_metrics(equity_rows, bar_seconds, bucket_days=10)
    rolling_metrics = calc_rolling_window_metrics(equity_rows, bar_records or bars, bar_seconds=bar_seconds, window_days=10, step_days=1)
    competition_report = build_competition_report(
        init_balance=init_balance,
        fee_rate=fee_rate,
        trades=trades,
        equity_rows=equity_rows,
    )

    if oc.get("save_equity", True):
        _write_csv(data_out / "equity_curve.csv", equity_rows)
    if oc.get("save_signal", True):
        _write_csv(data_out / "signals.csv", signal_rows)
    if oc.get("save_bars", True):
        _write_csv(data_out / "bars.csv", bar_records)
    if oc.get("save_trades", True) and trades:
        _write_csv(
            data_out / "trades.csv",
            [
                {
                    "datetime": t.datetime,
                    "symbol": t.symbol,
                    "side": t.side,
                    "price": t.price,
                    "quantity": t.quantity,
                    "notional": t.notional,
                    "fee": t.fee,
                    "pnl": t.pnl,
                    "position_after": t.position_after,
                    "target_weight_after": t.target_weight_after,
                }
                for t in trades
            ],
        )
    if bucket_metrics:
        _write_csv(data_out / "ten_day_metrics.csv", bucket_metrics)
    if rolling_metrics:
        _write_csv(data_out / "rolling_10d_metrics.csv", rolling_metrics)
    if signal_frame is not None and oc.get("save_predictions", True):
        pred_rows = [
            {
                "datetime": idx,
                "prediction": "" if pd.isna(row["prediction"]) else float(row["prediction"]),
                "target_return": "" if pd.isna(row["target_return"]) else float(row["target_return"]),
                "target_weight": float(row["target_weight"]),
                "model_id": row.get("model_id", ""),
            }
            for idx, row in signal_frame.iterrows()
        ]
        _write_csv(data_out / "predictions.csv", pred_rows)

    signal_meta = getattr(strategy, "signal_meta", {}) or {}
    with (exp_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with (exp_dir / "trade_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(trade_metrics, f, indent=2, ensure_ascii=False)
    with (exp_dir / "competition_report.json").open("w", encoding="utf-8") as f:
        json.dump(competition_report, f, indent=2, ensure_ascii=False)
    if signal_meta:
        with (exp_dir / "model_meta.json").open("w", encoding="utf-8") as f:
            json.dump(signal_meta, f, indent=2, ensure_ascii=False)

    try:
        plot_return_drawdown(equity_rows, figs_out / "return_drawdown.png", metrics)
        plot_price_signals(bar_records, signal_rows, figs_out / "price_signals.png")
        plot_monthly_returns(equity_rows, figs_out / "monthly_returns.png")
        if bucket_metrics:
            plot_bucket_metrics(bucket_metrics, figs_out / "ten_day_metrics.png", bucket_days=10)
        if rolling_metrics:
            plot_rolling_window_vs_buy_hold(rolling_metrics, figs_out / "rolling_10d_vs_buy_hold.png", window_days=10)
        if trades:
            plot_trade_analysis(trades, figs_out / "trade_analysis.png")
    except Exception as exc:
        logger.warning("Plot failed: %s", exc)

    if metrics.get("ok"):
        _print_report(strategy_name, symbol, start, end, bar_seconds, metrics, trade_metrics, exp_dir)

    return RunResult(
        symbol=symbol,
        strategy=strategy_name,
        ok=True,
        metrics=metrics,
        trade_metrics=trade_metrics,
        bucket_metrics=bucket_metrics,
        rolling_metrics=rolling_metrics,
        competition_report=competition_report,
        signal_meta=signal_meta,
        exp_dir=str(exp_dir),
    )


def _print_report(name: str, symbol: str, start: str, end: str, bar_seconds: int, metrics: Dict[str, Any], trade_metrics: Dict[str, Any], exp_dir: Path) -> None:
    print(f"\n{'=' * 72}")
    print(f"{name:15s} {symbol:10s} {start} -> {end}  {bar_seconds}s")
    print(f"{'-' * 72}")
    print(f"Portfolio Return  {metrics['portfolio_return'] * 100:+.2f}%")
    print(f"Sharpe Ratio      {metrics['sharpe']:.3f}")
    print(f"Sortino Ratio     {metrics['sortino']:.3f}")
    print(f"Calmar Ratio      {metrics['calmar']:.3f}")
    print(f"Composite Score   {metrics['composite_score']:.4f}")
    print(f"Max Drawdown      {metrics['max_drawdown'] * 100:.2f}%")
    print(f"Trades            {trade_metrics.get('n_trades', 0)}")
    print(f"Total Fee         {trade_metrics.get('total_fee', 0.0):.2f}")
    print(f"Output            {exp_dir}/")
    print(f"{'=' * 72}")

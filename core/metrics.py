"""绩效指标计算."""
from __future__ import annotations

import math
from datetime import datetime
from statistics import median
from typing import Any, Dict, List


def _parse_datetime(dt_value) -> datetime:
    if isinstance(dt_value, datetime):
        return dt_value
    if isinstance(dt_value, (int, float)):
        val = float(dt_value)
        if val > 1e15:
            return datetime.utcfromtimestamp(val / 1e6)
        if val > 1e12:
            return datetime.utcfromtimestamp(val / 1e3)
        return datetime.utcfromtimestamp(val)
    return datetime.fromisoformat(str(dt_value).replace("Z", "+00:00"))


def _safe_annual_return(total_return: float, bars_per_year: float, n_returns: int) -> float:
    if n_returns <= 0:
        return 0.0
    base = 1.0 + total_return
    if base <= 0:
        return -1.0
    return base ** (bars_per_year / n_returns) - 1.0


def calc_metrics(equity_rows: List[Dict[str, Any]], bar_seconds: int = 3600, risk_free_rate: float = 0.0) -> Dict[str, Any]:
    if len(equity_rows) < 2:
        return {"ok": False, "reason": "too_few_points"}

    key = "equity" if "equity" in equity_rows[0] else "balance"
    values = [float(r[key]) for r in equity_rows]
    start = values[0]
    end = values[-1]
    if start <= 0:
        return {"ok": False, "reason": "invalid_start_balance"}

    rets: List[float] = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        rets.append((values[i] - prev) / prev if prev != 0 else 0.0)

    n = len(rets)
    mean_ret = sum(rets) / n if n > 0 else 0.0
    var = sum((r - mean_ret) ** 2 for r in rets) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)

    eq_curve = [v / start for v in values]
    peak = eq_curve[0]
    max_drawdown = 0.0
    for val in eq_curve:
        peak = max(peak, val)
        drawdown = val / peak - 1.0
        max_drawdown = min(max_drawdown, drawdown)

    bars_per_year = 365.25 * 86400 / bar_seconds
    annual_factor = math.sqrt(bars_per_year)
    total_return = end / start - 1.0
    annual_return = _safe_annual_return(total_return, bars_per_year, n)
    annual_volatility = std * annual_factor

    rf_per_bar = risk_free_rate / bars_per_year
    sharpe = ((mean_ret - rf_per_bar) / std * annual_factor) if std > 0 else 0.0
    downside = [r for r in rets if r < 0]
    downside_var = sum(r ** 2 for r in downside) / len(downside) if downside else 0.0
    downside_std = math.sqrt(downside_var)
    sortino = ((mean_ret - rf_per_bar) / downside_std * annual_factor) if downside_std > 0 else 0.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    comp_sharpe = (mean_ret / std) if std > 0 else 0.0
    comp_sortino = (mean_ret / downside_std) if downside_std > 0 else 0.0
    comp_calmar = (mean_ret / abs(max_drawdown)) if max_drawdown != 0 else 0.0
    composite_score = 0.4 * comp_sortino + 0.3 * comp_sharpe + 0.3 * comp_calmar

    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    win_rate = len(wins) / n if n > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")

    return {
        "ok": True,
        "start_balance": start,
        "end_balance": end,
        "portfolio_return": total_return,
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "comp_sharpe": comp_sharpe,
        "comp_sortino": comp_sortino,
        "comp_calmar": comp_calmar,
        "composite_score": composite_score,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "n_bars": len(values),
        "n_trades_approx": len(wins) + len(losses),
    }


def calc_bucketed_metrics(
    equity_rows: List[Dict[str, Any]],
    bar_seconds: int = 3600,
    bucket_days: int = 10,
    risk_free_rate: float = 0.0,
) -> List[Dict[str, Any]]:
    if len(equity_rows) < 2 or bucket_days <= 0:
        return []

    keyed = []
    for row in equity_rows:
        try:
            keyed.append((_parse_datetime(row["datetime"]), row))
        except Exception:
            continue
    if len(keyed) < 2:
        return []

    keyed.sort(key=lambda x: x[0])
    first_dt = keyed[0][0]
    bucket_seconds = bucket_days * 86400
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for dt, row in keyed:
        idx = int((dt - first_dt).total_seconds() // bucket_seconds)
        buckets.setdefault(idx, []).append(row)

    results: List[Dict[str, Any]] = []
    for idx in sorted(buckets):
        rows = buckets[idx]
        if len(rows) < 2:
            continue
        metrics = calc_metrics(rows, bar_seconds=bar_seconds, risk_free_rate=risk_free_rate)
        if not metrics.get("ok"):
            continue
        results.append(
            {
                "bucket_index": idx,
                "window_start": _parse_datetime(rows[0]["datetime"]).isoformat(),
                "window_end": _parse_datetime(rows[-1]["datetime"]).isoformat(),
                "window_days": bucket_days,
                **metrics,
            }
        )
    return results


def summarize_bucket_metrics(bucket_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not bucket_metrics:
        return {
            "bucket_count": 0,
            "bucket_return_mean": 0.0,
            "bucket_return_median": 0.0,
            "bucket_return_std": 0.0,
            "bucket_return_positive_ratio": 0.0,
            "bucket_composite_mean": 0.0,
            "bucket_composite_median": 0.0,
            "bucket_composite_std": 0.0,
            "bucket_composite_min": 0.0,
            "bucket_max_drawdown_worst": 0.0,
        }

    returns = [float(x.get("total_return", 0.0)) for x in bucket_metrics]
    composites = [float(x.get("composite_score", 0.0)) for x in bucket_metrics]
    drawdowns = [float(x.get("max_drawdown", 0.0)) for x in bucket_metrics]

    def _std(vals: List[float]) -> float:
        if len(vals) <= 1:
            return 0.0
        mu = sum(vals) / len(vals)
        return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))

    return {
        "bucket_count": len(bucket_metrics),
        "bucket_return_mean": sum(returns) / len(returns),
        "bucket_return_median": median(returns),
        "bucket_return_std": _std(returns),
        "bucket_return_positive_ratio": sum(1 for x in returns if x > 0) / len(returns),
        "bucket_composite_mean": sum(composites) / len(composites),
        "bucket_composite_median": median(composites),
        "bucket_composite_std": _std(composites),
        "bucket_composite_min": min(composites),
        "bucket_max_drawdown_worst": min(drawdowns),
    }


def calc_rolling_window_metrics(
    equity_rows: List[Dict[str, Any]],
    bar_rows: List[Dict[str, Any]],
    bar_seconds: int = 3600,
    window_days: int = 10,
    step_days: int = 1,
) -> List[Dict[str, Any]]:
    if len(equity_rows) < 2 or len(bar_rows) < 2:
        return []

    n = min(len(equity_rows), len(bar_rows))
    dates = []
    for i in range(n):
        try:
            dates.append(_parse_datetime(equity_rows[i]["datetime"]))
        except Exception:
            return []

    window_seconds = window_days * 86400
    step_seconds = step_days * 86400
    last_start_dt = None
    end_idx = 0
    results: List[Dict[str, Any]] = []

    for start_idx in range(n):
        start_dt = dates[start_idx]
        if last_start_dt is not None and (start_dt - last_start_dt).total_seconds() < step_seconds:
            continue
        if end_idx < start_idx + 1:
            end_idx = start_idx + 1
        while end_idx < n and (dates[end_idx] - start_dt).total_seconds() < window_seconds:
            end_idx += 1
        if end_idx >= n:
            break

        eq_window = equity_rows[start_idx : end_idx + 1]
        metrics = calc_metrics(eq_window, bar_seconds=bar_seconds)
        if not metrics.get("ok"):
            continue
        start_close = float(bar_rows[start_idx]["close"])
        end_close = float(bar_rows[end_idx]["close"])
        buy_hold_return = (end_close / start_close - 1.0) if start_close > 0 else 0.0
        results.append(
            {
                "window_start": start_dt.isoformat(),
                "window_end": dates[end_idx].isoformat(),
                "window_days": window_days,
                "step_days": step_days,
                "buy_hold_return": buy_hold_return,
                "excess_return_vs_buy_hold": metrics["total_return"] - buy_hold_return,
                **metrics,
            }
        )
        last_start_dt = start_dt

    return results


def calc_trade_metrics(trades: list) -> Dict[str, Any]:
    if not trades:
        return {"n_trades": 0, "total_fee": 0.0}

    pnls = [t.pnl for t in trades if getattr(t, "side", "") == "SELL"]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_fee = sum(float(t.fee) for t in trades)

    return {
        "n_trades": len(trades),
        "n_round_trips": len(pnls),
        "win_trades": len(wins),
        "loss_trades": len(losses),
        "win_rate": (len(wins) / len(pnls)) if pnls else 0.0,
        "avg_pnl": (sum(pnls) / len(pnls)) if pnls else 0.0,
        "avg_win_pnl": (sum(wins) / len(wins)) if wins else 0.0,
        "avg_loss_pnl": (sum(losses) / len(losses)) if losses else 0.0,
        "gross_profit": sum(wins),
        "gross_loss": sum(losses),
        "total_fee": total_fee,
    }

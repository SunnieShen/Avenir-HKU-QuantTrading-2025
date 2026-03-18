"""回测可视化."""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _ensure_matplotlib():
    cache_root = Path(__file__).resolve().parent.parent / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _parse_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1e15:
            return datetime.utcfromtimestamp(v / 1e6)
        if v > 1e12:
            return datetime.utcfromtimestamp(v / 1e3)
        return datetime.utcfromtimestamp(v)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _setup_time_axis(ax, dates: List[datetime]) -> None:
    import matplotlib.dates as mdates

    if not dates:
        return
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_fontsize(8)


def plot_return_drawdown(equity_rows: List[Dict[str, Any]], out_png: Path, metrics: Optional[Dict[str, Any]] = None) -> None:
    plt = _ensure_matplotlib()
    if len(equity_rows) < 2:
        return

    dates = [_parse_datetime(r["datetime"]) for r in equity_rows]
    values = [float(r["equity"]) for r in equity_rows]
    start_value = values[0]
    returns_pct = [(v / start_value - 1.0) * 100 for v in values]
    peak = values[0]
    drawdowns = []
    for val in values:
        peak = max(peak, val)
        drawdowns.append((val / peak - 1.0) * 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax1.plot(dates, returns_pct, color="#1565C0", linewidth=1.2, label="Strategy")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_title("Portfolio Return", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Return (%)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)

    if metrics and metrics.get("ok"):
        text = (
            f"Return: {metrics['total_return'] * 100:+.2f}%\n"
            f"Sharpe: {metrics['sharpe']:.3f}\n"
            f"Sortino: {metrics['sortino']:.3f}\n"
            f"Calmar: {metrics['calmar']:.3f}\n"
            f"Composite: {metrics['composite_score']:.4f}"
        )
        ax1.text(
            0.98,
            0.96,
            text,
            transform=ax1.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
        )

    ax2.fill_between(dates, drawdowns, 0, alpha=0.25, color="#C62828")
    ax2.plot(dates, drawdowns, color="#C62828", linewidth=1.0, label="Drawdown")
    ax2.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Time")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=9)
    _setup_time_axis(ax2, dates)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_price_signals(bar_rows: List[Dict[str, Any]], signal_rows: List[Dict[str, Any]], out_png: Path) -> None:
    plt = _ensure_matplotlib()
    if len(bar_rows) < 2 or not signal_rows:
        return

    price_dates = [_parse_datetime(r["datetime"]) for r in bar_rows]
    closes = [float(r["close"]) for r in bar_rows]
    signal_by_dt = {r["datetime"]: r for r in signal_rows}
    weights = []
    predictions = []
    sig_dates = []
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    prev_weight = 0.0
    for row in bar_rows:
        dt = row["datetime"]
        sig = signal_by_dt.get(dt, {})
        weight = float(sig.get("target_weight", 0.0) or 0.0)
        pred = sig.get("prediction", np.nan)
        sig_dates.append(_parse_datetime(dt))
        weights.append(weight)
        predictions.append(float(pred) if pred not in ("", None) and pred == pred else np.nan)
        if weight > prev_weight + 1e-9:
            buy_x.append(_parse_datetime(dt))
            buy_y.append(float(row["close"]))
        elif weight < prev_weight - 1e-9:
            sell_x.append(_parse_datetime(dt))
            sell_y.append(float(row["close"]))
        prev_weight = weight

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    ax1.plot(price_dates, closes, color="#1A1A1A", linewidth=1.0, label="Close")
    if buy_x:
        ax1.scatter(buy_x, buy_y, marker="^", color="#2E7D32", s=28, label="Increase long")
    if sell_x:
        ax1.scatter(sell_x, sell_y, marker="v", color="#C62828", s=28, label="Reduce long")
    ax1.set_title("Price & Position Changes", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)

    ax2.step(sig_dates, weights, where="post", color="#1565C0", linewidth=1.0)
    ax2.fill_between(sig_dates, weights, 0, step="post", alpha=0.25, color="#1565C0")
    ax2.set_title("Target Weight", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Weight")
    ax2.grid(True, alpha=0.25)

    if np.isfinite(np.nanmean(predictions)):
        ax3.plot(sig_dates, predictions, color="#6A1B9A", linewidth=1.0)
        ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax3.set_title("Model Prediction", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Pred")
    ax3.set_xlabel("Time")
    ax3.grid(True, alpha=0.25)
    _setup_time_axis(ax3, sig_dates)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_returns(equity_rows: List[Dict[str, Any]], out_png: Path) -> None:
    plt = _ensure_matplotlib()
    if len(equity_rows) < 3:
        return

    dates = [_parse_datetime(r["datetime"]) for r in equity_rows]
    values = [float(r["equity"]) for r in equity_rows]
    monthly_end = {}
    for dt, val in zip(dates, values):
        monthly_end[(dt.year, dt.month)] = val
    keys = sorted(monthly_end)
    if len(keys) < 2:
        return

    monthly_ret = {}
    for i in range(1, len(keys)):
        prev_k, cur_k = keys[i - 1], keys[i]
        prev_v, cur_v = monthly_end[prev_k], monthly_end[cur_k]
        if prev_v > 0:
            monthly_ret[cur_k] = (cur_v / prev_v - 1.0) * 100
    if not monthly_ret:
        return

    years = sorted({k[0] for k in monthly_ret})
    data = np.full((len(years), 12), np.nan)
    for i, year in enumerate(years):
        for month in range(1, 13):
            if (year, month) in monthly_ret:
                data[i, month - 1] = monthly_ret[(year, month)]

    fig, ax = plt.subplots(figsize=(14, max(3, len(years) * 0.7 + 1)))
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1.0)
    im = ax.imshow(np.ma.masked_invalid(data), cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years])
    ax.set_title("Monthly Returns (%)", fontsize=12, fontweight="bold")
    for i in range(len(years)):
        for j in range(12):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Return (%)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_trade_analysis(trades: list, out_png: Path) -> None:
    plt = _ensure_matplotlib()
    pnls = [float(t.pnl) for t in trades if getattr(t, "side", "") == "SELL"]
    if len(pnls) < 2:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2E7D32" if p > 0 else "#C62828" for p in pnls]
    ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Trade PnL Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Closed Trade #")
    ax.set_ylabel("PnL (USDT)")
    ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_bucket_metrics(bucket_metrics: List[Dict[str, Any]], out_png: Path, bucket_days: int = 10) -> None:
    plt = _ensure_matplotlib()
    if len(bucket_metrics) < 1:
        return
    dates = [_parse_datetime(r["window_start"]) for r in bucket_metrics]
    returns = [float(r.get("total_return", 0.0)) * 100 for r in bucket_metrics]
    sortino = [float(r.get("sortino", 0.0)) for r in bucket_metrics]
    composite = [float(r.get("composite_score", 0.0)) for r in bucket_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.plot(dates, returns, marker="o", linewidth=1.0)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title(f"{bucket_days}-Day Return by Window")
    ax1.grid(True, alpha=0.25)
    _setup_time_axis(ax1, dates)

    ax2.plot(dates, sortino, marker="o", linewidth=1.0, color="#2E7D32")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_title(f"{bucket_days}-Day Sortino by Window")
    ax2.grid(True, alpha=0.25)
    _setup_time_axis(ax2, dates)

    ax3.hist(returns, bins=min(12, max(5, len(returns) // 2)), color="#1565C0", alpha=0.75)
    ax3.axvline(np.mean(returns), color="#C62828", linestyle="--", linewidth=1.0)
    ax3.set_title(f"{bucket_days}-Day Return Distribution")
    ax3.grid(True, alpha=0.25, axis="y")

    ax4.boxplot([returns, composite], labels=["Return %", "Composite"], patch_artist=True)
    ax4.set_title(f"{bucket_days}-Day Metric Distribution")
    ax4.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_return_vs_composite(scan_rows: List[Dict[str, Any]], out_png: Path) -> None:
    plt = _ensure_matplotlib()
    rows = [r for r in scan_rows if r.get("ok") and r.get("metrics_ok")]
    if len(rows) < 2:
        return
    xs = [float(r.get("total_return", 0.0)) * 100 for r in rows]
    ys = [float(r.get("composite_score", 0.0)) for r in rows]
    colors = [float(r.get("max_drawdown", 0.0)) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(xs, ys, c=colors, cmap="RdYlGn", alpha=0.75)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Portfolio Return vs Composite Score")
    ax.set_xlabel("Portfolio Return (%)")
    ax.set_ylabel("Composite Score")
    ax.grid(True, alpha=0.25)
    fig.colorbar(scatter, ax=ax, label="Max Drawdown (%)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ten_day_composite_distribution(scan_rows: List[Dict[str, Any]], out_png: Path, top_n: int = 8) -> None:
    plt = _ensure_matplotlib()
    rows = [r for r in scan_rows if r.get("bucket_composite_values")]
    if not rows:
        return
    top_rows = sorted(rows, key=lambda r: float(r.get("composite_score", 0.0)), reverse=True)[:top_n]
    series = [r["bucket_composite_values"] for r in top_rows]
    labels = [r.get("label", f"top{i+1}") for i, r in enumerate(top_rows)]
    fig, ax = plt.subplots(figsize=(max(10, len(series) * 1.2), 6))
    ax.boxplot(series, labels=labels, patch_artist=True)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("10-Day Composite Score Distribution")
    ax.set_ylabel("Composite Score")
    ax.grid(True, alpha=0.25, axis="y")
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_heatmap(scan_rows: List[Dict[str, Any]], out_png: Path) -> None:
    plt = _ensure_matplotlib()
    rows = [r for r in scan_rows if r.get("ok") and r.get("metrics_ok") and isinstance(r.get("params"), dict)]
    if not rows:
        return
    param_names = sorted({k for r in rows for k in r["params"].keys()})
    varying = [name for name in param_names if len({str(r["params"].get(name)) for r in rows}) > 1]
    if not varying:
        return

    x_name = varying[0]
    y_name = varying[1] if len(varying) > 1 else None
    x_vals = sorted({r["params"][x_name] for r in rows})
    y_vals = sorted({r["params"][y_name] for r in rows}) if y_name else [0]
    data = np.full((len(y_vals), len(x_vals)), np.nan)
    for row in rows:
        xi = x_vals.index(row["params"][x_name])
        yi = y_vals.index(row["params"][y_name]) if y_name else 0
        data[yi, xi] = float(row.get("composite_score", 0.0))

    fig, ax = plt.subplots(figsize=(max(8, len(x_vals) * 1.1), max(4, len(y_vals) * 0.8 + 2)))
    im = ax.imshow(np.ma.masked_invalid(data), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals])
    ax.set_xlabel(x_name)
    if y_name:
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])
        ax.set_ylabel(y_name)
        ax.set_title(f"Composite Score Heatmap: {y_name} vs {x_name}")
    else:
        ax.set_yticks([0])
        ax.set_yticklabels(["score"])
        ax.set_title(f"Composite Score Heatmap by {x_name}")
    fig.colorbar(im, ax=ax, label="Composite Score")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_rolling_window_vs_buy_hold(rolling_rows: List[Dict[str, Any]], out_png: Path, window_days: int = 10) -> None:
    plt = _ensure_matplotlib()
    if len(rolling_rows) < 2:
        return
    dates = [_parse_datetime(r["window_start"]) for r in rolling_rows]
    strat = [float(r.get("total_return", 0.0)) * 100 for r in rolling_rows]
    hold = [float(r.get("buy_hold_return", 0.0)) * 100 for r in rolling_rows]
    excess = [float(r.get("excess_return_vs_buy_hold", 0.0)) * 100 for r in rolling_rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    ax1.plot(dates, strat, label="Strategy", color="#1565C0")
    ax1.plot(dates, hold, label="Buy & Hold", color="#2E7D32")
    ax1.fill_between(dates, excess, 0, alpha=0.2, color="#FFA726", label="Excess")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title(f"Rolling {window_days}-Day Return")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)
    _setup_time_axis(ax1, dates)

    bins = min(20, max(8, len(strat) // 6))
    ax2.hist(hold, bins=bins, alpha=0.55, label="Buy & Hold", color="#2E7D32")
    ax2.hist(strat, bins=bins, alpha=0.55, label="Strategy", color="#1565C0")
    ax2.axvline(np.mean(hold), color="#1B5E20", linestyle="--", linewidth=1.0)
    ax2.axvline(np.mean(strat), color="#0D47A1", linestyle="--", linewidth=1.0)
    ax2.set_title(f"Rolling {window_days}-Day Return Distribution")
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

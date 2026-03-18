"""比赛规则摘要与合规检查."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


def build_competition_report(
    *,
    init_balance: float,
    fee_rate: float,
    trades: list,
    equity_rows: List[Dict[str, Any]],
    expected_init_balance: float = 1_000_000.0,
    taker_fee_rate: float = 0.001,
    maker_fee_rate: float = 0.0005,
) -> Dict[str, Any]:
    daily_trades = defaultdict(int)
    daily_fees = defaultdict(float)
    short_detected = False
    total_notional = 0.0

    for t in trades:
        day = str(t.datetime)[:10]
        daily_trades[day] += 1
        daily_fees[day] += float(t.fee)
        total_notional += float(t.notional)
        if float(getattr(t, "position_after", 0.0)) < 0:
            short_detected = True

    max_trades_per_day = max(daily_trades.values()) if daily_trades else 0
    avg_trades_per_day = sum(daily_trades.values()) / len(daily_trades) if daily_trades else 0.0
    total_fee = sum(float(t.fee) for t in trades)
    final_equity = float(equity_rows[-1]["equity"]) if equity_rows else init_balance

    warnings: List[str] = []
    if short_detected:
        warnings.append("Detected short positions, which violate spot-only competition rules.")
    if abs(init_balance - expected_init_balance) > 1e-9:
        warnings.append("Configured init_balance does not match competition mock portfolio ($1,000,000).")
    if abs(fee_rate - taker_fee_rate) > 1e-12:
        warnings.append("Configured fee_rate does not match competition taker fee (0.1%).")
    if avg_trades_per_day > 24 or max_trades_per_day > 48:
        warnings.append("Trading frequency may be too high for competition API constraints.")

    return {
        "rules": {
            "spot_only": True,
            "no_short_selling": True,
            "no_hft_market_making_arbitrage": True,
            "mock_portfolio_required": expected_init_balance,
            "taker_fee_rate": taker_fee_rate,
            "maker_fee_rate": maker_fee_rate,
            "current_executor_model": "market/taker only",
        },
        "portfolio": {
            "configured_init_balance": init_balance,
            "expected_init_balance": expected_init_balance,
            "init_balance_matches_competition": abs(init_balance - expected_init_balance) <= 1e-9,
            "final_equity": final_equity,
        },
        "fees": {
            "configured_fee_rate": fee_rate,
            "expected_taker_fee_rate": taker_fee_rate,
            "total_fee": total_fee,
            "avg_fee_per_trade": (total_fee / len(trades)) if trades else 0.0,
            "fee_as_pct_of_init_balance": (total_fee / init_balance) if init_balance > 0 else 0.0,
            "estimated_turnover_notional": total_notional,
        },
        "activity": {
            "n_trades": len(trades),
            "avg_trades_per_day": avg_trades_per_day,
            "max_trades_per_day": max_trades_per_day,
            "days_with_trades": len(daily_trades),
        },
        "compliance": {
            "short_detected": short_detected,
            "competition_ready": len(warnings) == 0,
            "warnings": warnings,
        },
    }

"""参数扫描."""
from __future__ import annotations

import csv
import itertools
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from core.metrics import summarize_bucket_metrics
from core.plot import plot_parameter_heatmap, plot_return_vs_composite, plot_ten_day_composite_distribution
from core.runner import run_backtest


def _parse_scalar(text: str) -> Any:
    v = text.strip()
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v


def parse_grid_specs(grid_specs: List[str]) -> Dict[str, List[Any]]:
    grid: Dict[str, List[Any]] = {}
    for item in grid_specs:
        if "=" not in item:
            raise ValueError(f"Invalid grid '{item}', expected key=v1,v2,...")
        key, raw = item.split("=", 1)
        vals = [_parse_scalar(x) for x in raw.split(",") if x.strip()]
        if not vals:
            raise ValueError(f"Grid '{item}' has no values")
        grid[key.strip()] = vals
    if not grid:
        raise ValueError("No grid parameters provided")
    return grid


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_parameter_scan(
    cfg: Dict[str, Any],
    *,
    grid_specs: List[str] | None = None,
    grid_config: Dict[str, List[Any]] | None = None,
    scan_name: str = "",
    top_n: int = 8,
) -> Dict[str, Any]:
    grid = grid_config or parse_grid_specs(grid_specs or [])
    strategy_name = cfg.get("strategy", {}).get("name", "lightgbm_btc")
    base_params = deepcopy(cfg.get("strategy", {}).get("params", {}))

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    exp_root = Path(cfg.get("experiment", {}).get("root_dir", Path(__file__).resolve().parent.parent / "experiments"))
    scan_name = scan_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    scan_dir = exp_root / "_scans" / strategy_name / scan_name
    data_out = scan_dir / "data"
    figs_out = scan_dir / "figs"
    scan_dir.mkdir(parents=True, exist_ok=True)
    data_out.mkdir(exist_ok=True)
    figs_out.mkdir(exist_ok=True)

    with (scan_dir / "scan_config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump({"base_cfg": cfg, "grid": grid}, f, allow_unicode=True, default_flow_style=False)

    rows: List[Dict[str, Any]] = []
    plot_rows: List[Dict[str, Any]] = []
    print(f"[scan] strategy={strategy_name} combinations={len(combos)}")

    for idx, combo in enumerate(combos, start=1):
        params = deepcopy(base_params)
        params.update(dict(zip(keys, combo)))
        run_cfg = deepcopy(cfg)
        run_cfg.setdefault("strategy", {})["params"] = params
        run_cfg.setdefault("experiment", {})["name"] = f"{scan_name}_{idx:03d}"
        print(f"[scan] {idx}/{len(combos)} params={params}")
        result = run_backtest(run_cfg)
        metrics = result.metrics or {}
        trade_metrics = result.trade_metrics or {}
        bucket_metrics = result.bucket_metrics or []
        competition = result.competition_report or {}
        row = {
            "rank": 0,
            "label": ", ".join(f"{k}={params[k]}" for k in keys),
            "ok": result.ok,
            "metrics_ok": bool(metrics.get("ok")),
            "exp_dir": result.exp_dir,
            "n_trades": trade_metrics.get("n_trades", 0),
            "total_fee": trade_metrics.get("total_fee", 0.0),
            "competition_ready": competition.get("compliance", {}).get("competition_ready", False),
            "avg_trades_per_day": competition.get("activity", {}).get("avg_trades_per_day", 0.0),
            **{k: params[k] for k in keys},
            **{k: metrics.get(k, 0.0) for k in ["total_return", "sharpe", "sortino", "calmar", "composite_score", "max_drawdown"]},
            **summarize_bucket_metrics(bucket_metrics),
        }
        rows.append(row)
        plot_rows.append({**row, "params": params, "bucket_composite_values": [float(x.get("composite_score", 0.0)) for x in bucket_metrics]})

    rows.sort(key=lambda r: (bool(r.get("competition_ready")), float(r.get("composite_score", -1e18))), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    _write_csv(data_out / "scan_results.csv", rows)
    with (data_out / "top_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows[:top_n], f, indent=2, ensure_ascii=False)

    plot_return_vs_composite(plot_rows, figs_out / "return_vs_composite.png")
    plot_ten_day_composite_distribution(plot_rows, figs_out / "ten_day_composite_distribution.png", top_n=top_n)
    plot_parameter_heatmap(plot_rows, figs_out / "parameter_heatmap.png")

    return {"scan_dir": str(scan_dir), "rows": rows}

#!/usr/bin/env python3
"""CLI 入口."""
from __future__ import annotations

import argparse
import os
import pkgutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_CACHE_ROOT = Path(__file__).parent / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))


def cmd_run(args):
    from core.config import apply_cli_overrides, defaults, load_config
    from core.runner import run_backtest

    cfg = load_config(args.config) if args.config else defaults()
    cfg = apply_cli_overrides(cfg, args)
    if args.name:
        cfg.setdefault("experiment", {})["name"] = args.name
    result = run_backtest(cfg)
    if not result.ok:
        print(f"ERROR: {result.error}")
        sys.exit(1)


def cmd_list(_args):
    import strategies

    print("Available strategies:\n")
    for _, name, _ in pkgutil.iter_modules([str(Path(strategies.__file__).parent)]):
        if name.startswith("_") or name == "base":
            continue
        print(f"  {name}")


def cmd_scan(args):
    from core.config import apply_cli_overrides, defaults, load_config
    from core.tune import run_parameter_scan

    cfg = load_config(args.config) if args.config else defaults()
    cfg = apply_cli_overrides(cfg, args)
    run_parameter_scan(cfg, grid_specs=args.grid, scan_name=args.name, top_n=args.top)


def main():
    ap = argparse.ArgumentParser(prog="roostoo-bt")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", default="")
    run_parser.add_argument("--strategy", default="")
    run_parser.add_argument("--param", action="append", default=[])
    run_parser.add_argument("--start", default="")
    run_parser.add_argument("--end", default="")
    run_parser.add_argument("--bar", default="")
    run_parser.add_argument("--symbol", default="")
    run_parser.add_argument("--data-dir", default="")
    run_parser.add_argument("--balance", default="")
    run_parser.add_argument("--fee", default="")
    run_parser.add_argument("--name", default="")

    sub.add_parser("list")

    scan_parser = sub.add_parser("scan")
    scan_parser.add_argument("--config", default="")
    scan_parser.add_argument("--strategy", default="")
    scan_parser.add_argument("--param", action="append", default=[])
    scan_parser.add_argument("--grid", action="append", default=[])
    scan_parser.add_argument("--start", default="")
    scan_parser.add_argument("--end", default="")
    scan_parser.add_argument("--bar", default="")
    scan_parser.add_argument("--symbol", default="")
    scan_parser.add_argument("--data-dir", default="")
    scan_parser.add_argument("--balance", default="")
    scan_parser.add_argument("--fee", default="")
    scan_parser.add_argument("--name", default="")
    scan_parser.add_argument("--top", type=int, default=8)

    args = ap.parse_args()
    {"run": cmd_run, "list": cmd_list, "scan": cmd_scan}[args.cmd](args)


if __name__ == "__main__":
    main()

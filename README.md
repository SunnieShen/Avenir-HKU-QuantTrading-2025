# Roostoo BTC Backtest Framework

这个仓库已经从原先偏 `Avenir` 比赛提交脚本的结构，改造成更贴近 `Roostoo` Web3 Quant Hackathon 的离线研究框架。

当前目标是：

- 使用 `data/BTCUSDT/` 下的 Binance 1 分钟数据
- 把原先 Avenir 风格的 `LightGBM` 特征工程迁到单标的 BTC
- 以比赛更关心的交易结果为核心，而不是只输出预测分数
- 输出 `Portfolio Return`、`Sharpe`、`Sortino`、`Calmar`
- 补充固定 10 天窗口与滚动 10 天窗口的 CSV 和图表

## 项目结构

```text
.
├── run.py
├── configs/
│   └── default.yaml
├── core/
│   ├── competition.py
│   ├── config.py
│   ├── data_loader.py
│   ├── executor.py
│   ├── metrics.py
│   ├── plot.py
│   ├── runner.py
│   └── tune.py
├── strategies/
│   ├── base.py
│   ├── lightgbm_btc.py
│   └── sma.py
├── src/
│   └── lightgbm_signal.py
├── docs/
│   ├── overview.md
│   ├── data-and-backtest.md
│   ├── metrics.md
│   └── strategy-development.md
├── data/
│   └── BTCUSDT/
└── experiments/
```

## 核心策略

默认策略是 `lightgbm_btc`：

- 数据：`BTCUSDT` 现货历史 1 分钟 K 线
- 模型：`LightGBMRegressor`
- 训练：walk-forward 滚动训练
- 信号：未来若干 bar 的收益预测
- 执行：把预测值映射为 `0 ~ max_weight` 的现货仓位

主要特征包括：

- 多窗口动量
- EMA 偏离
- 滚动波动率
- Parkinson 波动率
- 成交额变化波动
- 小时 / 星期季节性

## 比赛口径

当前回测默认对齐以下比赛设定：

- 初始资金：`$1,000,000`
- 手续费：`0.1%` taker
- 市场：spot only
- 做空：禁止

主指标：

- `Portfolio Return`
- `Sharpe Ratio`
- `Sortino Ratio`
- `Calmar Ratio`

同时输出：

- `0.4 * Sortino + 0.3 * Sharpe + 0.3 * Calmar`
- 固定 10 天分桶指标
- 滚动 10 天窗口指标

## 安装

```bash
pip install -r requirements.txt
```

## 运行

### 1. 查看策略

```bash
python run.py list
```

### 2. 跑默认配置

```bash
python run.py run --config configs/default.yaml
```

### 3. 自定义参数

```bash
python run.py run \
  --strategy lightgbm_btc \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --bar 3600 \
  --param prediction_horizon_bars=24 \
  --param long_threshold=0.002 \
  --param max_weight=0.9
```

### 4. 参数扫描

```bash
python run.py scan \
  --config configs/default.yaml \
  --strategy lightgbm_btc \
  --grid long_threshold=0.001,0.0015,0.002 \
  --grid max_weight=0.7,0.85,0.95 \
  --name lgbm_scan_smoke
```

## 输出目录

每次回测都会生成：

```text
experiments/lightgbm_btc/<timestamp>/
├── config.yaml
├── metrics.json
├── trade_metrics.json
├── competition_report.json
├── model_meta.json
├── data/
│   ├── bars.csv
│   ├── equity_curve.csv
│   ├── predictions.csv
│   ├── rolling_10d_metrics.csv
│   ├── signals.csv
│   ├── ten_day_metrics.csv
│   └── trades.csv
└── figs/
    ├── monthly_returns.png
    ├── price_signals.png
    ├── return_drawdown.png
    ├── rolling_10d_vs_buy_hold.png
    ├── ten_day_metrics.png
    └── trade_analysis.png
```

## 文档

- `docs/overview.md`
- `docs/data-and-backtest.md`
- `docs/metrics.md`
- `docs/strategy-development.md`

## 说明

- 原始 `src/strategy.py` 仍保留为早期 Avenir 版本的参考脚本
- 当前默认入口已经切换到 `run.py + core/ + strategies/ + src/lightgbm_signal.py`
- 本仓库当前优先支持 `BTCUSDT` 单标的离线研究流程

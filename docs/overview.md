## 项目概览

这个仓库现在面向 `Roostoo` Web3 量化比赛的离线研究流程，重点是：

- 使用 `data/BTCUSDT/` 下的 Binance 1 分钟历史数据
- 将原先偏 `Avenir` 风格的 `LightGBM` 特征工程迁移到单标的 BTC
- 以现货、手续费、`$1,000,000` 初始资金做回测
- 输出比赛重点关注的 `Portfolio Return`、`Sharpe`、`Sortino`、`Calmar`
- 补充固定 10 天窗口与滚动 10 天窗口分布

## 当前策略

默认策略为 `lightgbm_btc`：

- 输入：多窗口动量、EMA 偏离、波动率、Parkinson 波动率、成交额变化、季节性
- 模型：`LightGBMRegressor`
- 训练方式：walk-forward 滚动训练与预测
- 执行方式：根据预测收益映射为 `0 ~ max_weight` 的现货仓位

## 结果目录

每次运行会输出到：

`experiments/<strategy>/<timestamp>/`

包含：

- `config.yaml`
- `metrics.json`
- `trade_metrics.json`
- `competition_report.json`
- `model_meta.json`
- `data/`
- `figs/`

## 比赛指标

回测输出对齐比赛说明页中的核心指标：

- `Portfolio Return`
- `Sharpe Ratio`
- `Sortino Ratio`
- `Calmar Ratio`

同时输出组合分数：

`0.4 * Sortino + 0.3 * Sharpe + 0.3 * Calmar`

## 10 天窗口分析

为了贴近比赛 10 天实盘周期，框架会额外生成：

- `data/ten_day_metrics.csv`
  - 固定 10 天分桶指标
- `data/rolling_10d_metrics.csv`
  - 按 1 天步长滚动的 10 天窗口指标

对应图表：

- `figs/ten_day_metrics.png`
- `figs/rolling_10d_vs_buy_hold.png`

## 合规输出

`competition_report.json` 会给出：

- 初始资金是否匹配 `$1,000,000`
- 手续费是否匹配 `0.1%` taker
- 是否出现做空
- 平均每日交易次数
- 是否满足 `competition_ready`

## 数据口径

当前离线回测使用：

- 目录：`data/BTCUSDT/`
- 文件格式：Binance 1 分钟 K 线 CSV
- 时间戳：自动兼容毫秒与微秒
- 重采样：默认聚合为 `1h` bar，可通过 `--bar` 调整

## 回测假设

- 市场：现货 BTC
- 初始资金：`$1,000,000`
- 默认手续费：`0.1%` taker
- 做空：禁止
- 仓位：目标仓位权重模式，`0.0` 为空仓，`1.0` 为满仓

## 运行示例

```bash
python run.py run --config configs/default.yaml
```

```bash
python run.py run \
  --strategy lightgbm_btc \
  --start 2023-01-01 \
  --end 2024-12-31 \
  --bar 3600 \
  --param long_threshold=0.002 \
  --param max_weight=0.9
```

## 参数扫描示例

```bash
python run.py scan \
  --config configs/default.yaml \
  --strategy lightgbm_btc \
  --grid long_threshold=0.001,0.0015,0.002 \
  --grid max_weight=0.7,0.85,0.95 \
  --name lgbm_scan_smoke
```

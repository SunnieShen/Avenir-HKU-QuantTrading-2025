## 策略开发

策略统一放在 `strategies/`：

- `strategies/base.py`：基类
- `strategies/sma.py`：规则型示例
- `strategies/lightgbm_btc.py`：机器学习策略

## 两种策略接口

### 1. 逐 bar 规则策略

实现 `on_bar(bar)`，返回目标仓位权重：

- `1.0`：满仓
- `0.0`：空仓
- `None`：保持不变

### 2. 批量 ML 策略

实现 `generate_signal_frame(df)`，返回带以下列的 `DataFrame`：

- `prediction`
- `target_return`
- `target_weight`
- `model_id`

回测器会按这些列直接驱动执行和落盘。

## LightGBM 策略说明

`src/lightgbm_signal.py` 负责：

- 特征构造
- 标签生成
- walk-forward 训练
- 预测映射为目标仓位

如果要继续向 Avenir 风格靠拢，建议优先在这里扩展：

- 特征窗口
- 标签 horizon
- 训练窗口长度
- 重训练频率
- 仓位映射阈值

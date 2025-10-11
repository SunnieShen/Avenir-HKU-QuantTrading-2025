# Avenir–HKU Web3.0 Quantitative Trading Challenge 2025

**Organizer:** Avenir x The University of Hong Kong  
**Objective:** Build a trading signal model that predicts future 24-hour returns of multiple crypto pairs based on high-frequency OHLCV data.

### Data Overview
- **Format:** Parquet files per symbol (`BTCUSDT.parquet`, etc.)  
- **Fields:** timestamp, open_price, high_price, low_price, close_price, volume, amount  
- **Evaluation Metric:** Weighted Spearman correlation (mean across timestamps)  
- **Submission Format:** `id, predict_return` where `id = <timestamp>_<symbol>`

### Restrictions
All data files are proprietary and may only be used within the scope of the competition.
Participants are not permitted to redistribute or publicly upload the original or derived data.

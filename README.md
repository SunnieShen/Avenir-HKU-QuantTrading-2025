# Avenir–HKU Web3.0 Quantitative Trading Challenge 2025  
This repository contains the code I developed and submitted for the **Avenir–HKU Web3.0 Quantitative Trading Challenge 2025**, organized by Avenir and the University of Hong Kong.  
The competition focuses on **algorithmic trading signal generation** using crypto K-line (OHLCV) data, where participants predict 24-hour future returns across hundreds of symbols.

## Overview

This project implements a **LightGBM-based rank-learning model** (`lambdarank`) for predicting next-day returns using engineered time-series features such as moving averages, momentum, and volatility proxies.  
All data preprocessing, feature generation, and model training steps are automated through the `OlsModel` class.

## Key Features

| Module | Description |
|--------|-------------|
| **Data Handling** | Reads `.parquet` OHLCV data files for multiple trading symbols; merges them into a synchronized time grid. |
| **Feature Engineering** | Computes multi-window momentum, EMA gaps, volatility proxies, and intraday seasonality (hour/day of week). |
| **Caching System** | Uses NumPy `.npz` caching for large multi-symbol arrays to speed up reloads. |
| **Model** | LightGBM `lambdarank` objective optimized with NDCG and early stopping. |
| **Evaluation Metric** | Weighted Spearman correlation, aligned with competition’s official scoring. |
| **Parallel Loading** | Utilizes multi-threading for efficient parquet file reading. |
| **Submission Generator** | Automatically produces `submit.csv` and `check.csv` following Avenir’s competition schema. |

## Installation

```bash
git clone https://github.com/<your-username>/AvenirHKU_QuantTrading.git
cd AvenirHKU_QuantTrading
pip install -r requirements.txt
```

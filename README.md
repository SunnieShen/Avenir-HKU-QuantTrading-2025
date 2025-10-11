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

## Model summary
| Parameter             | Value                         | Description                                        |
| --------------------- | ----------------------------- | -------------------------------------------------- |
| **Objective**         | `lambdarank`                  | Rank-based objective for ordered return prediction |
| **Metric**            | `ndcg@5,10,20`                | Normalized Discounted Cumulative Gain for ranking  |
| **Learning Rate**     | `0.05`                        | Step size for gradient boosting                    |
| **Num Leaves**        | `63`                          | Controls model complexity                          |
| **Feature Fraction**  | `0.9`                         | Random subset of features per tree                 |
| **Bagging Fraction**  | `0.8`                         | Row subsampling per iteration                      |
| **Early Stopping**    | `200 rounds`                  | Stop if no improvement                             |
| **Validation Window** | `Last 60 days`                | Validation split by time                           |
| **Evaluation Metric** | Weighted Spearman Correlation | Official Avenir Challenge metric                   |

## Feature Engineering 
| Feature Group                     | Description                                |
| --------------------------------- | ------------------------------------------ |
| **Momentum (`mom_w`)**            | Rate of price change over window `w`       |
| **EMA Gap (`emagap_w`)**          | Deviation from exponential moving average  |
| **Volatility Proxy (`vol_w`)**    | Rolling standard deviation of returns      |
| **Parkinson Volatility (`pk_w`)** | Log-range volatility estimator             |
| **Turnover Std (`tostd_w`)**      | Standard deviation of turnover changes     |
| **Cross-sectional Ranks**         | Ranks momentum and EMA gap across assets   |
| **Seasonality**                   | Adds hourly and weekday sinusoidal signals |
Windows used: ```[8, 24, 48, 96, 288, 672, 1344]```

## Model Architecture
**1. Data Preprocessing**
- Reads all .parquet files in /train_data/ (one per crypto pair)
- Aligns timestamps using a reference symbol
- Builds a multi-asset panel of OHLCV and VWAP data

**2. Feature Construction**
- Calculates momentum, EMA deviations, volatility, and seasonality
- Normalizes features across assets (demeaning, z-scoring, ranking)
- Caches feature matrices as NumPy arrays for efficient access

**3. Target Generation**
- Computes 24-hour forward returns (future VWAP / current VWAP - 1)
- Aligns and shifts targets by 4×24 steps (representing 1 day)

**4. Training Pipeline**
- Splits training/validation by time
- Applies time-decay weighting to recent samples
- Trains LightGBM lambdarank with grouped ranking objective
- Early stopping based on NDCG score

**5. Inference & Submission**
- Loads submission_id.csv (competition test IDs)
- Extracts matching features per timestamp/symbol
- Streams predictions into submit.csv for official submission

## Installation

```bash
git clone https://github.com/<your-username>/AvenirHKU_QuantTrading.git
cd AvenirHKU_QuantTrading
pip install -r requirements.txt
```

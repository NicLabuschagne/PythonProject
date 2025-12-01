# Python Feature Engineering Toolkit for Time-Series Analysis

A specialized Python toolkit for **crypto data manipulation**, utilizing **Polars** for data processing. This package provides methods for securely fetching data from **Binance**, generating time-series features (like lags and log returns), and preparing the data for an **Autoregressive (AR) Model**.

---

## Project Context: AR Model Preparation

The primary goal of the features in this toolkit is to create a robust, high-quality dataset suitable for training a **Time-Series Forecasting Model**.

* **Source:** Data is sourced from the Binance exchange API (e.g., BTC/USDT pairs).
* **Target Model:** The prepared features (especially the **lag** variables) are specifically designed for use in a subsequent **Autoregressive (AR) Model** to predict future price movements.
* **Scope:** This repository focuses strictly on **Data Extraction and Feature Engineering**. The training and evaluation of the forecasting model are handled in a separate pipeline/script.

---

##  Features

This modules core logic, primarily in `src/features.py`, focuses on model-ready data creation:

* **Data Ingestion:** Dedicated functions for securely connecting to the **Binance API** and extracting OHLCV (Open, High, Low, Close, Volume) data.
* **Polars Data Prep:** Core data manipulation functions optimized for **Polars DataFrames** for superior performance and memory efficiency.
* **Time-Series Features:** Automated generation of essential time-series features, including **Log Returns** and **Lagged Features** necessary for Autoregressive (AR) modeling.
* **Logging:** Custom log configuration to track data flow, API calls, and detailed feature creation steps.

---

## Visualizations

This modules core logic, primarily in `src/visualization.py` focuses on displaying feature distributions,
and a correlation matrix to the target variable.

* **Seaborn Visualization:** Function related to the plotting the distribution pair-wise for the data set.
* **Correlation Matrix:** Function related to display the absolute correlation from all variables to the target.

---
## Model Development

This modules core logic, primarily in `src/model.py` utilizes the Pytorch libray to forecast log returns.
Two models were created, primarily the linear module, which was used in analysis for easier interoperability, while adding 
a neural network model to compare results.

Lastly, the performance summary is displayed in a data frame with key evaluation metrics. 

---


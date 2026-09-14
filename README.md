# Power Demand Forecasting in KSE (PSE)


A Data Science and Time Series Forecasting project dedicated to short-term electrical power demand forecasting in the Polish Power System (KSE), using actual transmission data from the Polish Power Grid (PSE).

> Detailed business findings and implementation recommendations: [BUSINESS_REPORT.md](BUSINESS_REPORT.md)

---

## Business Goal & Key Findings

* **Business Goal**: Accurately predict short-term power demand (1-3 days ahead) in KSE to optimize conventional power plant reserve scheduling and manage price risk on the balancing market.
* **Best Model**: **XGBoost ML Regressor** achieved a **MAPE of 1.59%** (MAE = 253.58 MW), reducing forecast error by **-91.6%** compared to the seasonal naive benchmark.
* **Key Drivers**: Consumer operational inertia (lag 1 hour) and weekly cycle profiles (lag 168 hours).

---

## Project Structure

```text
PowerDemandForecasting/
├── data/
│   ├── raw/
│   │   └── pse_energy_data.csv                    # Raw data from PSE API
│   └── processed/
│       ├── pse_hourly_data.csv                    # 1h resampled time series
│       └── future_forecast_48h.csv                # XGBoost generated predictions
├── notebooks/
│   ├── 01_data_acquisition_and_eda.ipynb          # EDA, Duck Curve, STL decomposition, ADF/KPSS tests
│   ├── 02_sarimax_modeling.ipynb                  # SARIMAX with RES variables + residual diagnostics
│   ├── 03_xgboost_modeling.ipynb                  # Time feature engineering + XGBoost + TimeSeriesSplit
│   └── 04_model_comparison_and_insights.ipynb     # Model comparison and business decision analysis
├── reports/
│   ├── executive_summary.md                       # Executive summary and findings
│   └── figures/                                   # High-resolution plots (PNG)
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py                            # Data retrieval from PSE API
│   ├── future_forecast.py                         # 48h ahead operational forecast generator
│   ├── time_series_utils.py                       # Statistical tests and metrics
│   └── visualization.py                           # Diagnostic and decomposition plots
├── scripts/
│   └── build_notebooks.py                         # Script to execute notebooks
├── BUSINESS_REPORT.md                             # Full business report and case study
├── README.md
└── requirements.txt
```

---

## Author

**Marcin**
* GitHub: [@MarcinGoo](https://github.com/MarcinGoo)

# Power Demand Forecasting in KSE (PSE)

[![pmdarima](https://img.shields.io/badge/pmdarima-2.1-red.svg)](https://alkaline-ml.com/pmdarima/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-yellow.svg)](https://xgboost.readthedocs.io/)
[![Business Report](https://img.shields.io/badge/Business_Report%20%26%20ROI-brightgreen.svg)](BUSINESS_REPORT.md)

A business-oriented Data Science & Time Series Forecasting project dedicated to short-term power demand forecasting in the National Power System (KSE) based on data from the Polish Power Grid (PSE).

> Detailed business report with ROI analysis and implementation recommendations: [BUSINESS_REPORT.md](BUSINESS_REPORT.md)

---

## Business Problem and Decision Goal

For the transmission system operator (PSE) and energy trading companies, an accurate demand forecast in a 1-3 day horizon (Day-Ahead / Multi-Day Forecast) is crucial. Because renewable energy generation (solar and wind) can be treated as a given constant in short-term planning, forecasting total gross demand allows operators to precisely determine the net load that must be covered by conventional energy sources. 

1. **Optimization of conventional sources**: Accurate forecasting reduces the standby costs of thermal power plants and lowers CO2 emissions.
2. **Balancing RES generation**: The growth in photovoltaic capacity creates a daily net load valley (Duck Curve), requiring high flexibility from balancing units to cover sharp evening peaks.
3. **Risk pricing in the energy market**: Demand forecasting minimizes risk exposure on the Balancing Market for energy traders.

---

## Model Results and Benchmarking (Test Horizon = 72 Hours)

* **XGBoost ML Regressor** proved to be the best model, reducing the forecast error by **-91.6%** compared to the naive baseline.
* **MAPE**: 1.59% (Mean Absolute Error = 253.58 MW)
* **Model Dynamics**: Captured non-linear relationships such as consumer inertia (lag 1 hour) and weekly calendar profiles (lag 168 hours).

---

## Project Structure

```text
CenyEnergi/
├── data/
│   ├── raw/
│   │   └── pse_energy_data.csv                    # Raw data from PSE API
│   └── processed/
│       ├── pse_hourly_data.csv                    # 1h resampled time series
│       └── future_forecast_48h.csv                # XGBoost generated predictions
├── notebooks/
│   ├── 01_data_acquisition_and_eda.ipynb          # EDA, Duck Curve, STL decomposition, ADF/KPSS tests
│   ├── 02_sarimax_modeling.ipynb                  # SARIMAX with RES variables + full residual diagnostics
│   ├── 03_xgboost_modeling.ipynb                  # Time feature engineering + XGBoost + TimeSeriesSplit
│   └── 04_model_comparison_and_insights.ipynb     # Model comparison, 48h ahead prediction and business decisions
├── reports/
│   ├── executive_summary.md                       # Executive summary and ROI analysis
│   └── figures/                                   # High-resolution plots (PNG)
│       ├── 01_duck_curve_profile.png
│       ├── 02_stl_decomposition.png
│       ├── 03_sarimax_residuals_diagnostics.png
│       ├── 04_xgboost_feature_importance.png
│       ├── 05_multi_model_forecast_comparison.png
│       ├── 06_hourly_error_distribution.png
│       └── 07_future_forecast_48h.png
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py                            # Fetching data from PSE API
│   ├── future_forecast.py                         # Generator for operational 48h ahead forecast
│   ├── time_series_utils.py                       # Statistical tests and metrics
│   └── visualization.py                           # Diagnostic and decomposition plots
├── scripts/
│   └── build_notebooks.py                         # Script to execute notebooks
├── BUSINESS_REPORT.md                             # Full business report and case study
├── README.md
└── requirements.txt
```

---

## Quick Start and Generating a Forecast

```bash
# 1. Clone the repository
git clone https://github.com/MarcinGoo/CenyEnergi.git
cd CenyEnergi

# 2. Activate environment and install packages
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Generate a new forecast for the next 48 hours
python src/future_forecast.py

# 4. Run interactive notebooks
jupyter lab
```

---

## Author

**Marcin**
* GitHub: [@MarcinGoo](https://github.com/MarcinGoo)
* Project prepared as part of a Data Science & Energy Analytics portfolio.

# BUSINESS REPORT: Power Demand Forecasting in KSE (PSE)

---

## 1. Business Goal

The primary business objective of this project is to accurately predict short-term electrical power demand (1-3 days ahead) in the National Power System (KSE) based on real data from the Polish Power Grid (PSE). 

Accurate power demand forecasting is the foundation of electricity market economics. It directly addresses key operational challenges:
* **Spinning Reserve Optimization**: Every megawatt of forecast error requires the activation of expensive intervention reserves or maintaining oversized spinning reserves in coal units. Accurate predictions help reduce these standby costs and lower CO2 emissions.
* **Balancing RES Generation**: The growth in photovoltaic capacity creates a daily net load valley (the "Duck Curve"), requiring high flexibility from balancing units to handle sharp net load peaks in the evening.
* **Price Risk Management**: Precise models allow energy trading companies to optimize volumes contracted on the Day-Ahead Market, minimizing their exposure to volatile Balancing Market prices.

---

## 2. Analysis Achievements and Results

The analysis and modeling phases yielded highly accurate predictions, significantly outperforming traditional linear models.

### Key Performance Metrics (72h Out-of-Sample Test)
* **XGBoost ML Regressor** emerged as the best-performing model, achieving a **MAPE error of 1.59%** (Mean Absolute Error of **253.58 MW**).
* **High fit coefficient**: $R^2 = 0.9854$.
* **Prediction accuracy of change direction**: MDA = 78.87%.
* **Forecast Error Reduction**: The XGBoost model reduced the forecast error by **-91.6%** compared to the Seasonal Naive benchmark.

### Analytical Insights
* **Non-linear time relationships**: The Machine Learning model successfully captured complex interactions between the time of day, the day of the week, and the dynamics of industry startup on Monday mornings.
* **Key predictive drivers**: 
  - Thermodynamic and operational inertia of consumers (lag 1 hour) proved to be the most impactful feature (43.12% impact).
  - The profile of the same day in the previous week (lag 168 hours) was the second most important driver (31.97% impact).

### Financial Impact
Reducing the uncertainty buffer by over 2,000 MW (the difference between the benchmark error and the XGBoost error) allows unnecessary conventional units operating at the technical minimum to be shut down. For a trading portfolio of 1,000 MW, reducing MAPE to 1.5% means a significant drop in the unbalanced volume, potentially reducing balancing costs by millions of zlotys annually.

---

## 3. Next Steps and Implementation Recommendations

To leverage these achievements, the solution should be transitioned into a production MLOps environment. 

### Operational Module
An operational forecasting module has already been built to generate load trajectories for the next 48 hours ahead with a 95% uncertainty interval. This module highlights critical hours (e.g., morning peaks) where maintaining an additional buffer is recommended.

### Recommended Architecture
For full production deployment, the following architecture and steps are recommended:
1. **Automated Data Pipeline**: Implement an Airflow data pipeline for automatic data retrieval every 15 minutes from the PSE API.
2. **Preprocessing & Feature Engineering**: Automate the 1-hour resampling, lag creation, and rolling window calculations.
3. **Cyclic Retraining**: Schedule weekly model retraining in a Rolling Window with TimeSeriesSplit validation to adapt to changing seasonal behaviors.
4. **Drift Monitoring**: Implement real-time tracking of the residual distribution using the Ljung-Box test and the MAPE metric to detect model degradation early.
5. **Integration**: Connect the 48h forecast and confidence intervals directly to the SCADA System or Trading Dashboard for real-time decision-making.

---

**Report Author:** Marcin (@MarcinGoo)
*Portfolio Project: Data Science & Energy Analytics*

"""
Operational future forecasting module (48-hour Day-Ahead / Multi-Day horizon).
Trains the final XGBoost model on full historical data and generates recursive forecast trajectory.
"""

import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor
from src.time_series_utils import create_time_features

def generate_future_forecast(
    data_path: str = 'data/processed/pse_hourly_data.csv',
    output_csv_path: str = 'data/processed/future_forecast_48h.csv',
    output_fig_path: str = 'reports/figures/07_future_forecast_48h.png',
    horizon_hours: int = 48
) -> pd.DataFrame:
    """
    Generates an operational forecast for the next `horizon_hours` beyond the last historical timestamp.
    """
    df = pd.read_csv(data_path, index_col='dtime', parse_dates=True)
    
    # Feature engineering
    df_feat = create_time_features(df, target_col='demand', lags=[1, 2, 3, 24, 48, 168], rolling_windows=[6, 24, 168]).dropna()
    feature_cols = [c for c in df_feat.columns if c not in ['demand', 'demand_diff1', 'demand_diff_seasonal', 'demand_diff_both']]
    
    X_all = df_feat[feature_cols]
    y_all = df_feat['demand']
    
    # Fit final model on all historical observations
    model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5, subsample=0.85, colsample_bytree=0.85, random_state=42)
    model.fit(X_all, y_all)
    
    # Recursive multi-step forecasting
    history_df = df.copy()
    future_records = []
    
    for step in range(1, horizon_hours + 1):
        next_dt = history_df.index.max() + pd.Timedelta(hours=1)
        temp_df = history_df.copy()
        temp_df.loc[next_dt] = np.nan
        
        # RES estimation from lag 24h as diurnal baseline
        if 'pv' in temp_df.columns:
            temp_df.loc[next_dt, 'pv'] = temp_df.loc[next_dt - pd.Timedelta(hours=24), 'pv']
        if 'wi' in temp_df.columns:
            temp_df.loc[next_dt, 'wi'] = temp_df.loc[next_dt - pd.Timedelta(hours=24), 'wi']
            
        feat_df = create_time_features(temp_df, target_col='demand', lags=[1, 2, 3, 24, 48, 168], rolling_windows=[6, 24, 168])
        x_next = feat_df.loc[[next_dt], feature_cols]
        pred_val = float(model.predict(x_next)[0])
        
        history_df.loc[next_dt, 'demand'] = pred_val
        if 'pv' in history_df.columns:
            history_df.loc[next_dt, 'pv'] = temp_df.loc[next_dt, 'pv']
        if 'wi' in history_df.columns:
            history_df.loc[next_dt, 'wi'] = temp_df.loc[next_dt, 'wi']
            
        # 95% Confidence bounds based on test MAE (~253.58 MW)
        margin = 1.96 * 253.58
        future_records.append({
            'timestamp': next_dt,
            'forecast_demand_mw': round(pred_val, 1),
            'lower_bound_95': round(pred_val - margin, 1),
            'upper_bound_95': round(pred_val + margin, 1)
        })
        
    df_future = pd.DataFrame(future_records).set_index('timestamp')
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_future.to_csv(output_csv_path)
    
    # Visualization of future trajectory
    plt.figure(figsize=(14, 6))
    last_hist_points = min(120, len(df))
    plt.plot(df.index[-last_hist_points:], df['demand'].iloc[-last_hist_points:], label='Historical Demand (Last 5 Days)', color='#1f77b4', lw=1.8)
    plt.plot(df_future.index, df_future['forecast_demand_mw'], label='XGBoost Model Forecast (Next 48h)', color='#d62728', lw=2.2, linestyle='--', marker='o', markersize=3)
    plt.fill_between(df_future.index, df_future['lower_bound_95'], df_future['upper_bound_95'], color='#d62728', alpha=0.18, label='95% Uncertainty Interval (±497 MW)')
    plt.axvline(df.index.max(), color='black', linestyle=':', lw=1.5, label='Forecast Origin (Present Time)')
    
    plt.title('Operational Power Demand Forecast for the Next 48 Hours (XGBoost ML)', fontweight='bold', fontsize=12)
    plt.xlabel('Timestamp')
    plt.ylabel('Electricity Demand [MW]')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
    plt.savefig(output_fig_path, dpi=150)
    plt.close()
    
    return df_future

if __name__ == '__main__':
    df_res = generate_future_forecast()
    print("48h operational future forecast successfully generated and saved.")

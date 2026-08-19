"""
Visualization module for time series analysis, diagnostic checking, and model forecasting.
All figures are formatted with English labels, adjusted margins, and publication styling.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_time_series_overview(df: pd.DataFrame, figsize=(14, 8)):
    """
    Plots primary time series: total demand, net demand, renewable generation, and conventional units.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # 1. Total and Net Demand
    axes[0].plot(df.index, df['demand'], label='Total Electricity Demand', color='#1f77b4', lw=1.5)
    if 'net_demand' in df.columns:
        axes[0].plot(df.index, df['net_demand'], label='Net Demand (Duck Curve)', color='#ff7f0e', lw=1.2, alpha=0.85)
    axes[0].set_title('National Electricity Demand [MW]', fontweight='bold')
    axes[0].set_ylabel('Power [MW]')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Renewable Generation (Solar PV and Wind)
    if 'pv' in df.columns and 'wi' in df.columns:
        axes[1].plot(df.index, df['pv'], label='Solar PV Generation', color='#e377c2', lw=1.2)
        axes[1].plot(df.index, df['wi'], label='Wind Generation', color='#2ca02c', lw=1.2)
        axes[1].set_title('Renewable Energy Sources (RES) Generation [MW]', fontweight='bold')
        axes[1].set_ylabel('Power [MW]')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Conventional Generation Units (JG)
    if 'jg' in df.columns:
        axes[2].plot(df.index, df['jg'], label='Conventional Generation Units (JG)', color='#7f7f7f', lw=1.2)
        axes[2].set_title('Centrally Dispatched Conventional Generation [MW]', fontweight='bold')
        axes[2].set_ylabel('Power [MW]')
        axes[2].set_xlabel('Timestamp')
        axes[2].legend(loc='upper right')
        axes[2].grid(True, linestyle='--', alpha=0.6)

    axes[2].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axes[2].get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    return fig


def plot_decomposition(decomp_result, title='STL Decomposition of Power Demand (Period s = 24h)', figsize=(14, 9)):
    """
    Plots 4-panel STL decomposition (Observed, Trend, Seasonal, Residual) with clean timestamp axes.
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    axes[0].plot(decomp_result.observed, color='#1f77b4', lw=1.3)
    axes[0].set_ylabel('Observed [MW]')
    axes[0].set_title(title, fontweight='bold', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(decomp_result.trend, color='#d62728', lw=1.8)
    axes[1].set_ylabel('Trend [MW]')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(decomp_result.seasonal, color='#2ca02c', lw=1.1)
    axes[2].set_ylabel('Seasonal [MW]')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    axes[3].scatter(decomp_result.resid.index, decomp_result.resid, color='#9467bd', s=10, alpha=0.7)
    axes[3].axhline(0, color='black', linestyle='--', lw=1)
    axes[3].set_ylabel('Residual [MW]')
    axes[3].set_xlabel('Timestamp')
    axes[3].grid(True, linestyle='--', alpha=0.6)

    axes[3].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axes[3].get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    return fig


def plot_acf_pacf_custom(series: pd.Series, lags: int = 48, title: str = 'Autocorrelation Analysis (ACF & PACF)', figsize=(14, 5)):
    """
    Plots ACF and PACF side by side with English labels.
    """
    clean = series.dropna()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_acf(clean, lags=lags, ax=axes[0], color='#1f77b4', alpha=0.05)
    axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold')
    axes[0].set_xlabel('Lag (Hours)')
    axes[0].set_ylabel('ACF Coefficient')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    plot_pacf(clean, lags=lags, ax=axes[1], color='#2ca02c', alpha=0.05, method='ywm')
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold')
    axes[1].set_xlabel('Lag (Hours)')
    axes[1].set_ylabel('PACF Coefficient')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_residual_diagnostics(residuals: pd.Series, model_name: str = 'SARIMAX', figsize=(14, 9.5)):
    """
    Generates a 4-panel diagnostic plot for model residuals:
    1. Standardized Residuals vs Time
    2. Residual Distribution vs Normal N(0,1)
    3. Normal Q-Q Plot
    4. Residual Autocorrelation (ACF)
    """
    res = residuals.dropna()
    mean_val = np.mean(res)
    std_val = np.std(res)
    standardized = (res - mean_val) / std_val if std_val > 0 else res

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Standardized residuals over time
    axes[0, 0].plot(res.index, standardized, color='#1f77b4', lw=1, alpha=0.85)
    axes[0, 0].axhline(0, color='red', linestyle='--', lw=1.2)
    axes[0, 0].axhline(2, color='orange', linestyle=':', lw=1)
    axes[0, 0].axhline(-2, color='orange', linestyle=':', lw=1)
    axes[0, 0].set_title('1. Standardized Residuals vs Time', fontweight='bold')
    axes[0, 0].set_ylabel('Standardized Residuals')
    axes[0, 0].set_xlabel('Timestamp')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    
    axes[0, 0].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    plt.setp(axes[0, 0].get_xticklabels(), rotation=30, ha='right')

    # 2. Histogram + KDE vs N(0,1)
    sns.histplot(standardized, kde=True, ax=axes[0, 1], stat='density', color='#17becf', alpha=0.4, label='Residual KDE')
    x_range = np.linspace(-4, 4, 200)
    axes[0, 1].plot(x_range, stats.norm.pdf(x_range, 0, 1), 'r--', lw=1.8, label='Normal Distribution N(0,1)')
    axes[0, 1].set_title('2. Residual Distribution vs Normal N(0,1)', fontweight='bold')
    axes[0, 1].set_xlabel('Standardized Residual Value')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].legend(loc='upper right', fontsize=9)
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 3. Normal Q-Q plot
    stats.probplot(standardized, dist='norm', plot=axes[1, 0])
    axes[1, 0].get_lines()[0].set_markerfacecolor('#9467bd')
    axes[1, 0].get_lines()[0].set_markersize(4.0)
    axes[1, 0].get_lines()[1].set_color('red')
    axes[1, 0].get_lines()[1].set_linewidth(1.5)
    axes[1, 0].set_title('3. Normal Q-Q Plot', fontweight='bold')
    axes[1, 0].set_xlabel('Theoretical Quantiles')
    axes[1, 0].set_ylabel('Sample Quantiles')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # 4. Residual ACF
    plot_acf(res, lags=min(36, len(res)//3), ax=axes[1, 1], color='#2ca02c', alpha=0.05)
    axes[1, 1].set_title('4. Residual Autocorrelation (White Noise Check)', fontweight='bold')
    axes[1, 1].set_xlabel('Lag (Hours)')
    axes[1, 1].set_ylabel('ACF Coefficient')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(f'4-Panel Residual Diagnostics: {model_name}', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_forecast_comparison(
    train_series: pd.Series,
    test_series: pd.Series,
    forecast_series: pd.Series,
    conf_int: pd.DataFrame = None,
    model_name: str = 'SARIMAX',
    figsize=(14, 6)
):
    """
    Plots out-of-sample forecast comparison against actual test data with 95% confidence intervals.
    """
    fig, ax = plt.subplots(figsize=figsize)

    last_train_points = min(168, len(train_series))
    ax.plot(train_series.index[-last_train_points:], train_series.values[-last_train_points:], label='Training History (Last 7 Days)', color='#1f77b4', lw=1.5)
    ax.plot(test_series.index, test_series.values, label='Actual Values (Test Ground Truth)', color='#2ca02c', lw=2.0, marker='o', markersize=3)
    ax.plot(forecast_series.index, forecast_series.values, label=f'Model Forecast ({model_name})', color='#d62728', lw=2.0, linestyle='--')

    if conf_int is not None and not conf_int.empty:
        lower_col = conf_int.columns[0]
        upper_col = conf_int.columns[1]
        ax.fill_between(forecast_series.index, conf_int[lower_col], conf_int[upper_col], color='#d62728', alpha=0.2, label='95% Confidence Interval')

    ax.set_title(f'Out-of-Sample Forecast (72h): {model_name} vs Actual Demand', fontweight='bold', fontsize=12)
    ax.set_ylabel('Electricity Demand [MW]')
    ax.set_xlabel('Timestamp')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    return fig


def plot_multi_models_comparison(
    test_series: pd.Series,
    predictions: dict,
    title: str = 'Multi-Model Forecast Comparison on Test Horizon (72h)',
    figsize=(14, 6)
):
    """
    Plots multi-model overlay against actual ground truth.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(test_series.index, test_series.values, label='Actual Demand (Ground Truth)', color='black', lw=2.5, marker='.', markersize=4)

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    styles = ['--', '-.', ':', '-', '--']

    for i, (name, pred) in enumerate(predictions.items()):
        c = colors[i % len(colors)]
        ls = styles[i % len(styles)]
        ax.plot(test_series.index, pred.values if hasattr(pred, 'values') else pred, label=name, color=c, lw=1.8, linestyle=ls)

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_ylabel('Electricity Demand [MW]')
    ax.set_xlabel('Timestamp')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    return fig

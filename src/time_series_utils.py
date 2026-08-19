"""
Time series utility module for statistical testing, residual diagnostics, metric calculations, and feature engineering.
"""

from typing import Dict, Any, List, Optional
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


def check_stationarity(series: pd.Series, significance: float = 0.05) -> Dict[str, Any]:
    """
    Performs stationarity tests: ADF (Augmented Dickey-Fuller) and KPSS.

    Args:
        series (pd.Series): Univariate time series.
        significance (float): Statistical significance level (default: 0.05).

    Returns:
        Dict[str, Any]: Test statistics, p-values, critical values, and analytical conclusion.
    """
    clean_series = series.dropna()

    # 1. ADF Test (H0: non-stationary / unit root present)
    adf_result = adfuller(clean_series, autolag="AIC")
    adf_stat, adf_pvalue, adf_lags, adf_nobs, adf_crit, _ = adf_result
    adf_stationary = bool(adf_pvalue < significance)

    # 2. KPSS Test (H0: stationary around a constant)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_result = kpss(clean_series, regression="c", nlags="auto")

    kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = kpss_result
    kpss_stationary = bool(kpss_pvalue >= significance)

    # Combined conclusion
    if adf_stationary and kpss_stationary:
        conclusion = "Series is strictly stationary (ADF and KPSS agree)."
    elif not adf_stationary and not kpss_stationary:
        conclusion = "Series is non-stationary (differencing required)."
    elif adf_stationary and not kpss_stationary:
        conclusion = "Series is trend-stationary (consider detrending or differencing)."
    else:
        conclusion = "Series is difference-stationary (differencing d=1 recommended)."

    return {
        "adf": {
            "statistic": float(adf_stat),
            "p_value": float(adf_pvalue),
            "used_lag": int(adf_lags),
            "critical_values": {k: float(v) for k, v in adf_crit.items()},
            "is_stationary": adf_stationary,
        },
        "kpss": {
            "statistic": float(kpss_stat),
            "p_value": float(kpss_pvalue),
            "used_lag": int(kpss_lags),
            "critical_values": {k: float(v) for k, v in kpss_crit.items()},
            "is_stationary": kpss_stationary,
        },
        "conclusion": conclusion,
    }


def run_residual_diagnostics(residuals: pd.Series, lags: int = 24, significance: float = 0.05) -> Dict[str, Any]:
    """
    Performs comprehensive statistical diagnostics on model residuals:
    - Ljung-Box test (white noise / no autocorrelation)
    - Jarque-Bera test (normality of distribution)
    - Shapiro-Wilk test (normality on sample)
    - Engle ARCH test (homoscedasticity / constant variance)

    Args:
        residuals (pd.Series): Model residuals (y_true - y_pred).
        lags (int): Number of lags for Ljung-Box test.
        significance (float): Significance level.

    Returns:
        Dict[str, Any]: Statistical test results and interpretations.
    """
    res = residuals.dropna().to_numpy()

    # 1. Ljung-Box test
    lb_df = acorr_ljungbox(res, lags=[lags], return_df=True)
    lb_stat = float(lb_df["lb_stat"].iloc[0])
    lb_pvalue = float(lb_df["lb_pvalue"].iloc[0])
    no_autocorr = bool(lb_pvalue > significance)

    # 2. Jarque-Bera test, skewness, and kurtosis
    jb_stat, jb_pvalue = stats.jarque_bera(res)
    skewness = float(stats.skew(res))
    kurt = float(stats.kurtosis(res))
    is_normal_jb = bool(jb_pvalue > significance)

    # 3. Shapiro-Wilk test
    sample_res = res if len(res) <= 5000 else np.random.choice(res, 5000, replace=False)
    shapiro_stat, shapiro_pvalue = stats.shapiro(sample_res)
    is_normal_shapiro = bool(shapiro_pvalue > significance)

    # 4. Engle ARCH test (Heteroskedasticity)
    arch_stat, arch_pvalue, _, _ = het_arch(res)
    homoscedastic = bool(arch_pvalue > significance)

    return {
        "mean_residual": float(np.mean(res)),
        "std_residual": float(np.std(res)),
        "skewness": skewness,
        "kurtosis": kurt,
        "ljung_box": {
            "lags": lags,
            "stat": lb_stat,
            "p_value": lb_pvalue,
            "white_noise": no_autocorr,
            "interpretation": "No significant autocorrelation (residuals are white noise)." if no_autocorr else "Autocorrelation detected in residuals (model missed temporal dependencies)."
        },
        "jarque_bera": {
            "stat": float(jb_stat),
            "p_value": float(jb_pvalue),
            "is_normal": is_normal_jb,
            "interpretation": "Residual distribution consistent with normal distribution." if is_normal_jb else "Residuals deviate from normal distribution (heavy tails or skewness)."
        },
        "shapiro_wilk": {
            "stat": float(shapiro_stat),
            "p_value": float(shapiro_pvalue),
            "is_normal": is_normal_shapiro
        },
        "engle_arch": {
            "stat": float(arch_stat),
            "p_value": float(arch_pvalue),
            "homoscedastic": homoscedastic,
            "interpretation": "No ARCH effect (residual variance is constant over time)." if homoscedastic else "ARCH effect detected (heteroskedasticity / varying residual variance)."
        }
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard evaluation metrics for forecast accuracy.

    Args:
        y_true: Ground truth actual values.
        y_pred: Forecasted values.

    Returns:
        Dict[str, float]: Dictionary with RMSE, MAE, MAPE, WAPE, R2, and MDA.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    # Mask NaNs
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t, y_p = y_t[mask], y_p[mask]

    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    mape = float(mean_absolute_percentage_error(y_t, y_p) * 100)
    wape = float((np.sum(np.abs(y_t - y_p)) / np.sum(np.abs(y_t))) * 100)
    r2 = float(r2_score(y_t, y_p))

    # Mean Directional Accuracy (MDA)
    if len(y_t) > 1:
        direction_true = np.sign(np.diff(y_t))
        direction_pred = np.sign(np.diff(y_p))
        mda = float(np.mean(direction_true == direction_pred) * 100)
    else:
        mda = 0.0

    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE [%]": round(mape, 2),
        "WAPE [%]": round(wape, 2),
        "R2": round(r2, 4),
        "MDA [%]": round(mda, 2)
    }


def create_time_features(
    df: pd.DataFrame,
    target_col: str = "demand",
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Generates a rich feature matrix for machine learning time series forecasting:
    - Calendar features (hour, day of week, day of month, month, weekend indicator)
    - Cyclical trigonometric transformations (sin/cos for hour and day of week)
    - Lagged target values
    - Rolling window statistics (mean, std, min, max) with shift(1) to avoid look-ahead bias

    Args:
        df (pd.DataFrame): DataFrame with a DateTimeIndex.
        target_col (str): Name of the target variable to forecast.
        lags (List[int]): List of lag steps.
        rolling_windows (List[int]): List of rolling window sizes.

    Returns:
        pd.DataFrame: Feature matrix.
    """
    if lags is None:
        lags = [1, 2, 3, 24, 48, 168]
    if rolling_windows is None:
        rolling_windows = [6, 24, 168]

    df_feat = df.copy()

    # Calendar features
    df_feat["hour"] = df_feat.index.hour
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["day"] = df_feat.index.day
    df_feat["month"] = df_feat.index.month
    df_feat["is_weekend"] = df_feat["dayofweek"].isin([5, 6]).astype(int)

    # Cyclical trigonometric encoding
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)
    df_feat["dow_sin"] = np.sin(2 * np.pi * df_feat["dayofweek"] / 7)
    df_feat["dow_cos"] = np.cos(2 * np.pi * df_feat["dayofweek"] / 7)

    # Lag features for target
    for lag in lags:
        df_feat[f"{target_col}_lag_{lag}"] = df_feat[target_col].shift(lag)

    # Rolling window statistics (shifted by 1 to prevent data leakage)
    for window in rolling_windows:
        lag1_series = df_feat[target_col].shift(1)
        df_feat[f"{target_col}_roll_mean_{window}"] = lag1_series.rolling(window=window).mean()
        df_feat[f"{target_col}_roll_std_{window}"] = lag1_series.rolling(window=window).std()
        df_feat[f"{target_col}_roll_min_{window}"] = lag1_series.rolling(window=window).min()
        df_feat[f"{target_col}_roll_max_{window}"] = lag1_series.rolling(window=window).max()

    # Exogenous RES features (if present)
    if "pv" in df_feat.columns:
        df_feat["pv_lag_24"] = df_feat["pv"].shift(24)
    if "wi" in df_feat.columns:
        df_feat["wi_lag_1"] = df_feat["wi"].shift(1)
        df_feat["wi_lag_24"] = df_feat["wi"].shift(24)

    return df_feat

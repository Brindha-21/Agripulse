"""
AgriPulse - Outlier Detection and Preprocessing Module
Utilizes Scikit-learn (IsolationForest, LocalOutlierFactor, RobustScaler, StandardScaler)
and statistical IQR/Z-Score methods to identify, visualize, and remediate agricultural anomalies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.05,
    scaler_type: str = 'robust',
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detects multivariate anomalies using Scikit-Learn IsolationForest.
    
    Parameters:
    - df: Input agricultural dataframe
    - features: Numerical feature columns to include in the anomaly isolation
    - contamination: Expected proportion of outliers in the data (0.01 to 0.20)
    - scaler_type: 'robust' (RobustScaler, resistant to outliers) or 'standard' (StandardScaler)
    - random_state: Reproducibility seed
    """
    valid_features = [f for f in features if f in df.columns and np.issubdtype(df[f].dtype, np.number)]
    if len(valid_features) == 0:
        raise ValueError("At least one valid numerical feature is required for outlier detection.")
        
    df_result = df.copy()
    X = df_result[valid_features].copy()
    
    # Handle any residual missing values for sklearn
    X_imputed = X.fillna(X.median())
    
    # Preprocessing scaling
    if scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Fit Isolation Forest
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=150,
        n_jobs=-1
    )
    preds = iso.fit_predict(X_scaled)
    # Decision function: lower values mean more anomalous
    scores = iso.decision_function(X_scaled)
    
    # Map: 1 = normal, -1 = outlier
    df_result['is_outlier'] = (preds == -1)
    df_result['anomaly_score'] = np.round(scores, 4)
    df_result['detection_method'] = 'IsolationForest'
    
    outlier_count = int(df_result['is_outlier'].sum())
    total_count = len(df_result)
    
    summary = {
        'method': 'Isolation Forest (Scikit-Learn)',
        'features_used': valid_features,
        'scaler': scaler_type,
        'contamination_set': contamination,
        'outliers_found': outlier_count,
        'outlier_pct': round((outlier_count / total_count) * 100, 2),
        'normal_count': total_count - outlier_count
    }
    
    return df_result, summary


def detect_outliers_lof(
    df: pd.DataFrame,
    features: List[str],
    contamination: float = 0.05,
    n_neighbors: int = 20,
    scaler_type: str = 'robust'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Detects local density-based anomalies using Scikit-Learn LocalOutlierFactor (LOF).
    """
    valid_features = [f for f in features if f in df.columns and np.issubdtype(df[f].dtype, np.number)]
    if len(valid_features) == 0:
        raise ValueError("At least one valid numerical feature is required for LOF outlier detection.")
        
    df_result = df.copy()
    X = df_result[valid_features].fillna(df_result[valid_features].median())
    
    scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False
    )
    preds = lof.fit_predict(X_scaled)
    negative_factor = lof.negative_outlier_factor_
    
    df_result['is_outlier'] = (preds == -1)
    df_result['anomaly_score'] = np.round(negative_factor, 4)
    df_result['detection_method'] = 'LocalOutlierFactor'
    
    outlier_count = int(df_result['is_outlier'].sum())
    total_count = len(df_result)
    
    summary = {
        'method': 'Local Outlier Factor (Scikit-Learn)',
        'features_used': valid_features,
        'scaler': scaler_type,
        'contamination_set': contamination,
        'n_neighbors': n_neighbors,
        'outliers_found': outlier_count,
        'outlier_pct': round((outlier_count / total_count) * 100, 2),
        'normal_count': total_count - outlier_count
    }
    
    return df_result, summary


def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Classical Univariate Interquartile Range (IQR) outlier detection.
    Useful for transparent per-feature bounding.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found.")
        
    df_result = df.copy()
    series = df_result[column].dropna()
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    
    lower_bound = q25 - (multiplier * iqr)
    upper_bound = q75 + (multiplier * iqr)
    
    is_outlier = (df_result[column] < lower_bound) | (df_result[column] > upper_bound)
    df_result['is_outlier'] = is_outlier
    df_result['detection_method'] = f'IQR_{multiplier}'
    
    outlier_count = int(is_outlier.sum())
    total_count = len(df_result)
    
    summary = {
        'method': f'IQR Rule ({multiplier}x IQR)',
        'column': column,
        'q25': round(float(q25), 3),
        'q75': round(float(q75), 3),
        'iqr': round(float(iqr), 3),
        'lower_bound': round(float(lower_bound), 3),
        'upper_bound': round(float(upper_bound), 3),
        'outliers_found': outlier_count,
        'outlier_pct': round((outlier_count / total_count) * 100, 2)
    }
    
    return df_result, summary


def winsorize_dataframe(
    df: pd.DataFrame,
    columns: List[str],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Winsorizes (caps and floors) extreme outliers at specified quantiles
    to protect downstream analytics without discarding data points.
    """
    df_winsor = df.copy()
    bounds_used = {}
    
    for col in columns:
        if col in df_winsor.columns and np.issubdtype(df_winsor[col].dtype, np.number):
            q_low = float(df_winsor[col].quantile(lower_quantile))
            q_high = float(df_winsor[col].quantile(upper_quantile))
            df_winsor[col] = df_winsor[col].clip(lower=q_low, upper=q_high)
            bounds_used[col] = {'capped_lower_at': round(q_low, 3), 'capped_upper_at': round(q_high, 3)}
            
    return df_winsor, bounds_used


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out rows flagged as outliers (`is_outlier == True`).
    """
    if 'is_outlier' in df.columns:
        return df[~df['is_outlier']].copy().reset_index(drop=True)
    return df.copy()

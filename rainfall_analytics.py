"""
AgriPulse - Rainfall Pattern and Yield Correlation Analytics Module
Analyzes precipitation trends, monsoon distribution, rainfall anomalies (drought/flood risks),
and calculates crop-specific rainfall sensitivity and water productivity indices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


def compute_rainfall_climatology(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes baseline climatological statistics for rainfall.
    """
    if 'Annual_Rainfall_mm' not in df.columns:
        return {}
        
    rf = df['Annual_Rainfall_mm'].dropna()
    mean_rf = float(rf.mean())
    std_rf = float(rf.std())
    median_rf = float(rf.median())
    min_rf = float(rf.min())
    max_rf = float(rf.max())
    
    monsoon_mean = float(df['Monsoon_Rainfall_mm'].mean()) if 'Monsoon_Rainfall_mm' in df.columns else (mean_rf * 0.72)
    monsoon_pct = round((monsoon_mean / mean_rf * 100), 1) if mean_rf > 0 else 0.0
    
    return {
        'mean_annual_mm': round(mean_rf, 1),
        'std_annual_mm': round(std_rf, 1),
        'median_annual_mm': round(median_rf, 1),
        'min_annual_mm': round(min_rf, 1),
        'max_annual_mm': round(max_rf, 1),
        'monsoon_mean_mm': round(monsoon_mean, 1),
        'monsoon_share_pct': monsoon_pct
    }


def compute_annual_rainfall_anomalies(df: pd.DataFrame, state: Optional[str] = None) -> pd.DataFrame:
    """
    Calculates year-by-year rainfall departures from the multi-year baseline.
    Classifies each year into:
    - Severe Deficit / Drought (Departure < -20%)
    - Mild Deficit (-20% to -10%)
    - Normal (-10% to +10%)
    - Mild Excess (+10% to +20%)
    - Severe Excess / Flood Risk (> +20%)
    """
    sub_df = df if (state is None or state == 'All States') else df[df['State'] == state]
    if 'Year' not in sub_df.columns or 'Annual_Rainfall_mm' not in sub_df.columns:
        return pd.DataFrame()
        
    annual = sub_df.groupby('Year').agg({
        'Annual_Rainfall_mm': 'mean',
        'Monsoon_Rainfall_mm': 'mean' if 'Monsoon_Rainfall_mm' in sub_df.columns else lambda x: np.nan,
        'Yield_Tonnes_Per_Ha': 'mean' if 'Yield_Tonnes_Per_Ha' in sub_df.columns else lambda x: np.nan
    }).reset_index()
    
    long_term_mean = annual['Annual_Rainfall_mm'].mean()
    annual['Baseline_Mean_mm'] = round(long_term_mean, 1)
    
    # Departure % = (Annual - Mean) / Mean * 100
    annual['Departure_Pct'] = (
        ((annual['Annual_Rainfall_mm'] - long_term_mean) / long_term_mean) * 100
    ).round(1)
    
    # Classification
    def classify_anomaly(dep):
        if dep <= -20.0:
            return 'Drought / Severe Deficit'
        elif -20.0 < dep <= -10.0:
            return 'Mild Deficit'
        elif -10.0 < dep < 10.0:
            return 'Normal Precipitation'
        elif 10.0 <= dep < 20.0:
            return 'Mild Excess'
        else:
            return 'Excess / Flood Risk'
            
    annual['Category'] = annual['Departure_Pct'].apply(classify_anomaly)
    return annual.round(2)


def compute_crop_rainfall_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson and Spearman correlation coefficients between Annual Rainfall
    and Crop Yield for each individual crop to gauge rainfall sensitivity.
    """
    required_cols = ['Crop', 'Annual_Rainfall_mm', 'Yield_Tonnes_Per_Ha']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
        
    results = []
    for crop_name, group in df.groupby('Crop'):
        if len(group) < 5:
            continue
            
        clean_g = group[['Annual_Rainfall_mm', 'Yield_Tonnes_Per_Ha']].dropna()
        if len(clean_g) < 5:
            continue
            
        pearson_corr = clean_g['Annual_Rainfall_mm'].corr(clean_g['Yield_Tonnes_Per_Ha'], method='pearson')
        spearman_corr = clean_g['Annual_Rainfall_mm'].corr(clean_g['Yield_Tonnes_Per_Ha'], method='spearman')
        
        # Water Productivity Index (kg of yield per cubic meter / mm of rain)
        avg_yield = clean_g['Yield_Tonnes_Per_Ha'].mean()
        avg_rf = clean_g['Annual_Rainfall_mm'].mean()
        water_productivity = (avg_yield * 1000.0 / avg_rf) if avg_rf > 0 else 0.0
        
        # Sensitivity interpretation
        if abs(pearson_corr) < 0.15:
            sensitivity = 'Drought-Tolerant / Irrigation-Driven'
        elif pearson_corr > 0.4:
            sensitivity = 'High Rainfall Dependent'
        elif pearson_corr > 0.15:
            sensitivity = 'Moderately Rainfall Sensitive'
        else:
            sensitivity = 'Excess-Water Sensitive'
            
        results.append({
            'Crop': crop_name,
            'Pearson_Correlation': round(float(pearson_corr), 3),
            'Spearman_Correlation': round(float(spearman_corr), 3),
            'Water_Productivity_Kg_Per_mm': round(float(water_productivity), 2),
            'Rainfall_Sensitivity': sensitivity,
            'Average_Yield': round(float(avg_yield), 2),
            'Average_Rainfall_mm': round(float(avg_rf), 1)
        })
        
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(by='Pearson_Correlation', ascending=False).reset_index(drop=True)
    return res_df


def compute_state_rainfall_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes rainfall distribution and climate risk per state.
    """
    if 'State' not in df.columns or 'Annual_Rainfall_mm' not in df.columns:
        return pd.DataFrame()
        
    state_rf = df.groupby('State').agg(
        Mean_Annual_Rainfall=('Annual_Rainfall_mm', 'mean'),
        Min_Rainfall=('Annual_Rainfall_mm', 'min'),
        Max_Rainfall=('Annual_Rainfall_mm', 'max'),
        Rainfall_Std=('Annual_Rainfall_mm', 'std'),
        Mean_Monsoon=('Monsoon_Rainfall_mm', 'mean') if 'Monsoon_Rainfall_mm' in df.columns else ('Annual_Rainfall_mm', 'mean')
    ).reset_index()
    
    # Variability index (CV)
    state_rf['Rainfall_CV_Pct'] = (
        (state_rf['Rainfall_Std'] / state_rf['Mean_Annual_Rainfall']) * 100
    ).round(1)
    
    return state_rf.sort_values(by='Mean_Annual_Rainfall', ascending=False).reset_index(drop=True).round(2)

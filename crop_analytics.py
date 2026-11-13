"""
AgriPulse - Crop Production and Productivity Analytics Module
Provides statistical computations for crop production trends, yields,
growth rates (CAGR, YoY), regional efficiency, and resource utilization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


def compute_kpi_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level key performance indicators (KPIs) for the dataset.
    """
    total_production = df['Production_Tonnes'].sum() if 'Production_Tonnes' in df.columns else 0.0
    total_area = df['Area_Hectares'].sum() if 'Area_Hectares' in df.columns else 0.0
    avg_yield = (total_production / total_area) if total_area > 0 else (df['Yield_Tonnes_Per_Ha'].mean() if 'Yield_Tonnes_Per_Ha' in df.columns else 0.0)
    
    mean_rainfall = df['Annual_Rainfall_mm'].mean() if 'Annual_Rainfall_mm' in df.columns else 0.0
    unique_crops = df['Crop'].nunique() if 'Crop' in df.columns else 0
    unique_states = df['State'].nunique() if 'State' in df.columns else 0
    years_span = f"{df['Year'].min()} - {df['Year'].max()}" if 'Year' in df.columns else "N/A"
    
    return {
        'total_production_tonnes': float(total_production),
        'total_area_ha': float(total_area),
        'weighted_avg_yield': float(round(avg_yield, 3)),
        'mean_rainfall_mm': float(round(mean_rainfall, 1)),
        'crops_count': unique_crops,
        'states_count': unique_states,
        'years_span': years_span,
        'total_records': len(df)
    }


def compute_annual_crop_trends(df: pd.DataFrame, crop: Optional[str] = None) -> pd.DataFrame:
    """
    Computes year-over-year production, harvested area, average yield, and YoY growth rates.
    """
    sub_df = df if (crop is None or crop == 'All Crops') else df[df['Crop'] == crop]
    
    if 'Year' not in sub_df.columns:
        return pd.DataFrame()
        
    annual = sub_df.groupby('Year').agg({
        'Production_Tonnes': 'sum',
        'Area_Hectares': 'sum',
        'Yield_Tonnes_Per_Ha': 'mean',
        'Annual_Rainfall_mm': 'mean'
    }).reset_index()
    
    # Calculate weighted yield
    annual['Weighted_Yield_Tonnes_Per_Ha'] = (annual['Production_Tonnes'] / annual['Area_Hectares']).round(3)
    
    # Calculate YoY growth
    annual['Production_YoY_Pct'] = annual['Production_Tonnes'].pct_change() * 100
    annual['Yield_YoY_Pct'] = annual['Weighted_Yield_Tonnes_Per_Ha'].pct_change() * 100
    
    return annual.round(2)


def compute_cagr(start_val: float, end_val: float, num_years: int) -> float:
    """
    Computes Compound Annual Growth Rate (CAGR).
    """
    if start_val <= 0 or end_val <= 0 or num_years <= 0:
        return 0.0
    return float(round(((end_val / start_val) ** (1.0 / num_years) - 1.0) * 100, 2))


def compute_crop_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks all crops by total production, cultivated area, average yield,
    and yield stability (Coefficient of Variation - lower means more stable).
    """
    if 'Crop' not in df.columns:
        return pd.DataFrame()
        
    grouped = df.groupby('Crop').agg(
        Total_Production_Tonnes=('Production_Tonnes', 'sum'),
        Total_Area_Hectares=('Area_Hectares', 'sum'),
        Avg_Yield_Tonnes_Per_Ha=('Yield_Tonnes_Per_Ha', 'mean'),
        Yield_Std=('Yield_Tonnes_Per_Ha', 'std'),
        Mean_Rainfall_mm=('Annual_Rainfall_mm', 'mean'),
        Mean_Fertilizer_Kg_Ha=('Fertilizer_Usage_Kg_Per_Ha', 'mean')
    ).reset_index()
    
    # Calculate Coefficient of Variation (CV = std / mean * 100)
    grouped['Yield_CV_Pct'] = ((grouped['Yield_Std'] / grouped['Avg_Yield_Tonnes_Per_Ha']) * 100).round(1)
    
    # National Production & Area shares
    total_prod = grouped['Total_Production_Tonnes'].sum()
    total_area = grouped['Total_Area_Hectares'].sum()
    grouped['Production_Share_Pct'] = ((grouped['Total_Production_Tonnes'] / total_prod) * 100).round(2)
    grouped['Area_Share_Pct'] = ((grouped['Total_Area_Hectares'] / total_area) * 100).round(2)
    
    # Sort by total production
    grouped = grouped.sort_values(by='Total_Production_Tonnes', ascending=False).reset_index(drop=True)
    return grouped.round(2)


def compute_state_performance(df: pd.DataFrame, crop: Optional[str] = None) -> pd.DataFrame:
    """
    Analyzes crop production and yield efficiency across states.
    """
    sub_df = df if (crop is None or crop == 'All Crops') else df[df['Crop'] == crop]
    if 'State' not in sub_df.columns:
        return pd.DataFrame()
        
    state_summary = sub_df.groupby('State').agg(
        Total_Production_Tonnes=('Production_Tonnes', 'sum'),
        Total_Area_Hectares=('Area_Hectares', 'sum'),
        Avg_Yield=('Yield_Tonnes_Per_Ha', 'mean'),
        Mean_Rainfall=('Annual_Rainfall_mm', 'mean'),
        Mean_Irrigation=('Irrigation_Coverage_Pct', 'mean')
    ).reset_index()
    
    state_summary['Weighted_Yield'] = (
        state_summary['Total_Production_Tonnes'] / state_summary['Total_Area_Hectares']
    ).round(3)
    
    state_summary = state_summary.sort_values(by='Total_Production_Tonnes', ascending=False).reset_index(drop=True)
    return state_summary.round(2)


def compute_seasonal_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares agricultural productivity across seasons (Kharif, Rabi, Zaid, Whole Year).
    """
    if 'Season' not in df.columns:
        return pd.DataFrame()
        
    seasonal = df.groupby('Season').agg(
        Record_Count=('Crop', 'count'),
        Total_Production=('Production_Tonnes', 'sum'),
        Total_Area=('Area_Hectares', 'sum'),
        Mean_Yield=('Yield_Tonnes_Per_Ha', 'mean'),
        Mean_Rainfall=('Annual_Rainfall_mm', 'mean')
    ).reset_index()
    
    total_prod = seasonal['Total_Production'].sum()
    seasonal['Share_Of_Total_Production_Pct'] = ((seasonal['Total_Production'] / total_prod) * 100).round(1)
    return seasonal.round(2)

"""
AgriPulse - Crop-wise Comparative Analytics Module
Provides multi-crop benchmarking, side-by-side metric comparison,
stability analysis, and normalized relative performance indexing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


def compare_selected_crops(df: pd.DataFrame, crops_list: List[str]) -> pd.DataFrame:
    """
    Computes side-by-side comparative metrics for a selected subset of crops.
    """
    if not crops_list or 'Crop' not in df.columns:
        return pd.DataFrame()
        
    sub_df = df[df['Crop'].isin(crops_list)]
    if sub_df.empty:
        return pd.DataFrame()
        
    summary_rows = []
    for crop in crops_list:
        c_df = sub_df[sub_df['Crop'] == crop]
        if c_df.empty:
            continue
            
        prod = float(c_df['Production_Tonnes'].sum()) if 'Production_Tonnes' in c_df.columns else 0.0
        area = float(c_df['Area_Hectares'].sum()) if 'Area_Hectares' in c_df.columns else 0.0
        yield_mean = float(c_df['Yield_Tonnes_Per_Ha'].mean()) if 'Yield_Tonnes_Per_Ha' in c_df.columns else 0.0
        yield_std = float(c_df['Yield_Tonnes_Per_Ha'].std()) if 'Yield_Tonnes_Per_Ha' in c_df.columns else 0.0
        yield_cv = round((yield_std / yield_mean * 100), 1) if yield_mean > 0 else 0.0
        
        rf_mean = float(c_df['Annual_Rainfall_mm'].mean()) if 'Annual_Rainfall_mm' in c_df.columns else 0.0
        fert_mean = float(c_df['Fertilizer_Usage_Kg_Per_Ha'].mean()) if 'Fertilizer_Usage_Kg_Per_Ha' in c_df.columns else 0.0
        
        # Resource efficiencies
        fert_efficiency = round((yield_mean * 1000.0 / fert_mean), 2) if fert_mean > 0 else 0.0
        water_efficiency = round((yield_mean * 1000.0 / rf_mean), 2) if rf_mean > 0 else 0.0
        
        summary_rows.append({
            'Crop': crop,
            'Total_Production_Tonnes': round(prod, 1),
            'Harvested_Area_Ha': round(area, 1),
            'Mean_Yield_Tonnes_Per_Ha': round(yield_mean, 3),
            'Yield_Volatility_CV_Pct': yield_cv,
            'Mean_Rainfall_mm': round(rf_mean, 1),
            'Mean_Fertilizer_Kg_Ha': round(fert_mean, 1),
            'Fertilizer_Efficiency_KgYield_Per_KgFert': fert_efficiency,
            'Water_Efficiency_KgYield_Per_mmRain': water_efficiency
        })
        
    return pd.DataFrame(summary_rows)


def compute_normalized_comparison_matrix(df: pd.DataFrame, crops_list: List[str]) -> pd.DataFrame:
    """
    Normalizes comparative dimensions into a 0 - 100 score for radar / profile chart.
    Dimensions:
    - Yield Level (High = 100)
    - Yield Stability (Lower CV = Higher Score 100)
    - Water Efficiency (High = 100)
    - Fertilizer Efficiency (High = 100)
    - Area Scale (High = 100)
    """
    comp_df = compare_selected_crops(df, crops_list)
    if comp_df.empty or len(comp_df) < 1:
        return pd.DataFrame()
        
    norm_df = pd.DataFrame()
    norm_df['Crop'] = comp_df['Crop']
    
    # Helper min-max scaler
    def scale_col(series, invert=False):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series(80.0, index=series.index)
        scaled = ((series - s_min) / (s_max - s_min)) * 80.0 + 20.0
        return (100.0 - scaled + 20.0) if invert else scaled
        
    norm_df['Yield Level'] = scale_col(comp_df['Mean_Yield_Tonnes_Per_Ha']).round(1)
    norm_df['Yield Stability'] = scale_col(comp_df['Yield_Volatility_CV_Pct'], invert=True).round(1)
    norm_df['Water Efficiency'] = scale_col(comp_df['Water_Efficiency_KgYield_Per_mmRain']).round(1)
    norm_df['Fertilizer Efficiency'] = scale_col(comp_df['Fertilizer_Efficiency_KgYield_Per_KgFert']).round(1)
    norm_df['Acreage Scale'] = scale_col(comp_df['Harvested_Area_Ha']).round(1)
    
    return norm_df

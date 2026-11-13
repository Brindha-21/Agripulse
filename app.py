"""
AgriPulse: Agricultural Data Quality and Crop Productivity Analytics System
An interactive dashboard powered by Streamlit, Pandas, NumPy, Matplotlib, and Scikit-Learn.
"""

import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

# Import internal modules
from data_cleaning import audit_data_quality, clean_pipeline
from outlier_detection import (
    detect_outliers_isolation_forest,
    detect_outliers_lof,
    detect_outliers_iqr,
    filter_outliers,
    winsorize_dataframe
)
from crop_analytics import (
    compute_kpi_summary,
    compute_annual_crop_trends,
    compute_crop_leaderboard,
    compute_state_performance,
    compute_seasonal_breakdown
)
from rainfall_analytics import (
    compute_rainfall_climatology,
    compute_annual_rainfall_anomalies,
    compute_crop_rainfall_correlations,
    compute_state_rainfall_profiles
)
from crop_comparison import (
    compare_selected_crops,
    compute_normalized_comparison_matrix
)
from visualization import (
    plot_quality_scorecard,
    plot_outlier_scatter,
    plot_production_yield_trends,
    plot_rainfall_anomalies,
    plot_rainfall_vs_yield_regression,
    plot_crop_leaderboard_horizontal,
    plot_crop_radar_comparison,
    plot_correlation_matrix
)
from utils import format_large_number, convert_df_to_csv, load_dataset

# Streamlit Page Setup
st.set_page_config(
    page_title="AgriPulse - Agricultural Analytics System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1B5E20;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #455A64;
        margin-top: 2px;
        margin-bottom: 18px;
    }
    .kpi-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #2E7D32;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #555555;
        text-transform: uppercase;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1B5E20;
        margin-top: 4px;
    }
    .badge-clean {
        background-color: #E8F5E9;

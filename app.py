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
from src.rainfall_analytics import (
    compute_rainfall_climatology,
    compute_annual_rainfall_anomalies,
    compute_crop_rainfall_correlations,
    compute_state_rainfall_profiles
)
from src.crop_comparison import (
    compare_selected_crops,
    compute_normalized_comparison_matrix
)
from src.visualization import (
    plot_quality_scorecard,
    plot_outlier_scatter,
    plot_production_yield_trends,
    plot_rainfall_anomalies,
    plot_rainfall_vs_yield_regression,
    plot_crop_leaderboard_horizontal,
    plot_crop_radar_comparison,
    plot_correlation_matrix
)
from src.utils import format_large_number, convert_df_to_csv, load_dataset

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
        color: #2E7D32;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-raw {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'crop_production_rainfall_raw.csv')
CLEAN_DATA_PATH = os.path.join(DATA_DIR, 'crop_production_rainfall_clean.csv')


# Data Loader with Caching
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    num_cols = ['Year', 'Area_Hectares', 'Production_Tonnes', 'Yield_Tonnes_Per_Ha',
                'Annual_Rainfall_mm', 'Monsoon_Rainfall_mm', 'Temperature_Celsius',
                'Fertilizer_Usage_Kg_Per_Ha', 'Irrigation_Coverage_Pct']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# Session State Initialization
if 'active_df' not in st.session_state:
    if os.path.exists(RAW_DATA_PATH):
        st.session_state.active_df = load_csv(RAW_DATA_PATH)
        st.session_state.data_origin = 'Raw Benchmark (With Anomalies)'
    elif os.path.exists(CLEAN_DATA_PATH):
        st.session_state.active_df = load_csv(CLEAN_DATA_PATH)
        st.session_state.data_origin = 'Clean Benchmark'
    else:
        st.session_state.active_df = pd.DataFrame()
        st.session_state.data_origin = 'None'


# --- SIDEBAR: DATASET SELECTION & GLOBAL FILTERS ---
with st.sidebar:
    st.markdown("### 🌾 AgriPulse Control Center")
    st.markdown("---")
    
    # Dataset Selector
    dataset_option = st.selectbox(
        "Select Dataset Source:",
        [
            "Raw Benchmark (Contains Noisy Records, Outliers, Missing Values)",
            "Clean Benchmark (Pre-sanitized Agricultural Data)",
            "Upload Custom File (CSV or Excel)"
        ],
        index=0
    )
    
    if dataset_option == "Upload Custom File (CSV or Excel)":
        uploaded_file = st.file_uploader("Upload Agricultural CSV / Excel", type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            try:
                st.session_state.active_df = load_dataset(uploaded_file, is_uploaded=True)
                st.session_state.data_origin = f"Custom Upload ({uploaded_file.name})"
                st.success(f"Loaded {len(st.session_state.active_df)} records.")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    elif "Raw Benchmark" in dataset_option:
        if st.session_state.data_origin != 'Raw Benchmark (With Anomalies)' and os.path.exists(RAW_DATA_PATH):
            st.session_state.active_df = load_csv(RAW_DATA_PATH)
            st.session_state.data_origin = 'Raw Benchmark (With Anomalies)'
    else:
        if st.session_state.data_origin != 'Clean Benchmark' and os.path.exists(CLEAN_DATA_PATH):
            st.session_state.active_df = load_csv(CLEAN_DATA_PATH)
            st.session_state.data_origin = 'Clean Benchmark'
            
    df_current = st.session_state.active_df
    
    st.markdown("---")
    st.markdown("#### 🔍 Global Data Filters")
    
    # Dynamic Filters
    if not df_current.empty and 'Year' in df_current.columns:
        valid_years = sorted(df_current['Year'].dropna().unique().astype(int))
        if len(valid_years) > 1:
            selected_years = st.slider(
                "Year Range:",
                min_value=int(min(valid_years)),
                max_value=int(max(valid_years)),
                value=(int(min(valid_years)), int(max(valid_years)))
            )
        else:
            selected_years = (int(valid_years[0]), int(valid_years[0]))
    else:
        selected_years = (2013, 2024)
        
    # State Filter
    if not df_current.empty and 'State' in df_current.columns:
        all_states = sorted([str(s) for s in df_current['State'].dropna().unique()])
        selected_states = st.multiselect("Filter by State:", options=all_states, default=all_states)
    else:
        selected_states = []
        
    # Crop Filter
    if not df_current.empty and 'Crop' in df_current.columns:
        all_crops = sorted([str(c) for c in df_current['Crop'].dropna().unique()])
        selected_crops = st.multiselect("Filter by Crop:", options=all_crops, default=all_crops)
    else:
        selected_crops = []
        
    # Season Filter
    if not df_current.empty and 'Season' in df_current.columns:
        all_seasons = sorted([str(s) for s in df_current['Season'].dropna().unique()])
        selected_seasons = st.multiselect("Filter by Season:", options=all_seasons, default=all_seasons)
    else:
        selected_seasons = []
        
    if st.button("Reset All Filters", use_container_width=True):
        st.rerun()


# Filter the dataframe for dashboard views
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    if 'Year' in filtered.columns and selected_years:
        filtered = filtered[(filtered['Year'] >= selected_years[0]) & (filtered['Year'] <= selected_years[1])]
    if 'State' in filtered.columns and selected_states:
        filtered = filtered[filtered['State'].astype(str).isin(selected_states)]
    if 'Crop' in filtered.columns and selected_crops:
        filtered = filtered[filtered['Crop'].astype(str).isin(selected_crops)]
    if 'Season' in filtered.columns and selected_seasons:
        filtered = filtered[filtered['Season'].astype(str).isin(selected_seasons)]
    return filtered

df_view = apply_filters(df_current)


# --- MAIN HEADER ---
st.markdown('<div class="main-title">🌱 AgriPulse Analytics System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Agricultural Data Quality, Outlier Isolation & Crop Productivity Analytics Dashboard</div>', unsafe_allow_html=True)

# Data Profile Banner
col_b1, col_b2, col_b3, col_b4 = st.columns([2.5, 1.2, 1.2, 1.2])
with col_b1:
    origin_badge = "badge-raw" if "Raw" in st.session_state.data_origin else "badge-clean"
    st.markdown(f"**Active Source:** <span class='{origin_badge}'>{st.session_state.data_origin}</span>", unsafe_allow_html=True)
with col_b2:
    st.markdown(f"**Total Records:** `{len(df_current):,}`")
with col_b3:
    st.markdown(f"**Filtered View:** `{len(df_view):,}`")
with col_b4:
    audit_quick = audit_data_quality(df_current)
    st.markdown(f"**Quality Score:** `{audit_quick['overall_score']:.1f}%`")

st.markdown("---")


# --- TABS NAVIGATION ---
tab_overview, tab_cleaning, tab_outliers, tab_rainfall, tab_comparison, tab_export = st.tabs([
    "📊 Executive Overview",
    "🧹 Data Quality & Cleaning",
    "🌲 Scikit-Learn Outliers",
    "🌧️ Rainfall & Yield Analysis",
    "🌾 Crop Comparisons",
    "📥 Data Explorer & Export"
])


# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW & KPI DASHBOARD
# ==============================================================================
with tab_overview:
    st.subheader("🌾 National Agricultural Performance & Production KPI Summary")
    
    if df_view.empty:
        st.warning("No data available under the selected filters. Please adjust the sidebar filters.")
    else:
        kpis = compute_kpi_summary(df_view)
        
        # 5 Top-Level KPI Cards
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Production</div>
                <div class="kpi-value">{format_large_number(kpis['total_production_tonnes'])} T</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Harvested Land</div>
                <div class="kpi-value">{format_large_number(kpis['total_area_ha'])} Ha</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Weighted Yield</div>
                <div class="kpi-value">{kpis['weighted_avg_yield']:.2f} T/Ha</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Mean Rainfall</div>
                <div class="kpi-value">{kpis['mean_rainfall_mm']:.1f} mm</div>
            </div>
            """, unsafe_allow_html=True)
        with k5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Data Quality</div>
                <div class="kpi-value">{audit_quick['overall_score']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 2: Production Trend & Top Crops by Volume
        col_t1, col_t2 = st.columns([1.3, 1.0])
        
        with col_t1:
            st.markdown("##### 📈 Annual Harvest Volume vs Yield Efficiency")
            annual_trends = compute_annual_crop_trends(df_view)
            if not annual_trends.empty:
                fig_trend = plot_production_yield_trends(annual_trends, crop_name="Selected Scope")
                st.pyplot(fig_trend, clear_figure=True)
                plt.close(fig_trend)
            else:
                st.info("Insufficient annual data to plot trends.")
                
        with col_t2:
            st.markdown("##### 🏆 Leading Crops by Total Output Volume")
            crop_leaders = compute_crop_leaderboard(df_view)
            if not crop_leaders.empty:
                fig_lead = plot_crop_leaderboard_horizontal(crop_leaders, top_n=7)
                st.pyplot(fig_lead, clear_figure=True)
                plt.close(fig_lead)
            else:
                st.info("Insufficient crop breakdown data.")
                
        st.markdown("---")
        
        # Row 3: Seasonal Performance Breakdown & Regional Summary
        col_s1, col_s2 = st.columns([1.1, 1.2])
        
        with col_s1:
            st.markdown("##### 🍂 Seasonal Harvest Distribution")
            seasonal_df = compute_seasonal_breakdown(df_view)
            if not seasonal_df.empty:
                st.dataframe(
                    seasonal_df[['Season', 'Total_Production', 'Total_Area', 'Mean_Yield', 'Share_Of_Total_Production_Pct']],
                    column_config={
                        'Season': 'Season',
                        'Total_Production': st.column_config.NumberColumn('Production (T)', format="%.0f"),
                        'Total_Area': st.column_config.NumberColumn('Area (Ha)', format="%.0f"),
                        'Mean_Yield': st.column_config.NumberColumn('Avg Yield (T/Ha)', format="%.2f"),
                        'Share_Of_Total_Production_Pct': st.column_config.ProgressColumn('Share (%)', min_value=0, max_value=100, format="%.1f%%")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
        with col_s2:
            st.markdown("##### 🗺️ Top Producing States / Regions")
            state_perf = compute_state_performance(df_view)
            if not state_perf.empty:
                st.dataframe(
                    state_perf.head(8)[['State', 'Total_Production_Tonnes', 'Weighted_Yield', 'Mean_Rainfall', 'Mean_Irrigation']],
                    column_config={
                        'Total_Production_Tonnes': st.column_config.NumberColumn('Production (T)', format="%.0f"),
                        'Weighted_Yield': st.column_config.NumberColumn('Yield (T/Ha)', format="%.2f"),
                        'Mean_Rainfall': st.column_config.NumberColumn('Rainfall (mm)', format="%.0f"),
                        'Mean_Irrigation': st.column_config.NumberColumn('Irrigation (%)', format="%.1f%%")
                    },
                    use_container_width=True,
                    hide_index=True
                )


# ==============================================================================
# TAB 2: AGRICULTURAL DATA QUALITY & CLEANING
# ==============================================================================
with tab_cleaning:
    st.subheader("🧹 Agricultural Data Quality Audit & Automated Cleaning Pipeline")
    st.markdown("""
    Evaluate data cleanliness across four core dimensions: **Completeness**, **Validity**, **Uniqueness**, and **Consistency**.
    Cleanse noisy sensor errors, out-of-range yields, missing records, casing discrepancies, and duplicate entries.
    """)
    
    current_audit = audit_data_quality(df_current)
    
    # Scorecard Chart & Summary Metrics
    col_q1, col_q2 = st.columns([1.2, 1.0])
    with col_q1:
        fig_score = plot_quality_scorecard(current_audit)
        st.pyplot(fig_score, clear_figure=True)
        plt.close(fig_score)
        
    with col_q2:
        st.markdown("##### 🔍 Data Quality Audit Summary")
        st.write(f"• **Overall Health Index:** `{current_audit['overall_score']:.1f}%`")
        st.write(f"• **Completeness:** `{current_audit['completeness_score']:.1f}%` ({current_audit['total_missing_cells']} missing values)")
        st.write(f"• **Validity:** `{current_audit['validity_score']:.1f}%` ({current_audit['invalid_records_count']} physical / range violations)")
        st.write(f"• **Uniqueness:** `{current_audit['uniqueness_score']:.1f}%` ({current_audit['duplicate_rows']} duplicate records)")
        st.write(f"• **Consistency:** `{current_audit['consistency_score']:.1f}%` (standardization check)")
        
        if current_audit['validity_issues']:
            st.warning("**Issues Flagged:**\n- " + "\n- ".join(current_audit['validity_issues'][:5]))
            
    st.markdown("---")
    
    # Cleaning Pipeline Studio
    st.subheader("⚙️ Configure Data Cleaning Pipeline")
    
    with st.expander("🛠️ Interactive Cleaning Pipeline Settings", expanded=True):
        c_pipe1, c_pipe2 = st.columns(2)
        
        with c_pipe1:
            impute_strat = st.selectbox(
                "Missing Value Imputation Strategy:",
                options=[
                    ('group_median', 'Hierarchical Group Median (Crop & State median) [Recommended for Agriculture]'),
                    ('median', 'Global Column Median'),
                    ('mean', 'Global Column Mean'),
                    ('knn', 'Scikit-Learn KNN Imputer (k=5)'),
                    ('drop', 'Drop Records with Missing Values')
                ],
                format_func=lambda x: x[1]
            )[0]
            
            do_recompute_yield = st.checkbox(
                "Recalculate Yield = (Production / Area) after correction",
                value=True,
                help="Ensures mathematical integrity between harvest production and harvested acreage."
            )
            
        with c_pipe2:
            do_dedup = st.checkbox("Deduplicate identical rows", value=True)
            do_standardize = st.checkbox("Standardize text casing & state/crop aliases (e.g. U.P. -> Uttar Pradesh)", value=True)
            do_sanitize = st.checkbox("Sanitize impossible values (Negative rainfall, extreme biological yield spikes)", value=True)
            
        if st.button("🚀 Execute Data Cleaning Pipeline", type="primary", use_container_width=True):
            with st.spinner("Processing agricultural data cleansing..."):
                cleaned_df, report = clean_pipeline(
                    df_current,
                    imputation_strategy=impute_strat,
                    deduplicate=do_dedup,
                    recompute_yield=do_recompute_yield
                )
                
                # Update session state with clean data
                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_report = report
                st.success(f"Pipeline finished! Quality Score improved by **+{report['quality_improvement']:.1f}%** (New Score: **{report['final_audit']['overall_score']:.1f}%**).")
                
    # Cleaning Results & Actions
    if 'cleaning_report' in st.session_state:
        rep = st.session_state.cleaning_report
        c_res1, c_res2 = st.columns([1.1, 1.2])
        
        with c_res1:
            st.markdown("##### 📝 Pipeline Execution Log")
            for action in rep['actions_performed']:
                st.markdown(f"- ✅ {action}")
                
        with c_res2:
            st.markdown("##### 📊 Quality Score Comparison")
            c_before = rep['initial_audit']['overall_score']
            c_after = rep['final_audit']['overall_score']
            st.metric(
                label="Overall Data Quality Score",
                value=f"{c_after:.1f}%",
                delta=f"+{rep['quality_improvement']:.1f}%"
            )
            
            if st.button("Apply Cleaned Data to Dashboard & Analytics", type="secondary", use_container_width=True):
                st.session_state.active_df = st.session_state.cleaned_df
                st.session_state.data_origin = 'Cleaned & Standardized Dataset'
                st.rerun()


# ==============================================================================
# TAB 3: SCIKIT-LEARN OUTLIER DETECTION & PREPROCESSING
# ==============================================================================
with tab_outliers:
    st.subheader("🌲 Multivariate Outlier Detection & Preprocessing (Scikit-Learn)")
    st.markdown("""
    Agricultural sensor errors, extreme drought/flood events, and recording mistakes require robust isolation.
    Use **Scikit-Learn's Isolation Forest**, **Local Outlier Factor (LOF)**, and **Robust Scalers** to isolate multidimensional anomalies.
    """)
    
    col_out_cfg1, col_out_cfg2, col_out_cfg3 = st.columns([1.2, 1.2, 1.0])
    
    with col_out_cfg1:
        algo_choice = st.selectbox(
            "Anomaly Detection Algorithm:",
            options=["Isolation Forest (Scikit-Learn)", "Local Outlier Factor (LOF)", "Interquartile Range (IQR Rule)"]
        )
        scaler_choice = st.selectbox(
            "Feature Preprocessing Scaler:",
            options=["robust", "standard"],
            format_func=lambda x: "RobustScaler (Resistant to outliers)" if x == "robust" else "StandardScaler (Z-Score)"
        )
        
    with col_out_cfg2:
        available_numeric = [
            c for c in ['Yield_Tonnes_Per_Ha', 'Annual_Rainfall_mm', 'Fertilizer_Usage_Kg_Per_Ha', 'Irrigation_Coverage_Pct', 'Temperature_Celsius']
            if c in df_current.columns
        ]
        selected_features = st.multiselect(
            "Select Numerical Features for Isolation:",
            options=available_numeric,
            default=available_numeric[:4] if len(available_numeric) >= 4 else available_numeric
        )
        
    with col_out_cfg3:
        contamination_rate = st.slider(
            "Contamination Rate (% Outliers):",
            min_value=0.01,
            max_value=0.15,
            value=0.05,
            step=0.01,
            format="%.2f"
        )
        run_detection = st.button("🔍 Run Outlier Isolation", type="primary", use_container_width=True)
        
    if run_detection or 'outlier_df' in st.session_state:
        if run_detection:
            if not selected_features:
                st.error("Please select at least one numeric feature for outlier detection.")
            else:
                with st.spinner("Executing Scikit-Learn outlier isolation..."):
                    if "Isolation Forest" in algo_choice:
                        out_df, out_summary = detect_outliers_isolation_forest(
                            df_current,
                            features=selected_features,
                            contamination=contamination_rate,
                            scaler_type=scaler_choice
                        )
                    elif "Local Outlier Factor" in algo_choice:
                        out_df, out_summary = detect_outliers_lof(
                            df_current,
                            features=selected_features,
                            contamination=contamination_rate,
                            scaler_type=scaler_choice
                        )
                    else:
                        out_df, out_summary = detect_outliers_iqr(
                            df_current,
                            column=selected_features[0],
                            multiplier=1.5
                        )
                        
                    st.session_state.outlier_df = out_df
                    st.session_state.outlier_summary = out_summary
                    
        if 'outlier_df' in st.session_state:
            out_df = st.session_state.outlier_df
            out_summary = st.session_state.outlier_summary
            
            # Metrics Row
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Algorithm", out_summary['method'])
            m2.metric("Total Records Evaluated", len(out_df))
            m3.metric("Anomalies Flagged", out_summary['outliers_found'], delta=f"{out_summary['outlier_pct']}% of data", delta_color="inverse")
            m4.metric("Clean Inliers", out_summary['normal_count'])
            
            # Scatter Visualizer
            st.markdown("##### 📍 Interactive Anomaly Scatter Visualizer")
            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                x_axis = st.selectbox("X-Axis Feature:", options=selected_features, index=0)
            with sc_col2:
                y_idx = 1 if len(selected_features) > 1 else 0
                y_axis = st.selectbox("Y-Axis Feature:", options=selected_features, index=y_idx)
                
            fig_out = plot_outlier_scatter(out_df, x_axis, y_axis)
            st.pyplot(fig_out, clear_figure=True)
            plt.close(fig_out)
            
            # Flagged Records Table & Remediation
            st.markdown("##### 📋 Flagged Anomaly Records")
            flagged = out_df[out_df['is_outlier']].copy()
            if not flagged.empty:
                display_cols = [c for c in ['Year', 'State', 'Crop', 'Yield_Tonnes_Per_Ha', 'Annual_Rainfall_mm', 'Fertilizer_Usage_Kg_Per_Ha', 'anomaly_score'] if c in flagged.columns]
                st.dataframe(flagged[display_cols].head(15), use_container_width=True, hide_index=True)
                
                # Remediation Controls
                st.markdown("##### 🛡️ Outlier Remediation Actions")
                rem_c1, rem_c2 = st.columns(2)
                with rem_c1:
                    if st.button("Prune / Drop Flagged Outliers from Active Dataset", use_container_width=True):
                        st.session_state.active_df = filter_outliers(out_df)
                        st.session_state.data_origin = 'Active (Outliers Filtered)'
                        st.success(f"Removed {len(flagged)} anomalous records.")
                        st.rerun()
                with rem_c2:
                    if st.button("Winsorize (Cap/Floor Extreme Quantiles at 1% & 99%)", use_container_width=True):
                        winsor_df, _ = winsorize_dataframe(df_current, columns=selected_features)
                        st.session_state.active_df = winsor_df
                        st.session_state.data_origin = 'Active (Winsorized Outliers)'
                        st.success("Extreme outliers safely clamped to quantile boundaries.")
                        st.rerun()
            else:
                st.success("No anomalies detected under the current contamination threshold.")


# ==============================================================================
# TAB 4: RAINFALL PATTERN & YIELD CORRELATION
# ==============================================================================
with tab_rainfall:
    st.subheader("🌧️ Rainfall Pattern, Monsoon Distribution & Yield Sensitivity")
    st.markdown("""
    Precipitation is a primary driver of agricultural productivity.
    Analyze multi-year rainfall departures (Drought vs Excess), monsoon shares, and evaluate crop-specific rainfall elasticity.
    """)
    
    climatology = compute_rainfall_climatology(df_view)
    
    if climatology:
        rf_c1, rf_c2, rf_c3, rf_c4 = st.columns(4)
        rf_c1.metric("Mean Annual Rainfall", f"{climatology['mean_annual_mm']} mm")
        rf_c2.metric("Monsoon Share", f"{climatology['monsoon_share_pct']}%", f"{climatology['monsoon_mean_mm']} mm")
        rf_c3.metric("Rainfall Spread (Min - Max)", f"{climatology['min_annual_mm']} - {climatology['max_annual_mm']} mm")
        rf_c4.metric("Standard Deviation", f"±{climatology['std_annual_mm']} mm")
        
    st.markdown("---")
    
    # Rainfall Anomaly Waterfall Chart
    col_rf_plot, col_rf_reg = st.columns([1.2, 1.1])
    
    with col_rf_plot:
        st.markdown("##### 📊 Annual Rainfall Anomaly & Meteorological Risk Indicator")
        anomalies_df = compute_annual_rainfall_anomalies(df_view)
        if not anomalies_df.empty:
            fig_anom = plot_rainfall_anomalies(anomalies_df, state_name="All Regions")
            st.pyplot(fig_anom, clear_figure=True)
            plt.close(fig_anom)
        else:
            st.info("Insufficient data to compute rainfall anomalies.")
            
    with col_rf_reg:
        st.markdown("##### 📉 Crop Yield vs Rainfall Sensitivity Regression")
        if 'Crop' in df_view.columns:
            crop_options = sorted([str(c) for c in df_view['Crop'].dropna().unique()])
            if crop_options:
                crop_for_reg = st.selectbox("Select Crop for Regression Analysis:", options=crop_options)
                fig_reg = plot_rainfall_vs_yield_regression(df_view, crop_name=crop_for_reg)
                st.pyplot(fig_reg, clear_figure=True)
                plt.close(fig_reg)
            else:
                st.info("No crops available for regression under current filter selection.")
            
    st.markdown("---")
    
    # Crop-wise Rainfall Sensitivity & Water Productivity
    st.markdown("##### 💧 Crop-wise Rainfall Sensitivity & Water Productivity Index (WPI)")
    st.markdown("Evaluates how responsive each crop is to precipitation variations and calculates kilograms produced per mm of rainfall.")
    
    crop_corr_df = compute_crop_rainfall_correlations(df_view)
    if not crop_corr_df.empty:
        st.dataframe(
            crop_corr_df,
            column_config={
                'Crop': 'Crop Variety',
                'Pearson_Correlation': st.column_config.NumberColumn('Pearson r (Linear)', format="%.3f"),
                'Spearman_Correlation': st.column_config.NumberColumn('Spearman ρ (Rank)', format="%.3f"),
                'Water_Productivity_Kg_Per_mm': st.column_config.NumberColumn('Water Productivity (kg / mm rain)', format="%.2f"),
                'Rainfall_Sensitivity': 'Agronomic Climate Sensitivity',
                'Average_Yield': st.column_config.NumberColumn('Avg Yield (T/Ha)', format="%.2f"),
                'Average_Rainfall_mm': st.column_config.NumberColumn('Avg Rainfall (mm)', format="%.0f")
            },
            use_container_width=True,
            hide_index=True
        )


# ==============================================================================
# TAB 5: CROP-WISE COMPARISONS & BENCHMARKING
# ==============================================================================
with tab_comparison:
    st.subheader("🌾 Multi-Crop Head-to-Head Comparative Analytics")
    st.markdown("""
    Benchmark crops side-by-side across agronomic dimensions: **Yield Level**, **Yield Stability (Risk)**, 
    **Water Efficiency**, **Fertilizer Responsiveness**, and **Harvest Scale**.
    """)
    
    if 'Crop' in df_view.columns:
        all_avail_crops = sorted(df_view['Crop'].dropna().unique().tolist())
        default_selection = all_avail_crops[:4] if len(all_avail_crops) >= 4 else all_avail_crops
        
        selected_compare_crops = st.multiselect(
            "Select 2 to 5 Crops for Head-to-Head Benchmarking:",
            options=all_avail_crops,
            default=default_selection
        )
        
        if len(selected_compare_crops) < 2:
            st.warning("Please select at least 2 crops to generate comparative benchmarking.")
        else:
            col_comp_radar, col_comp_tbl = st.columns([1.1, 1.3])
            
            with col_comp_radar:
                st.markdown("##### 🕸️ Agronomic Profile Radar Matrix")
                norm_matrix = compute_normalized_comparison_matrix(df_view, selected_compare_crops)
                if not norm_matrix.empty:
                    fig_radar = plot_crop_radar_comparison(norm_matrix)
                    st.pyplot(fig_radar, clear_figure=True)
                    plt.close(fig_radar)
                    
            with col_comp_tbl:
                st.markdown("##### 📊 Head-to-Head Metrics Comparison")
                comp_stats = compare_selected_crops(df_view, selected_compare_crops)
                if not comp_stats.empty:
                    st.dataframe(
                        comp_stats[['Crop', 'Mean_Yield_Tonnes_Per_Ha', 'Yield_Volatility_CV_Pct', 'Fertilizer_Efficiency_KgYield_Per_KgFert', 'Water_Efficiency_KgYield_Per_mmRain', 'Harvested_Area_Ha']],
                        column_config={
                            'Crop': 'Crop',
                            'Mean_Yield_Tonnes_Per_Ha': st.column_config.NumberColumn('Mean Yield (T/Ha)', format="%.2f"),
                            'Yield_Volatility_CV_Pct': st.column_config.NumberColumn('Volatility (CV %)', format="%.1f%%"),
                            'Fertilizer_Efficiency_KgYield_Per_KgFert': st.column_config.NumberColumn('Fertilizer Efficiency (kg/kg)', format="%.2f"),
                            'Water_Efficiency_KgYield_Per_mmRain': st.column_config.NumberColumn('Water Efficiency (kg/mm)', format="%.2f"),
                            'Harvested_Area_Ha': st.column_config.NumberColumn('Total Area (Ha)', format="%.0f")
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                    
            if len(comp_stats) >= 1:
                st.markdown("---")
                st.markdown("##### 💡 Agronomic Insights & Trade-Offs")
                top_yield_crop = comp_stats.sort_values(by='Mean_Yield_Tonnes_Per_Ha', ascending=False).iloc[0]['Crop']
                most_stable_crop = comp_stats.sort_values(by='Yield_Volatility_CV_Pct', ascending=True).iloc[0]['Crop']
                best_water_crop = comp_stats.sort_values(by='Water_Efficiency_KgYield_Per_mmRain', ascending=False).iloc[0]['Crop']
                
                ci1, ci2, ci3 = st.columns(3)
                ci1.info(f"🏆 **Highest Yield Potential:** `{top_yield_crop}`")
                ci2.success(f"🛡️ **Most Stable (Lowest Risk CV):** `{most_stable_crop}`")
                ci3.warning(f"💧 **Most Water-Efficient:** `{best_water_crop}`")


# ==============================================================================
# TAB 6: DATA EXPLORER & EXPORT CENTER
# ==============================================================================
with tab_export:
    st.subheader("📥 Agricultural Data Explorer, Correlations & Export Hub")
    st.markdown("Inspect raw tabular data, correlation heatmaps, feature definitions, and download clean datasets.")
    
    col_exp_meta1, col_exp_meta2 = st.columns([1.3, 1.0])
    
    with col_exp_meta1:
        st.markdown("##### 📄 Active Dataset Records")
        st.dataframe(df_view.head(50), use_container_width=True)
        
    with col_exp_meta2:
        st.markdown("##### 🔥 Correlation Heatmap")
        fig_corr = plot_correlation_matrix(df_view)
        st.pyplot(fig_corr, clear_figure=True)
        plt.close(fig_corr)
        
    st.markdown("---")
    
    # Export Section
    st.markdown("##### 💾 Export Datasets & Quality Reports")
    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        csv_data = convert_df_to_csv(df_view)
        st.download_button(
            label="📥 Download Active Cleaned Dataset (CSV)",
            data=csv_data,
            file_name="agripulse_cleaned_agricultural_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with d_col2:
        audit_export_df = pd.DataFrame([audit_quick])
        audit_csv = convert_df_to_csv(audit_export_df)
        st.download_button(
            label="📋 Download Data Quality Audit Report (CSV)",
            data=audit_csv,
            file_name="agripulse_data_quality_report.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    # Schema Documentation
    with st.expander("📖 Agricultural Data Dictionary & Feature Schema"):
        st.markdown("""
        | Feature Name | Data Type | Units / Range | Description |
        |---|---|---|---|
        | `Year` | Integer | 2013 - 2024 | Agricultural harvest calendar year |
        | `State` | Categorical | Names | Indian agricultural state or geographical territory |
        | `Crop` | Categorical | Names | Crop variety (Rice, Wheat, Maize, Cotton, Sugarcane, etc.) |
        | `Season` | Categorical | Kharif, Rabi, Zaid, Whole Year | Cropping season |
        | `Soil_Type` | Categorical | Alluvial, Black, Red, etc. | Soil taxonomy classification |
        | `Area_Hectares` | Numeric (Float) | Hectares (Ha) | Total land area cultivated for the crop |
        | `Production_Tonnes` | Numeric (Float) | Metric Tonnes (T) | Total harvest output produced |
        | `Yield_Tonnes_Per_Ha` | Numeric (Float) | Tonnes / Ha | Crop productivity index = `Production / Area` |
        | `Annual_Rainfall_mm` | Numeric (Float) | Millimeters (mm) | Total cumulative rainfall received in the year |
        | `Monsoon_Rainfall_mm` | Numeric (Float) | Millimeters (mm) | Seasonal monsoon precipitation received |
        | `Temperature_Celsius` | Numeric (Float) | °C | Mean ambient agricultural season temperature |
        | `Fertilizer_Usage_Kg_Per_Ha` | Numeric (Float) | Kg / Ha | Total NPK chemical/organic fertilizer applied per hectare |
        | `Irrigation_Coverage_Pct` | Numeric (Float) | 0% - 100% | Percentage of cultivated area under assured irrigation |
        """)

# Footer
st.markdown("---")
st.markdown(
    "<center><small>AgriPulse © 2026 | Agricultural Data Quality and Crop Productivity Analytics System | Built with Streamlit, Pandas, NumPy, Matplotlib & Scikit-Learn</small></center>",
    unsafe_allow_html=True
)

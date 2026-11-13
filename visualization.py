"""
AgriPulse - Matplotlib Data Visualization Engine
Generates publication-quality, aesthetically styled charts and plots
for agricultural analytics, quality auditing, rainfall patterns, and crop comparisons.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any

# Styling Palette Constants
COLORS = {
    'primary': '#2E7D32',      # Forest Green
    'secondary': '#1565C0',    # Deep Ocean Blue
    'accent': '#E65100',       # Harvest Orange / Amber
    'warning': '#D32F2F',      # Crimson Red
    'neutral_dark': '#212121', # Charcoal
    'neutral_light': '#F5F5F5',# Soft Gray
    'grid': '#E0E0E0',
    'outlier': '#D32F2F',      # Red for anomalies
    'normal': '#2E7D32',       # Green for valid points
    'palette': ['#2E7D32', '#1565C0', '#F57C00', '#6A1B9A', '#00838F', '#C2185B', '#455A64', '#E64A19', '#2E7D32', '#5D4037']
}

def set_plot_style():
    """Applies clean, modern formatting to Matplotlib plots."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#EBEBEB'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'


def plot_quality_scorecard(audit_result: Dict[str, Any]) -> plt.Figure:
    """
    Renders a multi-component quality scorecard showing overall score
    and breakdown across Completeness, Validity, Uniqueness, and Consistency.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=120)
    
    metrics = ['Completeness', 'Validity', 'Uniqueness', 'Consistency', 'OVERALL']
    scores = [
        audit_result.get('completeness_score', 0),
        audit_result.get('validity_score', 0),
        audit_result.get('uniqueness_score', 0),
        audit_result.get('consistency_score', 0),
        audit_result.get('overall_score', 0)
    ]
    
    colors = [
        '#4CAF50' if s >= 85 else ('#FF9800' if s >= 65 else '#F44336')
        for s in scores[:-1]
    ]
    # Highlight overall score
    overall_color = '#1B5E20' if scores[-1] >= 85 else ('#E65100' if scores[-1] >= 65 else '#B71C1C')
    colors.append(overall_color)
    
    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, scores, color=colors, height=0.55, edgecolor='none', zorder=3)
    
    # Value annotations on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1.2, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                va='center', ha='left', fontsize=10, fontweight='bold', color='#333333')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics, fontsize=10, fontweight='bold')
    ax.set_xlim(0, 115)
    ax.set_xlabel('Quality Health Score (%)', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_title(f"Agricultural Data Quality Health Index: {scores[-1]:.1f}/100", 
                 fontsize=12, fontweight='bold', pad=12, color='#1B5E20')
    ax.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
    ax.axvline(85, color='#4CAF50', linestyle=':', alpha=0.7, label='Target Benchmark (85%)')
    ax.legend(loc='lower right', frameon=True, fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_outlier_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> plt.Figure:
    """
    Renders a 2D scatter plot highlighting detected anomalies with distinct halos and labels.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    
    if 'is_outlier' not in df.columns:
        ax.scatter(df[x_col], df[y_col], color=COLORS['primary'], alpha=0.6, s=40)
    else:
        normal = df[~df['is_outlier']]
        outliers = df[df['is_outlier']]
        
        ax.scatter(normal[x_col], normal[y_col], 
                   color='#2E7D32', alpha=0.5, s=35, label=f'Inliers / Normal ({len(normal)})', zorder=2)
        ax.scatter(outliers[x_col], outliers[y_col], 
                   color='#D32F2F', edgecolors='#8B0000', linewidths=1.2,
                   s=75, marker='X', label=f'Outliers Flagged ({len(outliers)})', zorder=3)
        
    ax.set_xlabel(x_col.replace('_', ' '), fontsize=10, fontweight='bold')
    ax.set_ylabel(y_col.replace('_', ' '), fontsize=10, fontweight='bold')
    ax.set_title(f"Outlier Isolation Diagnostics: {x_col.replace('_', ' ')} vs {y_col.replace('_', ' ')}", 
                 fontsize=11, fontweight='bold', pad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_production_yield_trends(annual_df: pd.DataFrame, crop_name: str = 'All Crops') -> plt.Figure:
    """
    Generates a dual-axis trend plot:
    Left Y-axis: Total Production (Tonnes)
    Right Y-axis: Average Yield (Tonnes/Ha)
    """
    set_plot_style()
    fig, ax1 = plt.subplots(figsize=(9, 4.2), dpi=120)
    
    years = annual_df['Year'].astype(int)
    prod = annual_df['Production_Tonnes'] / 1000.0  # in Thousands
    
    # Left Axis: Production
    color_prod = '#1E88E5'
    ax1.set_xlabel('Harvest Year', fontsize=10, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Total Production (Thousand Tonnes)', color=color_prod, fontsize=10, fontweight='bold')
    ax1.bar(years - 0.15, prod, width=0.35, color=color_prod, alpha=0.75, label='Production', zorder=2)
    ax1.tick_params(axis='y', labelcolor=color_prod)
    ax1.set_xticks(years)
    ax1.set_xticklabels(years, rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    # Right Axis: Yield
    ax2 = ax1.twinx()
    color_yield = '#2E7D32'
    yield_val = annual_df['Weighted_Yield_Tonnes_Per_Ha'] if 'Weighted_Yield_Tonnes_Per_Ha' in annual_df.columns else annual_df['Yield_Tonnes_Per_Ha']
    ax2.set_ylabel('Average Yield (Tonnes / Hectare)', color=color_yield, fontsize=10, fontweight='bold')
    line = ax2.plot(years + 0.15, yield_val, color=color_yield, marker='o', linewidth=2.5,
                    markersize=6, label='Avg Yield', zorder=4)
    ax2.tick_params(axis='y', labelcolor=color_yield)
    
    plt.title(f"Production Volume & Productivity Trends: {crop_name}", fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    return fig


def plot_rainfall_anomalies(anomaly_df: pd.DataFrame, state_name: str = 'National') -> plt.Figure:
    """
    Generates a waterfall/diverging bar plot of annual rainfall departures (%)
    highlighting drought deficits vs excess precipitation.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120)
    
    years = anomaly_df['Year'].astype(int)
    departures = anomaly_df['Departure_Pct']
    
    colors = ['#2E7D32' if d >= 0 else '#C62828' for d in departures]
    bars = ax.bar(years, departures, color=colors, width=0.55, edgecolor='none', zorder=3)
    
    # Add baseline and thresholds
    ax.axhline(0, color='#333333', linewidth=1.2, zorder=4)
    ax.axhline(10, color='#81C784', linestyle=':', linewidth=1.0, label='Mild Excess (+10%)')
    ax.axhline(-10, color='#EF5350', linestyle=':', linewidth=1.0, label='Mild Deficit (-10%)')
    ax.axhline(-20, color='#B71C1C', linestyle='--', linewidth=1.2, label='Severe Drought Deficit (-20%)')
    
    # Labels on bars
    for bar in bars:
        h = bar.get_height()
        va = 'bottom' if h >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2, h + (1.2 if h >= 0 else -2.5),
                f"{h:+.1f}%", ha='center', va=va, fontsize=8, fontweight='bold', color='#333333')
        
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel('Rainfall Departure from Normal (%)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Year', fontsize=10, fontweight='bold', labelpad=6)
    ax.set_title(f"Annual Rainfall Anomaly & Meteorological Risks: {state_name}", 
                 fontsize=12, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.legend(loc='upper right', frameon=True, fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_rainfall_vs_yield_regression(df: pd.DataFrame, crop_name: str) -> plt.Figure:
    """
    Renders scatter plot with linear regression trendline for rainfall vs yield.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    
    c_df = df[df['Crop'] == crop_name].dropna(subset=['Annual_Rainfall_mm', 'Yield_Tonnes_Per_Ha'])
    if len(c_df) < 3:
        ax.text(0.5, 0.5, "Insufficient data for regression", ha='center', va='center')
        return fig
        
    x = c_df['Annual_Rainfall_mm']
    y = c_df['Yield_Tonnes_Per_Ha']
    
    # Scatter
    ax.scatter(x, y, color='#1565C0', alpha=0.6, s=45, edgecolors='none', label='Observations')
    
    # Polynomial / Linear Fit
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), color='#D84315', linewidth=2.2, linestyle='-', label=f'Trend (Slope: {z[0]:.4f})')
    
    # Pearson r
    r = x.corr(y)
    ax.set_title(f"Rainfall Elasticity of Yield: {crop_name} (Pearson r = {r:.3f})", 
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Annual Rainfall (mm)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Crop Yield (Tonnes / Hectare)', fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True, fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_crop_leaderboard_horizontal(df_leaders: pd.DataFrame, top_n: int = 8) -> plt.Figure:
    """
    Horizontal grouped bar chart showing top crops ranked by total production.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    
    top_df = df_leaders.head(top_n).sort_values(by='Total_Production_Tonnes', ascending=True)
    y_pos = np.arange(len(top_df))
    prod_millions = top_df['Total_Production_Tonnes'] / 1e6
    
    bars = ax.barh(y_pos, prod_millions, color='#2E7D32', height=0.6, edgecolor='none', zorder=3)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.05 * prod_millions.max(), bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}M T", va='center', ha='left', fontsize=9, fontweight='bold', color='#2E7D32')
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df['Crop'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Total Harvested Production (Million Metric Tonnes)', fontsize=10, fontweight='bold')
    ax.set_title(f"Top {top_n} Agricultural Crops by Volume", fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
    
    plt.tight_layout()
    return fig


def plot_crop_radar_comparison(norm_matrix: pd.DataFrame) -> plt.Figure:
    """
    Draws a multi-axis Radar / Spider chart comparing crops across 5 key dimensions:
    Yield Level, Yield Stability, Water Efficiency, Fertilizer Efficiency, Acreage Scale.
    """
    set_plot_style()
    categories = ['Yield Level', 'Yield Stability', 'Water Efficiency', 'Fertilizer Efficiency', 'Acreage Scale']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete circle
    
    fig, ax = plt.subplots(figsize=(6.5, 6), subplot_kw=dict(polar=True), dpi=120)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=9, fontweight='bold', color='#333333')
    ax.set_rlabel_position(0)
    plt.yticks([25, 50, 75, 100], ["25", "50", "75", "100"], color="#777777", size=8)
    plt.ylim(0, 105)
    
    color_palette = ['#2E7D32', '#1565C0', '#E65100', '#6A1B9A', '#00838F']
    for idx, row in norm_matrix.iterrows():
        c_name = row['Crop']
        values = row[categories].values.flatten().tolist()
        values += values[:1]
        col = color_palette[idx % len(color_palette)]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=c_name, color=col)
        ax.fill(angles, values, color=col, alpha=0.15)
        
    plt.title("Comparative Agronomic Profile Index (0 - 100)", size=12, fontweight='bold', y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True, fontsize=9)
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    """
    Renders correlation matrix heatmap for all agricultural numeric metrics.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter only relevant columns
    cols = [c for c in num_cols if c not in ['Year', 'is_outlier']]
    corr = df[cols].corr()
    
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, shrink=0.8, pad=0.04)
    
    # Ticks
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    clean_labels = [c.replace('_', ' ') for c in cols]
    ax.set_xticklabels(clean_labels, rotation=45, ha='left', fontsize=8, fontweight='bold')
    ax.set_yticklabels(clean_labels, fontsize=8, fontweight='bold')
    
    # Annotations
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')
            
    ax.set_title("Agricultural Variables Correlation Matrix", fontsize=11, fontweight='bold', pad=25)
    plt.tight_layout()
    return fig

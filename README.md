# 🌾 AgriPulse: Agricultural Data Quality and Crop Productivity Analytics System

An end-to-end, interactive agricultural intelligence platform designed to audit and cleanse messy agricultural datasets, isolate multidimensional sensor and biological anomalies using machine learning, analyze rainfall patterns and drought vulnerability, benchmark crop productivity, and visualize agronomic trends.

---

## 🚀 Key Modules & Capabilities

### 1. 🧹 Agricultural Data Quality & Cleansing
- **Quality Health Scoring (0–100%)**: Evaluates datasets along 4 core dimensions:
  - **Completeness**: Quantifies missing values across yield, rainfall, and fertilizer records.
  - **Validity**: Detects physical impossibilities (negative rainfall, zero/negative acreage) and biological yield violations (e.g. wheat > 14 T/Ha).
  - **Uniqueness**: Identifies and removes duplicate harvest entries.
  - **Consistency**: Standardizes casing, whitespace, and regional naming aliases (e.g., `U.P.` $\rightarrow$ `Uttar Pradesh`).
- **Configurable Imputation**:
  - Hierarchical Group Median (Crop + State agronomic hierarchy).
  - Column Median and Mean.
  - **Scikit-Learn KNN Imputer** ($k=5$).
- **Mathematical Yield Recalculation**: Recomputes $\text{Yield} = \frac{\text{Production}}{\text{Area}}$ after cleaning to maintain integrity.

### 2. 🌲 Machine Learning Outlier Detection & Preprocessing (Scikit-Learn)
- **Isolation Forest (`sklearn.ensemble.IsolationForest`)**: Multidimensional anomaly isolation across Yield, Rainfall, Fertilizer, and Irrigation.
- **Local Outlier Factor (`sklearn.neighbors.LocalOutlierFactor`)**: Density-based local anomaly scoring.
- **Preprocessing Scalers (`sklearn.preprocessing.RobustScaler` / `StandardScaler`)**: Normalizes features with resilience against extreme values.
- **Univariate IQR Rule**: Classical interquartile range filtering for transparent bounding.
- **Remediation**: Instant pruning or quantile **Winsorization** (capping at 1st and 99th percentiles).

### 3. 🌧️ Rainfall Pattern & Climate Risk Analytics
- **Precipitation Climatology**: Multi-year baseline, monsoon share (%), and variability index.
- **Annual Rainfall Anomaly & Departure Tracking**: Categorizes annual precipitation into:
  - *Severe Drought / Deficit* ($< -20\%$)
  - *Mild Deficit* ($-20\%$ to $-10\%$)
  - *Normal* ($-10\%$ to $+10\%$)
  - *Excess / Flood Risk* ($> +20\%$)
- **Crop-Specific Rainfall Sensitivity**: Computes Pearson $r$ and Spearman $\rho$ correlation coefficients.
- **Water Productivity Index (WPI)**: Calculates kg of harvest output generated per millimeter of rainfall.

### 4. 🌾 Crop Production & Trend Analysis
- **Annual Harvest & Yield Dual-Axis Trends**: Tracks total production alongside yield efficiency.
- **Growth Rates**: Compound Annual Growth Rate (CAGR) and Year-over-Year (YoY) productivity changes.
- **Seasonal Distribution**: Performance breakdown across **Kharif**, **Rabi**, **Zaid**, and **Whole Year**.
- **Regional Leaderboard**: Identifies top-performing states and yield-efficient production zones.

### 5. ⚖️ Crop-wise Head-to-Head Comparisons
- **Agronomic Profile Radar Chart**: 5-axis normalized spider comparison (Yield Level, Yield Stability, Water Efficiency, Fertilizer Efficiency, Acreage Scale).
- **Yield Volatility (Risk)**: Evaluates harvest stability via Coefficient of Variation ($\text{CV} = \frac{\sigma}{\mu} \times 100\%$).
- **Resource Conversion Efficiency**: Compares kg of yield per kg of fertilizer applied.

### 6. 📊 Publication-Grade Matplotlib Visualizations
- Custom styled, high-contrast agricultural charts (emerald greens, earth ambers, ocean blues).
- Dual-axis trend plots, diverging waterfall anomaly charts, scatter plots with OLS regression lines, horizontal ranking bars, correlation heatmaps, and polar radar charts.

---

## 🛠️ Technology Stack

| Component | Library / Framework | Primary Purpose |
|---|---|---|
| **Programming Language** | **Python 3.9+** | Core application logic and execution engine |
| **Data Manipulation** | **Pandas** | Tabular data wrangling, cleaning, grouping, and aggregations |
| **Numerical Calculations** | **NumPy** | Array computations, statistical metrics, regression fitting |
| **Data Visualization** | **Matplotlib** | High-DPI, publication-quality agronomic charts and plots |
| **Machine Learning** | **Scikit-Learn** | `IsolationForest`, `LocalOutlierFactor`, `RobustScaler`, `StandardScaler`, `KNNImputer` |
| **Interactive Dashboard** | **Streamlit** | Multi-tab UI, reactive filters, metric cards, and file downloaders |

---

## 📁 Project Directory Structure

```text
agripulse/
│
├── app.py                      # Main Streamlit dashboard application
├── requirements.txt            # Python dependencies
├── README.md                   # Complete system documentation
├── run.bat                     # Windows one-click runner batch script
├── run.ps1                     # PowerShell one-click runner script
│
├── data/                       # Datasets directory
│   ├── crop_production_rainfall_clean.csv   # Pre-sanitized baseline dataset
│   ├── crop_production_rainfall_raw.csv     # Noisy benchmark with injected anomalies
│   ├── generate_data.ps1       # Benchmark dataset generator script
│   └── sample_generator.py     # Python agricultural dataset generator
│
└── src/                        # Modular source code
    ├── __init__.py             # Package marker
    ├── data_cleaning.py        # Quality auditing, imputation & sanitization
    ├── outlier_detection.py    # Scikit-learn Isolation Forest, LOF & Scalers
    ├── crop_analytics.py       # Production trends, KPIs, CAGR, leaderboard
    ├── rainfall_analytics.py   # Rainfall departures, drought tracking, WPI
    ├── crop_comparison.py      # Multi-crop radar index, stability & efficiency
    ├── visualization.py        # Custom Matplotlib plotting engine
    └── utils.py                # Number formatting, file loaders, CSV export
```

---

## ⚡ Quick Start & Installation

### Step 1: Open Terminal in Project Directory
```powershell
cd C:\Users\Brindha\.gemini\antigravity\scratch\agripulse
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch the AgriPulse Dashboard
Run using Streamlit:
```bash
streamlit run app.py
```
*Or double-click `run.bat` / run `./run.ps1` in PowerShell.*

The interactive dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## 📖 Feature Walkthrough & Tab Guide

1. **Executive Overview**: Immediate view of national production KPIs (Total Volume, Harvested Land, Weighted Yield, Mean Rainfall, Quality Index), historical trends, and seasonal distribution.
2. **Data Quality & Cleaning**: View the composite Data Health Scorecard, inspect missing/invalid entries, select your imputation strategy (e.g. hierarchical group median), execute cleaning, and compare before/after deltas.
3. **Scikit-Learn Outliers**: Tune contamination rates, pick between Isolation Forest or LOF, visualize anomalies on 2D scatter plots, and prune or Winsorize records.
4. **Rainfall & Yield Analysis**: Inspect annual departure waterfall charts, identify severe drought years, and explore crop-specific rainfall regression lines.
5. **Crop Comparisons**: Select any 2 to 5 crops to display a 5-dimension radar profile and side-by-side efficiency benchmarks.
6. **Data Explorer & Export**: Search and filter tabular records, view correlation heatmaps, and download cleaned datasets as CSV.

---

## 📜 License
AgriPulse is developed for agricultural research, data quality monitoring, and agronomic productivity analytics.

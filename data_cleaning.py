"""
AgriPulse - Data Cleaning and Quality Assurance Module
Performs comprehensive data auditing, missing value imputation, out-of-bounds sanitation,
text normalization, duplicate handling, and data quality scoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any

# Biological upper bounds for yields (tonnes per hectare) to detect impossible values
CROP_YIELD_LIMITS = {
    'Sugarcane': 150.0,
    'Rice': 15.0,
    'Wheat': 14.0,
    'Maize': 16.0,
    'Cotton': 6.0,
    'Soybean': 6.0,
    'Groundnut': 6.0,
    'Mustard': 5.0,
    'Pulses': 4.5,
    'Millets': 5.0,
    'DEFAULT': 20.0
}

STATE_ALIASES = {
    'u.p.': 'Uttar Pradesh',
    'up': 'Uttar Pradesh',
    'uttar pradesh': 'Uttar Pradesh',
    'punjab': 'Punjab',
    'haryana': 'Haryana',
    'maharashtra': 'Maharashtra',
    'gujarat': 'Gujarat',
    'andhra pradesh': 'Andhra Pradesh',
    'karnataka': 'Karnataka',
    'tamil nadu': 'Tamil Nadu',
    'west bengal': 'West Bengal',
    'madhya pradesh': 'Madhya Pradesh'
}


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes a comprehensive data quality audit report and score (0-100%).
    Evaluates:
    - Completeness (% of non-missing cells)
    - Uniqueness (% of non-duplicate records)
    - Validity (% of values within valid physical/biological ranges)
    - Consistency (text casing & formatting consistency)
    """
    total_rows = len(df)
    total_cells = df.size
    if total_rows == 0:
        return {'score': 0, 'total_rows': 0, 'issues': ['Dataset is empty']}
    
    # 1. Completeness
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / total_rows) * 100
    total_missing_cells = df.isnull().sum().sum()
    completeness_score = max(0.0, 100.0 - (total_missing_cells / total_cells * 100))
    
    missing_summary = pd.DataFrame({
        'Missing Values': missing_counts,
        'Percentage (%)': missing_pct.round(2)
    })
    missing_summary = missing_summary[missing_summary['Missing Values'] > 0]
    
    # 2. Uniqueness
    duplicate_count = int(df.duplicated().sum())
    uniqueness_score = max(0.0, 100.0 - (duplicate_count / total_rows * 100))
    
    # 3. Validity (Range and Physical checks)
    invalid_records = 0
    validity_issues = []
    
    # Negative rainfall
    if 'Annual_Rainfall_mm' in df.columns:
        neg_rf = (df['Annual_Rainfall_mm'] < 0).sum()
        if neg_rf > 0:
            invalid_records += neg_rf
            validity_issues.append(f"{neg_rf} records with negative Annual Rainfall (< 0 mm)")
            
    if 'Monsoon_Rainfall_mm' in df.columns:
        neg_mrf = (df['Monsoon_Rainfall_mm'] < 0).sum()
        if neg_mrf > 0:
            invalid_records += neg_mrf
            validity_issues.append(f"{neg_mrf} records with negative Monsoon Rainfall")
            
    # Negative production or area
    if 'Production_Tonnes' in df.columns:
        neg_prod = (df['Production_Tonnes'] < 0).sum()
        if neg_prod > 0:
            invalid_records += neg_prod
            validity_issues.append(f"{neg_prod} records with negative Production")
            
    if 'Area_Hectares' in df.columns:
        neg_area = (df['Area_Hectares'] <= 0).sum()
        if neg_area > 0:
            invalid_records += neg_area
            validity_issues.append(f"{neg_area} records with zero or negative Area")
            
    # Impossible Yields
    if 'Yield_Tonnes_Per_Ha' in df.columns and 'Crop' in df.columns:
        impossible_yields = 0
        for crop_name, group in df.groupby(df['Crop'].astype(str).str.strip().str.title()):
            max_limit = CROP_YIELD_LIMITS.get(crop_name, CROP_YIELD_LIMITS['DEFAULT'])
            exceeded = (group['Yield_Tonnes_Per_Ha'] > max_limit).sum()
            impossible_yields += exceeded
        if impossible_yields > 0:
            invalid_records += impossible_yields
            validity_issues.append(f"{impossible_yields} records exceed biological maximum crop yields")
            
    validity_score = max(0.0, 100.0 - (invalid_records / (total_rows * 2) * 100))
    
    # 4. Consistency (Casing and Whitespace in Categorical Columns)
    inconsistency_count = 0
    for col in ['State', 'Crop', 'Season', 'Soil_Type']:
        if col in df.columns:
            raw_uniques = df[col].dropna().nunique()
            cleaned_uniques = df[col].dropna().astype(str).str.strip().str.title().nunique()
            if raw_uniques > cleaned_uniques:
                inconsistencies = raw_uniques - cleaned_uniques
                inconsistency_count += inconsistencies
                validity_issues.append(f"Inconsistent text casing/spaces detected in '{col}' ({inconsistencies} redundant variants)")
                
    consistency_score = max(0.0, 100.0 - (inconsistency_count * 5.0))
    
    # Composite Quality Score (Weighted Average)
    overall_score = (
        completeness_score * 0.35 +
        validity_score * 0.30 +
        uniqueness_score * 0.20 +
        consistency_score * 0.15
    )
    
    return {
        'overall_score': round(overall_score, 1),
        'completeness_score': round(completeness_score, 1),
        'validity_score': round(validity_score, 1),
        'uniqueness_score': round(uniqueness_score, 1),
        'consistency_score': round(consistency_score, 1),
        'total_rows': total_rows,
        'total_columns': df.shape[1],
        'total_missing_cells': int(total_missing_cells),
        'duplicate_rows': duplicate_count,
        'invalid_records_count': invalid_records,
        'missing_summary': missing_summary,
        'validity_issues': validity_issues
    }


def standardize_text(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Trims whitespace, unifies title casing, and standardizes common state/crop abbreviations.
    """
    df_clean = df.copy()
    changes_count = 0
    
    for col in df_clean.select_dtypes(include=['object', 'category']).columns:
        original = df_clean[col].copy()
        # Strip and Title Case
        df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
        
        # Apply Aliases if state column
        if col.lower() in ['state', 'state_name']:
            df_clean[col] = df_clean[col].apply(
                lambda x: STATE_ALIASES.get(str(x).lower().strip(), x) if pd.notna(x) else x
            )
            
        changes_count += int((df_clean[col] != original.astype(str)).sum())
        
    return df_clean, changes_count


def sanitize_numerical_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Detects and sanitizes negative physical values (rainfall, fertilizer, area, production)
    and biological impossibilities by setting invalid values to NaN so they can be imputed properly.
    """
    df_clean = df.copy()
    audit_notes = []
    
    # Rainfall range (0 to 6000 mm)
    for col in ['Annual_Rainfall_mm', 'Monsoon_Rainfall_mm']:
        if col in df_clean.columns:
            neg_mask = df_clean[col] < 0
            neg_count = neg_mask.sum()
            if neg_count > 0:
                df_clean.loc[neg_mask, col] = np.nan
                audit_notes.append(f"Sanitized {neg_count} negative values in '{col}' to NaN.")
                
            extreme_mask = df_clean[col] > 6000.0
            extreme_count = extreme_mask.sum()
            if extreme_count > 0:
                df_clean.loc[extreme_mask, col] = np.nan
                audit_notes.append(f"Sanitized {extreme_count} extreme rainfall outliers (>6000mm) in '{col}' to NaN.")
                
    # Production and Area (> 0)
    if 'Production_Tonnes' in df_clean.columns:
        neg_prod_mask = df_clean['Production_Tonnes'] < 0
        neg_prod = neg_prod_mask.sum()
        if neg_prod > 0:
            df_clean.loc[neg_prod_mask, 'Production_Tonnes'] = np.nan
            audit_notes.append(f"Sanitized {neg_prod} negative Production records to NaN.")
            
    if 'Area_Hectares' in df_clean.columns:
        neg_area_mask = df_clean['Area_Hectares'] <= 0
        neg_area = neg_area_mask.sum()
        if neg_area > 0:
            df_clean.loc[neg_area_mask, 'Area_Hectares'] = np.nan
            audit_notes.append(f"Sanitized {neg_area} zero/negative Area records to NaN.")
            
    # Biological Yield Bounds
    if 'Yield_Tonnes_Per_Ha' in df_clean.columns and 'Crop' in df_clean.columns:
        crop_clean = df_clean['Crop'].astype(str).str.strip().str.title()
        impossible_mask = pd.Series(False, index=df_clean.index)
        
        for crop_name, max_limit in CROP_YIELD_LIMITS.items():
            if crop_name == 'DEFAULT':
                continue
            c_mask = (crop_clean == crop_name) & (df_clean['Yield_Tonnes_Per_Ha'] > max_limit)
            impossible_mask = impossible_mask | c_mask
            
        def_mask = (~crop_clean.isin(CROP_YIELD_LIMITS.keys())) & (df_clean['Yield_Tonnes_Per_Ha'] > CROP_YIELD_LIMITS['DEFAULT'])
        impossible_mask = impossible_mask | def_mask
        
        imp_count = impossible_mask.sum()
        if imp_count > 0:
            df_clean.loc[impossible_mask, 'Yield_Tonnes_Per_Ha'] = np.nan
            if 'Production_Tonnes' in df_clean.columns:
                df_clean.loc[impossible_mask, 'Production_Tonnes'] = np.nan
            audit_notes.append(f"Sanitized {imp_count} biologically impossible yields (> upper threshold) to NaN.")
            
    return df_clean, audit_notes


def impute_missing_data(df: pd.DataFrame, strategy: str = 'group_median') -> Tuple[pd.DataFrame, List[str]]:
    """
    Imputes missing values using chosen statistical strategy:
    - 'group_median': Crop & State grouped median (most accurate for agricultural data)
    - 'mean': Column-wise mean
    - 'median': Column-wise median
    - 'drop': Drops rows with any missing numerical values
    - 'knn': Scikit-learn KNNImputer
    """
    df_clean = df.copy()
    audit_notes = []
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == 'drop':
        initial_len = len(df_clean)
        df_clean = df_clean.dropna().reset_index(drop=True)
        dropped = initial_len - len(df_clean)
        audit_notes.append(f"Dropped {dropped} rows containing missing values.")
        return df_clean, audit_notes
        
    elif strategy == 'group_median' and 'Crop' in df_clean.columns:
        for col in num_cols:
            if df_clean[col].isnull().sum() > 0:
                missing_before = df_clean[col].isnull().sum()
                # 1st pass: Crop + State median
                if 'State' in df_clean.columns:
                    df_clean[col] = df_clean.groupby(['Crop', 'State'])[col].transform(
                        lambda s: s.fillna(s.median())
                    )
                # 2nd pass: Crop median
                df_clean[col] = df_clean.groupby('Crop')[col].transform(
                    lambda s: s.fillna(s.median())
                )
                # 3rd pass: Global median fallback
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                audit_notes.append(f"Imputed {missing_before} missing values in '{col}' using Group Median (Crop/State hierarchy).")
                
    elif strategy == 'knn':
        from sklearn.impute import KNNImputer
        missing_count = df_clean[num_cols].isnull().sum().sum()
        if missing_count > 0:
            imputer = KNNImputer(n_neighbors=5)
            df_clean[num_cols] = imputer.fit_transform(df_clean[num_cols])
            audit_notes.append(f"Imputed {missing_count} missing values using Scikit-Learn KNNImputer (k=5).")
            
    else:  # mean or median
        for col in num_cols:
            missing_before = df_clean[col].isnull().sum()
            if missing_before > 0:
                fill_val = df_clean[col].median() if strategy == 'median' else df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(fill_val)
                audit_notes.append(f"Imputed {missing_before} missing values in '{col}' using global {strategy} ({round(fill_val, 2)}).")
                
    return df_clean, audit_notes


def clean_pipeline(df: pd.DataFrame, 
                   imputation_strategy: str = 'group_median',
                   deduplicate: bool = True,
                   recompute_yield: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete end-to-end cleaning pipeline.
    Returns cleaned dataframe and audit report.
    """
    initial_audit = audit_data_quality(df)
    log_actions = []
    
    # Step 1: Remove exact duplicates
    working_df = df.copy()
    if deduplicate:
        dup_count = working_df.duplicated().sum()
        if dup_count > 0:
            working_df = working_df.drop_duplicates().reset_index(drop=True)
            log_actions.append(f"Removed {dup_count} exact duplicate rows.")
            
    # Step 2: Standardize text columns
    working_df, text_mods = standardize_text(working_df)
    if text_mods > 0:
        log_actions.append(f"Standardized text casing, whitespace, and aliases across categorical fields ({text_mods} edits).")
        
    # Step 3: Sanitize impossible / out-of-range physical values
    working_df, sanity_notes = sanitize_numerical_ranges(working_df)
    log_actions.extend(sanity_notes)
    
    # Step 4: Impute missing data
    working_df, impute_notes = impute_missing_data(working_df, strategy=imputation_strategy)
    log_actions.extend(impute_notes)
    
    # Step 5: Recalculate yield if area and production are present
    if recompute_yield and 'Production_Tonnes' in working_df.columns and 'Area_Hectares' in working_df.columns:
        valid_mask = (working_df['Area_Hectares'] > 0) & (working_df['Production_Tonnes'] >= 0)
        working_df.loc[valid_mask, 'Yield_Tonnes_Per_Ha'] = (
            working_df.loc[valid_mask, 'Production_Tonnes'] / working_df.loc[valid_mask, 'Area_Hectares']
        ).round(3)
        log_actions.append("Recalculated 'Yield_Tonnes_Per_Ha' as (Production / Area) for mathematical consistency.")
        
    final_audit = audit_data_quality(working_df)
    
    report = {
        'initial_audit': initial_audit,
        'final_audit': final_audit,
        'actions_performed': log_actions,
        'quality_improvement': round(final_audit['overall_score'] - initial_audit['overall_score'], 1)
    }
    
    return working_df, report

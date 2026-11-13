"""
AgriPulse - Synthetic Realistic Agricultural Dataset Generator
Generates multi-year, multi-state crop production and rainfall records with
realistic agricultural distributions and injected data quality anomalies
to demonstrate data cleaning and outlier detection.
"""

import numpy as np
import pandas as pd
import os

def generate_agricultural_dataset(n_samples=1800, random_state=42, inject_anomalies=True):
    np.random.seed(random_state)
    
    years = list(range(2013, 2025))
    states_data = {
        'Punjab': {'soil': 'Alluvial', 'rainfall_base': 650, 'rainfall_std': 120, 'temp_base': 24.5, 'irrigation_base': 95},
        'Haryana': {'soil': 'Alluvial', 'rainfall_base': 580, 'rainfall_std': 110, 'temp_base': 25.0, 'irrigation_base': 90},
        'Uttar Pradesh': {'soil': 'Alluvial', 'rainfall_base': 950, 'rainfall_std': 180, 'temp_base': 26.0, 'irrigation_base': 82},
        'Madhya Pradesh': {'soil': 'Black', 'rainfall_base': 1050, 'rainfall_std': 220, 'temp_base': 25.5, 'irrigation_base': 55},
        'Maharashtra': {'soil': 'Black', 'rainfall_base': 1150, 'rainfall_std': 250, 'temp_base': 27.0, 'irrigation_base': 42},
        'Gujarat': {'soil': 'Sandy Loam', 'rainfall_base': 820, 'rainfall_std': 210, 'temp_base': 27.5, 'irrigation_base': 60},
        'Andhra Pradesh': {'soil': 'Red', 'rainfall_base': 920, 'rainfall_std': 190, 'temp_base': 28.5, 'irrigation_base': 65},
        'Karnataka': {'soil': 'Red', 'rainfall_base': 1200, 'rainfall_std': 280, 'temp_base': 26.5, 'irrigation_base': 48},
        'Tamil Nadu': {'soil': 'Red', 'rainfall_base': 980, 'rainfall_std': 200, 'temp_base': 29.0, 'irrigation_base': 70},
        'West Bengal': {'soil': 'Alluvial', 'rainfall_base': 1650, 'rainfall_std': 320, 'temp_base': 26.8, 'irrigation_base': 68}
    }
    
    crops_info = {
        'Wheat': {'season': 'Rabi', 'base_yield': 3.8, 'yield_std': 0.6, 'fert_base': 160, 'water_req': 450},
        'Rice': {'season': 'Kharif', 'base_yield': 3.2, 'yield_std': 0.7, 'fert_base': 140, 'water_req': 1200},
        'Maize': {'season': 'Kharif', 'base_yield': 2.9, 'yield_std': 0.5, 'fert_base': 120, 'water_req': 600},
        'Sugarcane': {'season': 'Whole Year', 'base_yield': 68.0, 'yield_std': 9.0, 'fert_base': 220, 'water_req': 1800},
        'Cotton': {'season': 'Kharif', 'base_yield': 1.6, 'yield_std': 0.35, 'fert_base': 130, 'water_req': 700},
        'Pulses': {'season': 'Rabi', 'base_yield': 0.95, 'yield_std': 0.22, 'fert_base': 50, 'water_req': 350},
        'Soybean': {'season': 'Kharif', 'base_yield': 1.7, 'yield_std': 0.35, 'fert_base': 90, 'water_req': 550},
        'Groundnut': {'season': 'Kharif', 'base_yield': 1.8, 'yield_std': 0.4, 'fert_base': 85, 'water_req': 500},
        'Mustard': {'season': 'Rabi', 'base_yield': 1.4, 'yield_std': 0.28, 'fert_base': 75, 'water_req': 380},
        'Millets': {'season': 'Kharif', 'base_yield': 1.2, 'yield_std': 0.25, 'fert_base': 45, 'water_req': 300}
    }
    
    records = []
    states_list = list(states_data.keys())
    crops_list = list(crops_info.keys())
    
    for _ in range(n_samples):
        yr = int(np.random.choice(years))
        state = np.random.choice(states_list)
        crop = np.random.choice(crops_list)
        s_info = states_data[state]
        c_info = crops_info[crop]
        
        # Environmental factors
        rainfall = max(200.0, np.random.normal(s_info['rainfall_base'], s_info['rainfall_std']))
        monsoon_share = np.random.uniform(0.65, 0.85)
        monsoon_rainfall = rainfall * monsoon_share
        temperature = np.random.normal(s_info['temp_base'], 1.8)
        irrigation_pct = np.clip(np.random.normal(s_info['irrigation_base'], 8.0), 10.0, 99.0)
        
        # Fertilizer
        fertilizer = max(20.0, np.random.normal(c_info['fert_base'], 25.0))
        
        # Area in hectares (log-normal distribution)
        area = float(np.round(np.random.lognormal(mean=9.5, sigma=0.85), 1))
        area = np.clip(area, 200.0, 180000.0)
        
        # Yield calculation with weather dependency & irrigation buffer
        water_ratio = min(rainfall / c_info['water_req'], 1.6)
        water_impact = 0.7 + 0.3 * np.tanh(water_ratio)
        irrigation_buffer = 0.8 + 0.2 * (irrigation_pct / 100.0)
        weather_factor = water_impact * irrigation_buffer
        
        # Base yield with noise
        simulated_yield = c_info['base_yield'] * weather_factor * np.random.normal(1.0, 0.12)
        simulated_yield = max(0.2, simulated_yield)
        
        # Total production in tonnes
        production = float(np.round(area * simulated_yield, 2))
        yield_per_ha = float(np.round(production / area, 3))
        
        records.append({
            'Year': yr,
            'State': state,
            'Crop': crop,
            'Season': c_info['season'],
            'Soil_Type': s_info['soil'],
            'Area_Hectares': area,
            'Production_Tonnes': production,
            'Yield_Tonnes_Per_Ha': yield_per_ha,
            'Annual_Rainfall_mm': round(rainfall, 1),
            'Monsoon_Rainfall_mm': round(monsoon_rainfall, 1),
            'Temperature_Celsius': round(temperature, 1),
            'Fertilizer_Usage_Kg_Per_Ha': round(fertilizer, 1),
            'Irrigation_Coverage_Pct': round(irrigation_pct, 1)
        })
        
    df = pd.DataFrame(records)
    
    # Save a clean copy
    clean_df = df.copy()
    
    if inject_anomalies:
        # 1. Inconsistent state names and casing
        noisy_states = {'Punjab': ['punjab', 'PUNJAB', 'Punjab '],
                        'Uttar Pradesh': ['U.P.', 'uttar pradesh', 'Uttar Pradesh  '],
                        'Maharashtra': ['maharashtra', 'MAHARASHTRA', 'Maharashtra ']}
        for orig, variants in noisy_states.items():
            mask = df['State'] == orig
            indices = df[mask].sample(frac=0.18, random_state=random_state).index
            df.loc[indices, 'State'] = np.random.choice(variants, size=len(indices))
            
        # 2. Inconsistent crop casing and spaces
        noisy_crops = {'Wheat': ['wheat', 'WHEAT', 'Wheat '],
                       'Rice': ['rice', 'RICE', 'Rice '],
                       'Cotton': ['cotton', 'COTTON', 'Cotton ']}
        for orig, variants in noisy_crops.items():
            mask = df['Crop'] == orig
            indices = df[mask].sample(frac=0.15, random_state=random_state).index
            df.loc[indices, 'Crop'] = np.random.choice(variants, size=len(indices))
            
        # 3. Missing values injection (approx 3-5% in key numerical columns)
        nan_cols = ['Production_Tonnes', 'Annual_Rainfall_mm', 'Fertilizer_Usage_Kg_Per_Ha', 'Irrigation_Coverage_Pct']
        for col in nan_cols:
            sample_idx = df.sample(frac=0.035, random_state=random_state).index
            df.loc[sample_idx, col] = np.nan
            
        # 4. Out-of-bounds / impossible entries
        # Negative rainfall (sensor error)
        neg_rf_idx = df.sample(n=6, random_state=random_state).index
        df.loc[neg_rf_idx, 'Annual_Rainfall_mm'] = -np.random.uniform(50, 400, size=6)
        
        # Negative yield or production
        neg_prod_idx = df.sample(n=4, random_state=random_state).index
        df.loc[neg_prod_idx, 'Production_Tonnes'] = -np.random.uniform(100, 2500, size=4)
        
        # Extreme biological outliers (e.g. typing error with extra 0 or huge numbers)
        extreme_yield_idx = df[df['Crop'] == 'Wheat'].sample(n=5, random_state=random_state).index
        df.loc[extreme_yield_idx, 'Yield_Tonnes_Per_Ha'] = np.random.uniform(45.0, 95.0, size=len(extreme_yield_idx))
        df.loc[extreme_yield_idx, 'Production_Tonnes'] = df.loc[extreme_yield_idx, 'Area_Hectares'] * df.loc[extreme_yield_idx, 'Yield_Tonnes_Per_Ha']
        
        # Extreme high rainfall outlier (e.g. 9800 mm flash error)
        rf_outlier_idx = df.sample(n=4, random_state=random_state).index
        df.loc[rf_outlier_idx, 'Annual_Rainfall_mm'] = np.random.uniform(7500, 11000, size=4)
        
        # 5. Exact duplicate rows
        dup_rows = df.sample(n=25, random_state=random_state)
        df = pd.concat([df, dup_rows], ignore_index=True)
        
    return clean_df, df

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    clean_df, raw_df = generate_agricultural_dataset()
    
    clean_path = os.path.join(data_dir, 'crop_production_rainfall_clean.csv')
    raw_path = os.path.join(data_dir, 'crop_production_rainfall_raw.csv')
    
    clean_df.to_csv(clean_path, index=False)
    raw_df.to_csv(raw_path, index=False)
    print(f"Generated clean dataset: {clean_path} with {len(clean_df)} rows.")
    print(f"Generated raw noisy dataset: {raw_path} with {len(raw_df)} rows.")

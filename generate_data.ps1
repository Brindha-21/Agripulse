$years = 2013..2024
$states = @(
  @{name='Punjab'; soil='Alluvial'; rf_base=650; rf_sd=120; temp=24.5; irr=95},
  @{name='Haryana'; soil='Alluvial'; rf_base=580; rf_sd=110; temp=25.0; irr=90},
  @{name='Uttar Pradesh'; soil='Alluvial'; rf_base=950; rf_sd=180; temp=26.0; irr=82},
  @{name='Madhya Pradesh'; soil='Black'; rf_base=1050; rf_sd=220; temp=25.5; irr=55},
  @{name='Maharashtra'; soil='Black'; rf_base=1150; rf_sd=250; temp=27.0; irr=42},
  @{name='Gujarat'; soil='Sandy Loam'; rf_base=820; rf_sd=210; temp=27.5; irr=60},
  @{name='Andhra Pradesh'; soil='Red'; rf_base=920; rf_sd=190; temp=28.5; irr=65},
  @{name='Karnataka'; soil='Red'; rf_base=1200; rf_sd=280; temp=26.5; irr=48},
  @{name='Tamil Nadu'; soil='Red'; rf_base=980; rf_sd=200; temp=29.0; irr=70},
  @{name='West Bengal'; soil='Alluvial'; rf_base=1650; rf_sd=320; temp=26.8; irr=68}
)
$crops = @(
  @{name='Wheat'; season='Rabi'; y_base=3.8; fert=160; water=450},
  @{name='Rice'; season='Kharif'; y_base=3.2; fert=140; water=1200},
  @{name='Maize'; season='Kharif'; y_base=2.9; fert=120; water=600},
  @{name='Sugarcane'; season='Whole Year'; y_base=68.0; fert=220; water=1800},
  @{name='Cotton'; season='Kharif'; y_base=1.6; fert=130; water=700},
  @{name='Pulses'; season='Rabi'; y_base=0.95; fert=50; water=350},
  @{name='Soybean'; season='Kharif'; y_base=1.7; fert=90; water=550},
  @{name='Groundnut'; season='Kharif'; y_base=1.8; fert=85; water=500},
  @{name='Mustard'; season='Rabi'; y_base=1.4; fert=75; water=380},
  @{name='Millets'; season='Kharif'; y_base=1.2; fert=45; water=300}
)

$rnd = New-Object System.Random(42)
$cleanLines = [System.Collections.Generic.List[string]]::new()
$cleanLines.Add("Year,State,Crop,Season,Soil_Type,Area_Hectares,Production_Tonnes,Yield_Tonnes_Per_Ha,Annual_Rainfall_mm,Monsoon_Rainfall_mm,Temperature_Celsius,Fertilizer_Usage_Kg_Per_Ha,Irrigation_Coverage_Pct")

$rawLines = [System.Collections.Generic.List[string]]::new()
$rawLines.Add("Year,State,Crop,Season,Soil_Type,Area_Hectares,Production_Tonnes,Yield_Tonnes_Per_Ha,Annual_Rainfall_mm,Monsoon_Rainfall_mm,Temperature_Celsius,Fertilizer_Usage_Kg_Per_Ha,Irrigation_Coverage_Pct")

for ($i = 0; $i -lt 1200; $i++) {
    $y = $years[$rnd.Next($years.Length)]
    $s = $states[$rnd.Next($states.Length)]
    $c = $crops[$rnd.Next($crops.Length)]
    
    $rf = [Math]::Max(250.0, [Math]::Round($s.rf_base + ($rnd.NextDouble() * 2 - 1) * $s.rf_sd, 1))
    $monsoon_share = 0.65 + ($rnd.NextDouble() * 0.18)
    $m_rf = [Math]::Round($rf * $monsoon_share, 1)
    $temp = [Math]::Round($s.temp + ($rnd.NextDouble() * 4 - 2), 1)
    $irr = [Math]::Min(98.0, [Math]::Max(15.0, [Math]::Round($s.irr + ($rnd.NextDouble() * 10 - 5), 1)))
    $fert = [Math]::Max(20.0, [Math]::Round($c.fert + ($rnd.NextDouble() * 30 - 15), 1))
    
    $area = [Math]::Round(2000 + ($rnd.NextDouble() * 85000), 1)
    $w_factor = [Math]::Min($rf / $c.water, 1.4)
    $eff = 0.75 + 0.25 * $w_factor + ($irr / 100.0) * 0.15
    $yld = [Math]::Max(0.3, [Math]::Round($c.y_base * $eff * (0.9 + $rnd.NextDouble() * 0.2), 3))
    $prod = [Math]::Round($area * $yld, 2)
    
    $cLine = "$y,$($s.name),$($c.name),$($c.season),$($s.soil),$area,$prod,$yld,$rf,$m_rf,$temp,$fert,$irr"
    $cleanLines.Add($cLine)
    
    # Injected raw variants
    $r_state = $s.name
    $r_crop = $c.name
    $r_rf = [string]$rf
    $r_prod = [string]$prod
    $r_yld = [string]$yld
    $r_fert = [string]$fert
    $r_irr = [string]$irr
    
    # Casing / whitespace
    if ($rnd.Next(100) -lt 8) {
        if ($r_state -eq 'Punjab') { $r_state = 'punjab ' }
        elseif ($r_state -eq 'Uttar Pradesh') { $r_state = 'U.P.' }
        elseif ($r_state -eq 'Maharashtra') { $r_state = 'MAHARASHTRA' }
    }
    if ($rnd.Next(100) -lt 7) {
        if ($r_crop -eq 'Wheat') { $r_crop = 'wheat ' }
        elseif ($r_crop -eq 'Rice') { $r_crop = 'RICE' }
        elseif ($r_crop -eq 'Cotton') { $r_crop = 'cotton' }
    }
    # Missing
    if ($rnd.Next(100) -lt 4) { $r_prod = '' }
    if ($rnd.Next(100) -lt 3) { $r_rf = '' }
    if ($rnd.Next(100) -lt 3) { $r_fert = '' }
    # Out of bounds
    if ($i -in @(15, 88, 245, 510)) { $r_rf = (-1 * [Math]::Round($rf * 0.5, 1)).ToString() }
    if ($i -in @(32, 190, 420)) { $r_prod = (-1 * [Math]::Round($prod * 0.4, 2)).ToString() }
    if ($i -in @(55, 330, 715)) { 
        $r_yld = '88.5'
        $r_prod = ([Math]::Round($area * 88.5, 2)).ToString()
    }
    
    $rLine = "$y,$r_state,$r_crop,$($c.season),$($s.soil),$area,$r_prod,$r_yld,$r_rf,$m_rf,$temp,$r_fert,$r_irr"
    $rawLines.Add($rLine)
}

# Duplicate rows in raw
for ($k = 0; $k -lt 25; $k++) {
    $rawLines.Add($rawLines[$k + 10])
}

$cleanPath = "C:\Users\Brindha\.gemini\antigravity\scratch\agripulse\data\crop_production_rainfall_clean.csv"
$rawPath = "C:\Users\Brindha\.gemini\antigravity\scratch\agripulse\data\crop_production_rainfall_raw.csv"

[System.IO.File]::WriteAllLines($cleanPath, $cleanLines)
[System.IO.File]::WriteAllLines($rawPath, $rawLines)
Write-Output "Generated clean dataset with $($cleanLines.Count) rows at $cleanPath"
Write-Output "Generated raw dataset with $($rawLines.Count) rows at $rawPath"

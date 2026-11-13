$raw = Import-Csv "C:\Users\Brindha\.gemini\antigravity\scratch\agripulse\data\crop_production_rainfall_raw.csv"
Write-Output "Record 0:"
$raw[0] | Format-List
Write-Output "Total raw rows: $($raw.Count)"

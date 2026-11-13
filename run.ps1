Write-Host "========================================================" -ForegroundColor Green
Write-Host "      Starting AgriPulse Analytics Dashboard           " -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (Get-Command streamlit -ErrorAction SilentlyContinue) {
    streamlit run app.py --server.port 8501
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    py -m streamlit run app.py --server.port 8501
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python -m streamlit run app.py --server.port 8501
} else {
    Write-Host "[ERROR] Neither streamlit, py, nor python is recognized in PATH." -ForegroundColor Red
    Write-Host "Please check that Python is installed with 'Add to PATH'." -ForegroundColor Yellow
}

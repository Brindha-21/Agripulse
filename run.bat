@echo off
title AgriPulse Analytics System
echo ========================================================
echo       Starting AgriPulse Analytics Dashboard
echo ========================================================
echo.
cd /d "%~dp0"
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using Python Launcher (py)...
    py -m streamlit run app.py --server.port 8501
    goto done
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using Python (python)...
    python -m streamlit run app.py --server.port 8501
    goto done
)

where streamlit >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Using Streamlit directly...
    streamlit run app.py --server.port 8501
    goto done
)

echo [ERROR] Neither py, python, nor streamlit was found in your PATH.
echo Please ensure Python is installed with 'Add to PATH' checked.
pause
exit /b 1

:done
pause

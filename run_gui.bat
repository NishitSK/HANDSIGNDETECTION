@echo off
title ISL Translation System GUI
echo ======================================================
echo   Indian Sign Language Translation System - GUI
echo ======================================================
echo Starting application...

:: Suppress TensorFlow verbose oneDNN/C++ logs for cleaner, faster startup
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0

:: Navigate to project directory containing src\gui_app.py
if exist "%~dp0mini_project 22\src\gui_app.py" (
    cd /d "%~dp0mini_project 22"
) else if exist "%~dp0mini_project22\src\gui_app.py" (
    cd /d "%~dp0mini_project22"
) else if exist "%~dp0src\gui_app.py" (
    cd /d "%~dp0"
) else (
    cd /d "%~dp0.."
)

:: Locate Python virtualenv
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0.venv\Scripts\python.exe"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PY_BIN=%~dp0..\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY_BIN=.venv\Scripts\python.exe"
) else (
    set "PY_BIN=python"
)

"%PY_BIN%" src\gui_app.py

echo.
echo Application has closed.
pause

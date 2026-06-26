@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Unified Vision Console

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" call :create_environment
if not exist "%VENV_PYTHON%" goto :setup_failed

"%VENV_PYTHON%" -c "import torch, cv2, numpy, sklearn, pandas; from mediapipe import solutions" >nul 2>nul
if not !errorlevel! equ 0 call :install_dependencies

"%VENV_PYTHON%" -c "import torch, cv2, numpy, sklearn, pandas; from mediapipe import solutions" >nul 2>nul
if not !errorlevel! equ 0 goto :setup_failed

"%VENV_PYTHON%" unified_prediction\run_web.py
set "EXIT_CODE=!errorlevel!"
if not "!EXIT_CODE!"=="0" goto :failed
exit /b 0

:create_environment
echo.
echo [SETUP] Creating the Unified Python environment...
where py >nul 2>nul
if !errorlevel! equ 0 (
    py -3.11 -m venv .venv >nul 2>nul
    if not exist "%VENV_PYTHON%" py -3 -m venv .venv
) else (
    where python >nul 2>nul
    if not !errorlevel! equ 0 goto :no_python
    python -m venv .venv
)
exit /b

:install_dependencies
echo.
echo [SETUP] Installing prediction dependencies. The first run may take several minutes...
"%VENV_PYTHON%" -m pip install --upgrade pip
if not !errorlevel! equ 0 exit /b
"%VENV_PYTHON%" -m pip install -r requirements-unified.txt
exit /b

:no_python
echo.
echo Python 3 was not found.
echo Install Python 3 and enable "Add Python to PATH", then try again.
goto :pause_error

:setup_failed
echo.
echo Failed to prepare the prediction environment.
echo Check the network connection and the installation output above.
echo Retry with: .venv\Scripts\python -m pip install -r requirements-unified.txt
goto :pause_error

:failed
echo.
echo Unified Vision Console stopped with exit code !EXIT_CODE!.
echo Review the error above.

:pause_error
echo.
pause
exit /b 1

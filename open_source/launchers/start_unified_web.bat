@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Unified Vision Console

set "VENV_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PYTHON%" call :create_environment
if not exist "%VENV_PYTHON%" goto :setup_failed

call :check_python "%VENV_PYTHON%"
if not !errorlevel! equ 0 goto :bad_python

"%VENV_PYTHON%" -c "import torch, cv2, numpy, sklearn, pandas; from mediapipe import solutions" >nul 2>nul
if not !errorlevel! equ 0 call :install_dependencies

"%VENV_PYTHON%" -c "import torch, cv2, numpy, sklearn, pandas; from mediapipe import solutions" >nul 2>nul
if not !errorlevel! equ 0 goto :setup_failed

"%VENV_PYTHON%" ..\inference\unified_prediction\run_web.py
set "EXIT_CODE=!errorlevel!"
if not "!EXIT_CODE!"=="0" goto :failed
exit /b 0

:create_environment
echo.
echo [SETUP] Creating the Unified Python environment...
call :find_compatible_python
if not defined BASE_PYTHON goto :no_compatible_python
%BASE_PYTHON% -m venv .venv
exit /b

:find_compatible_python
set "BASE_PYTHON="
where py >nul 2>nul
if !errorlevel! equ 0 (
    py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
    if !errorlevel! equ 0 (
        set "BASE_PYTHON=py -3.11"
        exit /b 0
    )
    py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
    if !errorlevel! equ 0 (
        set "BASE_PYTHON=py -3.12"
        exit /b 0
    )
)
where python >nul 2>nul
if !errorlevel! equ 0 (
    python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)" >nul 2>nul
    if !errorlevel! equ 0 set "BASE_PYTHON=python"
)
exit /b 0

:check_python
%~1 -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)" >nul 2>nul
exit /b !errorlevel!

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

:no_compatible_python
echo.
echo Python 3.11 or 3.12 is required for this runtime.
echo The current Python is incompatible with NumPy/MediaPipe wheels and will try to build NumPy from source.
echo Install 64-bit Python 3.11 from https://www.python.org/downloads/release/python-3119/
echo Then delete .venv if it exists and run this script again.
goto :pause_error

:bad_python
echo.
"%VENV_PYTHON%" -c "import sys; print('Current venv Python:', sys.version)"
echo This venv uses an incompatible Python version.
echo Delete this folder and rerun after installing Python 3.11 or 3.12:
echo   %~dp0.venv
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

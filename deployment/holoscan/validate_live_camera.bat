@echo off
setlocal
cd /d "%~dp0\..\.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0validate_live_camera.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
  echo Validation failed with exit code %EXIT_CODE%.
) else (
  echo Validation completed successfully.
)
pause
exit /b %EXIT_CODE%

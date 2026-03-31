@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
  call ".venv\Scripts\activate.bat"
) else if exist "crawler\.venv\Scripts\activate.bat" (
  call "crawler\.venv\Scripts\activate.bat"
) else (
  echo [ERROR] No virtual environment found.
  exit /b 1
)

python incubator_scheduler.py --cycles 0 --queries-per-cycle 5 --cycle-minutes 180
if errorlevel 1 (
  echo [ERROR] Scheduler exited with an error.
  exit /b %errorlevel%
)

endlocal

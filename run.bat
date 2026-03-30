@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo ==============================================
echo WebCrawler launcher
echo Root: %ROOT%
echo ==============================================

where uv >nul 2>nul
if errorlevel 1 (
  echo [ERROR] uv is not installed or not on PATH.
  echo Install from: https://docs.astral.sh/uv/getting-started/installation/
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm is not installed or not on PATH.
  echo Install Node.js LTS from: https://nodejs.org/
  exit /b 1
)

if /I "%~1"=="--dry-run" goto :dryrun

echo [1/3] Syncing Python dependencies...
uv sync --project crawler
if errorlevel 1 (
  echo [ERROR] Failed to sync Python dependencies.
  exit /b 1
)

if not exist "%ROOT%frontend\node_modules" (
  echo [2/3] Installing frontend dependencies...
  pushd "%ROOT%frontend"
  call npm install
  if errorlevel 1 (
    popd
    echo [ERROR] npm install failed.
    exit /b 1
  )
  popd
) else (
  echo [2/3] Frontend dependencies already present.
)

echo [3/3] Starting services in new windows...
start "WebCrawler Backend" cmd /k "cd /d ""%ROOT%"" && uv run --project crawler uvicorn api:app --reload --port 8000"
start "WebCrawler Frontend" cmd /k "cd /d ""%ROOT%frontend"" && npm run dev"

echo.
echo Started:
echo - Backend:  http://127.0.0.1:8000
echo - Frontend: http://127.0.0.1:5173
echo.
echo Optional if needed for search discovery:
echo docker compose -f docker-compose.searxng.yml up -d
exit /b 0

:dryrun
echo Dry run commands:
echo uv sync --project crawler
echo cd /d "%ROOT%" ^&^& uv run --project crawler uvicorn api:app --reload --port 8000
echo cd /d "%ROOT%frontend" ^&^& npm run dev
echo docker compose -f docker-compose.searxng.yml up -d
exit /b 0

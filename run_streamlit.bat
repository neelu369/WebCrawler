@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

echo ==============================================
echo WebCrawler Streamlit launcher
echo Root: %ROOT%
echo ==============================================

where uv >nul 2>nul
if errorlevel 1 (
  echo [ERROR] uv is not installed or not on PATH.
  echo Install from: https://docs.astral.sh/uv/getting-started/installation/
  exit /b 1
)

echo [1/2] Syncing Python dependencies...
uv sync --project crawler
if errorlevel 1 (
  echo [ERROR] uv sync failed.
  exit /b 1
)

echo [2/2] Starting backend and Streamlit in new windows...
start "WebCrawler Backend" cmd /k "cd /d ""%ROOT%"" && uv run --project crawler uvicorn api:app --reload --port 8000"
start "WebCrawler Streamlit" cmd /k "cd /d ""%ROOT%"" && uv run --project crawler streamlit run streamlit_app.py --server.port 8501"

echo.
echo Started:
echo - API:       http://127.0.0.1:8000
echo - Streamlit: http://127.0.0.1:8501
echo.
echo Optional for web discovery:
echo docker compose -f docker-compose.searxng.yml up -d
exit /b 0

@echo off
cd /d "%~dp0"

echo Stopping existing process on port 8000...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting LOHA server at http://127.0.0.1:8000
cd LOHA
uvicorn loha.server:app --host 127.0.0.1 --port 8000

pause

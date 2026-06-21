@echo off
setlocal

echo ==============================================
echo      DeviceDNA Platform - Startup Sequence
echo ==============================================

:: Set root path automatically from script location
set ROOT=%~dp0
set ROOT=%ROOT:~0,-1%

echo Root directory: %ROOT%
echo.

:: Step 1 — Start all Docker services
echo [1/4] Starting Docker infrastructure (Postgres, Redis, InfluxDB, Kafka)...
docker-compose -f "%ROOT%\docker-compose.yml" up -d postgres redis influxdb zookeeper kafka
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker failed to start. Make sure Docker Desktop is running.
    pause
    exit /b 1
)
echo Waiting for services to become healthy...
timeout /t 15 /nobreak >nul

:: Step 2 — Run seeder (one-time demo data)
echo [2/4] Seeding demo data...
docker-compose -f "%ROOT%\docker-compose.yml" up seeder
timeout /t 3 /nobreak >nul

:: Step 3 — Start FastAPI backend
echo [3/4] Starting FastAPI Backend...
start "DeviceDNA - Backend" cmd /c "cd /d "%ROOT%\backend" && call venv\Scripts\activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo Waiting for backend to start...
timeout /t 8 /nobreak >nul

:: Step 4 — Start simulator
echo [4/4] Starting Fleet Simulator (50 devices)...
start "DeviceDNA - Simulator" cmd /c "cd /d "%ROOT%\backend" && call venv\Scripts\activate && python -m simulator.main"
timeout /t 2 /nobreak >nul

:: Step 5 — Start frontend
echo [5/5] Starting Next.js Frontend...
start "DeviceDNA - Frontend" cmd /c "cd /d "%ROOT%\frontend" && npm run dev"
timeout /t 2 /nobreak >nul

echo.
powershell -Command "Write-Host '============================================' -ForegroundColor Cyan"
powershell -Command "Write-Host '  DeviceDNA is starting up!' -ForegroundColor Green"
powershell -Command "Write-Host '  Frontend:  http://localhost:3000' -ForegroundColor Yellow"
powershell -Command "Write-Host '  Backend:   http://localhost:8000/docs' -ForegroundColor Yellow"
powershell -Command "Write-Host '  Wait ~30s for all services to be ready.' -ForegroundColor Gray"
powershell -Command "Write-Host '============================================' -ForegroundColor Cyan"
echo.

endlocal

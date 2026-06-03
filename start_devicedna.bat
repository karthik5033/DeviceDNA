@echo off
setlocal

echo ==============================================
echo      DeviceDNA Platform - Startup Sequence
echo ==============================================

echo [1/9] Starting Mosquitto MQTT Broker...
start "DeviceDNA - Mosquitto" /MIN cmd /c "mosquitto -c mosquitto.conf"
timeout /t 2 /nobreak >nul

echo [2/9] Starting Redis Server (via WSL)...
wsl redis-server --daemonize yes
timeout /t 2 /nobreak >nul

echo [3/9] Starting PostgreSQL Service...
net start postgresql-x64-15
timeout /t 3 /nobreak >nul

echo [4/9] Starting InfluxDB...
start "DeviceDNA - InfluxDB" /MIN cmd /c "influxd"
timeout /t 3 /nobreak >nul

echo [5/9] Starting Kafka ^& Zookeeper...
start "DeviceDNA - Zookeeper" /MIN cmd /c ".\kafka\bin\windows\zookeeper-server-start.bat .\kafka\config\zookeeper.properties"
timeout /t 3 /nobreak >nul
start "DeviceDNA - Kafka" /MIN cmd /c ".\kafka\bin\windows\kafka-server-start.bat .\kafka\config\server.properties"
timeout /t 5 /nobreak >nul

echo [6/9] Starting Hardware Gateway Bridge...
start "DeviceDNA - Hardware Gateway" cmd /c "cd backend && call venv\Scripts\activate && python hardware_gateway.py"

echo [7/9] Starting FastAPI Backend...
start "DeviceDNA - FastAPI" cmd /c "cd backend && call venv\Scripts\activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak >nul

echo [8/9] Starting Fleet Simulator...
start "DeviceDNA - Simulator" cmd /c "cd backend && call venv\Scripts\activate && python -m simulator.main"

echo [9/9] Starting React Frontend...
start "DeviceDNA - Frontend" cmd /c "cd frontend && npm run dev"

echo.
powershell -Command "Write-Host 'DeviceDNA READY - Open http://localhost:3000' -ForegroundColor Green"
echo ==============================================
endlocal

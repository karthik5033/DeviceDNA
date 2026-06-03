@echo off
setlocal

echo ==============================================
echo      DeviceDNA Platform - Shutdown Sequence
echo ==============================================

echo [1/8] Stopping React Frontend (Node.js)...
taskkill /F /IM node.exe >nul 2>&1

echo [2/8] Stopping Python Services (FastAPI, Simulator, Gateway)...
taskkill /F /IM python.exe >nul 2>&1

echo [3/8] Stopping Kafka ^& Zookeeper (Java)...
taskkill /F /IM java.exe >nul 2>&1

echo [4/8] Stopping InfluxDB...
taskkill /F /IM influxd.exe >nul 2>&1

echo [5/8] Stopping PostgreSQL...
net stop postgresql-x64-15 >nul 2>&1

echo [6/8] Stopping Redis Server (WSL)...
wsl pkill redis-server >nul 2>&1

echo [7/8] Stopping Mosquitto MQTT...
taskkill /F /IM mosquitto.exe >nul 2>&1

echo [8/8] Closing stray terminal windows...
taskkill /FI "WINDOWTITLE eq DeviceDNA*" >nul 2>&1

echo.
powershell -Command "Write-Host 'DeviceDNA has been successfully shut down.' -ForegroundColor Green"
echo ==============================================
endlocal

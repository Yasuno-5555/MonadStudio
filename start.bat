@echo off
title Monad Studio Launcher
echo ========================================
echo   Monad Studio - Starting all servers
echo ========================================
echo.

:: Get the script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

:: Start Python Orchestrator in background
echo [1/3] Starting Python Orchestrator...
start "Monad Orchestrator" cmd /k "cd orchestrator && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

:: Wait for orchestrator to start
timeout /t 3 /nobreak > nul

:: Start Frontend dev server in background
echo [2/3] Starting Frontend (Vite)...
start "Monad Frontend" cmd /k "cd frontend && npm run dev"

:: Wait for frontend to start
echo [3/3] Waiting for servers to be ready...
timeout /t 5 /nobreak > nul

:: Open browser
echo Opening browser...
start http://localhost:5173

echo.
echo ========================================
echo   Monad Studio is ready!
echo ========================================
echo.
echo   Frontend:    http://localhost:5173
echo   Orchestrator: http://localhost:8000
echo.
echo   Close this window to stop all servers.
echo ========================================
pause

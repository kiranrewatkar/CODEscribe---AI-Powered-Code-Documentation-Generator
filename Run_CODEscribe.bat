
@echo off
cd /d D:\MCA\CODEscribe

start "CODEscribe Backend" cmd /k ".venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

start "CODEscribe Frontend" cmd /k "cd /d D:\MCA\CODEscribe\frontend && npm run dev"

timeout /t 3 /nobreak >nul

start "" http://localhost:5173
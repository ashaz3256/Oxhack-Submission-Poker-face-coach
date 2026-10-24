@echo off
rem Double-click to update and start Tilt: backend and frontend in their own windows, then the browser.
cd /d %~dp0
git pull
start "Tilt backend" cmd /k "cd backend && .venv\Scripts\activate && pip install -q -r requirements.txt && uvicorn tilt.api:app --port 8000"
start "Tilt frontend" cmd /k "cd frontend && npm install && npm run dev"
timeout /t 12 >nul
start chrome http://localhost:5173 || start http://localhost:5173

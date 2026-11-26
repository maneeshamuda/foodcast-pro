@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Run SETUP_WINDOWS.bat first. & pause & exit /b 1)
if not exist frontend\dist\index.html (echo Run SETUP_WINDOWS.bat first. & pause & exit /b 1)
start "FoodCast Pro" http://localhost:8000
.venv\Scripts\python.exe -m uvicorn api.main:app --port 8000
pause

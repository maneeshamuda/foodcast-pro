@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe py -3.12 -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
if errorlevel 1 goto fail
cd frontend
call npm install
if errorlevel 1 goto fail
call npm run build
if errorlevel 1 goto fail
cd ..
echo Setup complete. Run RUN_WINDOWS.bat.
pause
exit /b 0
:fail
echo Setup failed. Review the error above.
pause
exit /b 1

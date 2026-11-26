@echo off
cd /d "%~dp0"
cd frontend
call npm install
if errorlevel 1 goto fail
call npm run build
if errorlevel 1 goto fail
echo Frontend rebuilt. Restart RUN_WINDOWS.bat from the project folder.
pause
exit /b 0
:fail
echo Build failed. Review the error above.
pause
exit /b 1

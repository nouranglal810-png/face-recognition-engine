@echo off
echo ========================================
echo   Face Recognition - Starting...
echo ========================================
echo.

:: Project folder mein jao
cd /d "%~dp0"

:: Virtual environment activate karo (agar hai to)
if exist venv\Scripts\activate (
    call venv\Scripts\activate
)

:: Flask ko background mein start karo
start python app.py

:: 3 second wait karo taaki Flask ready ho jaye
timeout /t 3 >nul

:: Browser automatically kholo
start "" "http://127.0.0.1:5001"

echo.
echo Server chal raha hai!
echo Band karne ke liye ye window close karein.
pause
```
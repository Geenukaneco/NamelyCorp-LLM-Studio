@echo off
REM ============================================================
REM  LLM Training Studio - One-Click Launcher
REM  Place this in C:\dev\llm\studio\
REM ============================================================

setlocal enabledelayedexpansion
set ROOT=%~dp0
cd /d "%ROOT%"

echo.
echo ============================================================
echo  🚀 LLM Training Studio Launcher
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "..\\.venv\\Scripts\\python.exe" (
    echo [ERROR] Virtual environment not found at .venv
    echo.
    echo Please run setup first:
    echo   1. cd C:\dev\llm
    echo   2. setup_llm.bat
    echo.
    pause
    exit /b 1
)

REM Check if dependencies are installed
echo [1/3] Checking dependencies...
"..\\.venv\\Scripts\\python.exe" -c "import fastapi" 2>nul
if %errorlevel% neq 0 (
    echo [WARN] FastAPI not found. Installing dependencies...
    "..\\.venv\\Scripts\\python.exe" -m pip install -q fastapi uvicorn websockets psutil py3nvml pydantic python-multipart aiofiles
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies found
)

REM Create static directory if missing
echo [2/3] Setting up directories...
if not exist "static" mkdir static
echo [OK] Directories ready

REM Check if files exist
echo [3/3] Verifying files...
set MISSING=0
if not exist "app.py" (
    echo [ERROR] app.py not found
    set MISSING=1
)
if not exist "static\\index.html" (
    echo [ERROR] static\\index.html not found
    set MISSING=1
)
if not exist "static\\style.css" (
    echo [ERROR] static\\style.css not found
    set MISSING=1
)
if not exist "static\\app.js" (
    echo [ERROR] static\\app.js not found
    set MISSING=1
)

if %MISSING% equ 1 (
    echo.
    echo [ERROR] Some files are missing. Please ensure all files are in place:
    echo   - app.py
    echo   - static/index.html
    echo   - static/style.css
    echo   - static/app.js
    echo.
    pause
    exit /b 1
)
echo [OK] All files present

echo.
echo ============================================================
echo  🌐 Starting LLM Training Studio
echo ============================================================
echo.
echo  URL: http://localhost:7860
echo  Press Ctrl+C to stop
echo.
echo ============================================================
echo.

REM Start the server
"..\\.venv\\Scripts\\python.exe" app.py

endlocal

@echo off
REM ============================================================
REM  LLM Training Studio - Complete Setup Script
REM  Run this once to set up everything
REM  Place in C:\dev\llm\studio\
REM ============================================================

setlocal enabledelayedexpansion
set ROOT=%~dp0
cd /d "%ROOT%"

echo.
echo ============================================================
echo  🚀 LLM Training Studio - Complete Setup
echo ============================================================
echo.

REM Step 1: Check Python
echo [1/5] Checking Python installation...
where py >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python launcher not found
    echo Please install Python 3.11 from python.org
    pause
    exit /b 1
)

py -3.11 --version >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.11 not found
    echo Please install Python 3.11 from python.org
    pause
    exit /b 1
)
echo [OK] Python 3.11 found

REM Step 2: Check/Create venv
echo.
echo [2/5] Setting up virtual environment...
if not exist "..\\.venv" (
    echo Creating new virtual environment...
    cd ..
    py -3.11 -m venv .venv
    cd studio
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment exists
)

REM Step 3: Install dependencies
echo.
echo [3/5] Installing Python packages...
echo This may take a few minutes...
"..\\.venv\\Scripts\\python.exe" -m pip install -q --upgrade pip
"..\\.venv\\Scripts\\python.exe" -m pip install -q fastapi uvicorn[standard] websockets psutil py3nvml pydantic python-multipart aiofiles
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install packages
    pause
    exit /b 1
)
echo [OK] Python packages installed

REM Step 4: Create directories
echo.
echo [4/5] Creating project structure...
if not exist "static" mkdir static
if not exist "..\\docs" mkdir "..\\docs"
if not exist "..\\ft_out" mkdir "..\\ft_out"
echo [OK] Directories created

REM Step 5: Check files
echo.
echo [5/5] Verifying setup files...
set MISSING=0
set FILES_NEEDED=app.py static/index.html static/style.css static/app.js

for %%F in (%FILES_NEEDED%) do (
    if not exist "%%F" (
        echo [WARN] Missing: %%F
        set MISSING=1
    )
)

if %MISSING% equ 1 (
    echo.
    echo [INFO] Some files are missing. You need to create:
    echo   1. app.py - FastAPI backend
    echo   2. static/index.html - Main UI
    echo   3. static/style.css - Styling
    echo   4. static/app.js - Frontend logic
    echo.
    echo Copy the provided files to these locations.
)

echo.
echo ============================================================
echo  ✅ Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Copy all provided files to their locations
echo   2. Run: launch_studio.bat
echo   3. Open: http://localhost:7860
echo.
echo Project structure:
echo   C:\dev\llm\
echo   ├── studio\              (this folder)
echo   │   ├── app.py
echo   │   ├── static\
echo   │   │   ├── index.html
echo   │   │   ├── style.css
echo   │   │   └── app.js
echo   │   ├── launch_studio.bat
echo   │   └── setup_studio.bat (this file)
echo   ├── .venv\              (virtual environment)
echo   ├── docs\               (put documents here)
echo   └── ft_out\             (training outputs)
echo.
echo ============================================================
echo.

pause
endlocal

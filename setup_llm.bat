@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  Llama 3.2 Q&A Trainer – Windows one-shot setup
REM  - Creates .venv
REM  - Installs CUDA 12.1 PyTorch + all Python deps
REM  - Prints next steps
REM ============================================================

set ROOT=%~dp0
cd /d "%ROOT%"

echo.
echo [1/6] Creating venv (Python 3.11)...
where py >nul 2>nul
if %errorlevel% neq 0 (
  echo   ERROR: Python launcher "py" not found. Install Python 3.11 and retry.
  exit /b 1
)
py -3.11 -m venv .venv
if %errorlevel% neq 0 (
  echo   ERROR: Failed to create venv. Check Python install.
  exit /b 1
)

echo.
echo [2/6] Upgrading pip...
".\.venv\Scripts\python.exe" -m pip install --upgrade pip

echo.
echo [3/6] Installing PyTorch (CUDA 12.1)...
".\.venv\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

echo.
echo [4/6] Installing core libraries...
".\.venv\Scripts\python.exe" -m pip install ^
 transformers datasets peft accelerate safetensors sentencepiece huggingface_hub

echo.
echo [5/6] Installing document + OCR libraries...
".\.venv\Scripts\python.exe" -m pip install ^
 pymupdf pypdf python-docx pandas openpyxl tabulate ^
 pdf2image pytesseract pillow

echo.
echo [6/6] Creating docs folder (if missing)...
if not exist ".\docs" mkdir ".\docs"

echo.
echo ============================================================
echo  Setup complete! Activate your venv and login to HF:
echo.
echo    .\.venv\Scripts\Activate.ps1
echo    hf auth login
echo.
echo  Put files in: %ROOT%docs
echo  Then run:
echo    python .\make_dataset_from_docs.py
echo    python .\validate_qa.py --write-clean .\data_qa.cleaned.csv --report .\qa_report.md
echo    python .\train_ft.py --merge-full
echo.
echo  OCR note: Install Tesseract and add to PATH if you have scanned PDFs:
echo    https://github.com/UB-Mannheim/tesseract/wiki
echo ============================================================
echo.

endlocal

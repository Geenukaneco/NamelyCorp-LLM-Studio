# app.py - LLM Training Studio Main Application
"""
LLM Training Studio - One-Shot FastAPI Application
Run with: python app.py
Access at: http://localhost:7860
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import asyncio
import json
import os
import sys
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
import psutil

# GPU monitoring (optional - gracefully degrade if not available)
try:
    import py3nvml.py3nvml as nvml
    nvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False
    print("[WARN] NVIDIA GPU monitoring not available")

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent.parent  # C:\dev\llm
DOCS_DIR = BASE_DIR / "docs"
SCRIPTS_DIR = BASE_DIR  # Where your Python scripts are
OUTPUT_DIR = BASE_DIR / "ft_out"
MODELS_DIR = Path(r"C:\models")

# Global state for background tasks
active_tasks = {}
task_logs = {}

# =============================================================================
# Pydantic Models
# =============================================================================

class ProjectStats(BaseModel):
    docs_count: int
    docs_size_mb: float
    dataset_rows: int
    dataset_path: Optional[str]
    models_count: int
    gpu_name: Optional[str]
    vram_total_gb: Optional[float]
    vram_used_gb: Optional[float]
    vram_percent: Optional[float]

class DatasetBuildRequest(BaseModel):
    docs_dir: str = str(DOCS_DIR)
    output_csv: str = str(BASE_DIR / "data_qa.csv")
    model_id: str = str(MODELS_DIR / "Llama-3.2-3B-Instruct")
    chars_per_chunk: int = 1400
    max_tokens: int = 1024
    enable_ocr: bool = True
    ocr_lang: str = "eng"
    max_files: Optional[int] = None
    mix_patterns: bool = True

class ValidationRequest(BaseModel):
    csv_path: str = str(BASE_DIR / "data_qa.csv")
    model_id: str = str(MODELS_DIR / "Llama-3.2-3B-Instruct")
    max_tokens: int = 1024
    write_clean: bool = True
    clean_path: str = str(BASE_DIR / "data_qa.cleaned.csv")
    report_path: str = str(BASE_DIR / "qa_report.md")

class TrainingRequest(BaseModel):
    csv_path: str = str(BASE_DIR / "data_qa.cleaned.csv")
    model_id: str = str(MODELS_DIR / "Llama-3.2-3B-Instruct")
    output_dir: str = str(OUTPUT_DIR)
    run_name: str = "training_run"
    epochs: int = 2
    batch_size: int = 2
    grad_accum: int = 16
    learning_rate: float = 1e-5
    max_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    include_mlp: bool = True
    merge_full: bool = True
    val_split: float = 0.05

class GGUFConvertRequest(BaseModel):
    model_path: str
    output_path: str
    quant_type: str = "q4_k_m"

# =============================================================================
# System Utilities
# =============================================================================

def get_gpu_stats() -> Dict[str, Any]:
    """Get current GPU statistics"""
    if not GPU_AVAILABLE:
        return {
            "available": False,
            "name": "No GPU",
            "vram_total_gb": 0,
            "vram_used_gb": 0,
            "vram_percent": 0,
            "gpu_percent": 0
        }
    
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        name = nvml.nvmlDeviceGetName(handle)
        
        return {
            "available": True,
            "name": name.decode() if isinstance(name, bytes) else name,
            "vram_total_gb": round(info.total / 1024**3, 2),
            "vram_used_gb": round(info.used / 1024**3, 2),
            "vram_percent": round((info.used / info.total) * 100, 1),
            "gpu_percent": util.gpu
        }
    except Exception as e:
        print(f"[ERROR] GPU stats: {e}")
        return {"available": False, "error": str(e)}

def check_tesseract() -> Dict[str, Any]:
    """Check if Tesseract OCR is installed and accessible"""
    try:
        import shutil
        
        # First, try to find in PATH
        tesseract_path = shutil.which('tesseract')
        
        # If not in PATH, search common Windows installation locations
        if not tesseract_path:
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\devdesk\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
                r"C:\Tesseract-OCR\tesseract.exe",
            ]
            
            for path in common_paths:
                if Path(path).exists():
                    tesseract_path = path
                    break
        
        if tesseract_path:
            # Try to get version
            result = subprocess.run([tesseract_path, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                
                # Optionally configure pytesseract to use this path
                try:
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                except:
                    pass
                
                return {
                    "available": True,
                    "path": tesseract_path,
                    "version": version_line
                }
        
        return {"available": False, "reason": "Not found in PATH or common locations"}
    except Exception as e:
        return {"available": False, "reason": str(e)}

def get_system_status() -> Dict[str, Any]:
    """Get overall system status"""
    gpu = get_gpu_stats()
    tesseract = check_tesseract()
    
    return {
        "gpu": {
            "status": "ok" if gpu.get("available") else "error",
            "name": gpu.get("name", "No GPU")
        },
        "cuda": {
            "status": "ok" if gpu.get("available") else "error"
        },
        "tesseract": {
            "status": "ok" if tesseract.get("available") else "warning",
            "info": tesseract.get("version", tesseract.get("reason", "Unknown"))
        }
    }

def get_project_stats() -> ProjectStats:
    """Gather project statistics"""
    # Count docs
    docs_count = 0
    docs_size = 0
    if DOCS_DIR.exists():
        for f in DOCS_DIR.rglob("*"):
            if f.is_file():
                docs_count += 1
                docs_size += f.stat().st_size
    
    # Count dataset rows
    dataset_rows = 0
    dataset_path = None
    for candidate in [BASE_DIR / "data_qa.cleaned.csv", BASE_DIR / "data_qa.csv"]:
        if candidate.exists():
            dataset_path = str(candidate)
            try:
                with open(candidate, 'r', encoding='utf-8') as f:
                    dataset_rows = sum(1 for _ in f) - 1  # minus header
            except:
                pass
            break
    
    # Count models
    models_count = 0
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir() and any(d.glob("*.safetensors")) or any(d.glob("*.bin")):
                models_count += 1
    
    # GPU stats
    gpu = get_gpu_stats()
    
    return ProjectStats(
        docs_count=docs_count,
        docs_size_mb=round(docs_size / 1024**2, 2),
        dataset_rows=dataset_rows,
        dataset_path=dataset_path,
        models_count=models_count,
        gpu_name=gpu.get("name"),
        vram_total_gb=gpu.get("vram_total_gb"),
        vram_used_gb=gpu.get("vram_used_gb"),
        vram_percent=gpu.get("vram_percent")
    )

def scan_folder(path: str) -> List[Dict[str, Any]]:
    """Scan a folder for files"""
    files = []
    try:
        p = Path(path)
        if p.exists() and p.is_dir():
            for f in p.rglob("*"):
                if f.is_file():
                    files.append({
                        "name": f.name,
                        "path": str(f),
                        "size_mb": round(f.stat().st_size / 1024**2, 3),
                        "ext": f.suffix.lower()
                    })
    except Exception as e:
        print(f"[ERROR] Scanning {path}: {e}")
    return files

# =============================================================================
# Background Task Runners
# =============================================================================

def run_dataset_builder(task_id: str, request: DatasetBuildRequest):
    """Run dataset builder in background"""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "make_dataset_from_docs.py"),
        "--docs-dir", request.docs_dir,
        "--out-csv", request.output_csv,
        "--model-id", request.model_id,
        "--chars-per-chunk", str(request.chars_per_chunk),
        "--max-tokens", str(request.max_tokens),
        "--ocr-lang", request.ocr_lang,
    ]
    
    if not request.enable_ocr:
        cmd.append("--no-ocr")
    if request.max_files:
        cmd.extend(["--max-files", str(request.max_files)])
    if not request.mix_patterns:
        cmd.append("--no-bullets")
    
    active_tasks[task_id] = {"status": "running", "progress": 0, "message": "Starting..."}
    task_logs[task_id] = []
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            if line:
                task_logs[task_id].append(line.strip())
                # Parse progress from logs
                if "[INFO] Processed" in line:
                    try:
                        parts = line.split()
                        idx = parts.index("Processed")
                        progress_str = parts[idx + 1]  # e.g., "5/47"
                        current, total = map(int, progress_str.split('/'))
                        progress = int((current / total) * 100)
                        active_tasks[task_id]["progress"] = progress
                        active_tasks[task_id]["message"] = f"Processing {current}/{total} files"
                    except:
                        pass
        
        process.wait()
        
        if process.returncode == 0:
            active_tasks[task_id] = {"status": "completed", "progress": 100, "message": "Dataset built successfully"}
        else:
            active_tasks[task_id] = {"status": "failed", "progress": 0, "message": "Build failed"}
    
    except Exception as e:
        active_tasks[task_id] = {"status": "failed", "progress": 0, "message": str(e)}
        task_logs[task_id].append(f"ERROR: {e}")

def run_validation(task_id: str, request: ValidationRequest):
    """Run validation in background"""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "validate_qa.py"),
        "--csv", request.csv_path,
        "--model-id", request.model_id,
        "--max-tokens", str(request.max_tokens),
    ]
    
    if request.write_clean:
        cmd.extend(["--write-clean", request.clean_path])
    if request.report_path:
        cmd.extend(["--report", request.report_path])
    
    active_tasks[task_id] = {"status": "running", "progress": 50, "message": "Validating..."}
    task_logs[task_id] = []
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        task_logs[task_id] = result.stdout.split('\n') + result.stderr.split('\n')
        
        if result.returncode in [0, 3]:  # 0 or 3 (overlong warnings) are OK
            active_tasks[task_id] = {"status": "completed", "progress": 100, "message": "Validation complete"}
        else:
            active_tasks[task_id] = {"status": "failed", "progress": 0, "message": "Validation failed"}
    
    except Exception as e:
        active_tasks[task_id] = {"status": "failed", "progress": 0, "message": str(e)}
        task_logs[task_id].append(f"ERROR: {e}")

def run_training(task_id: str, request: TrainingRequest):
    """Run training in background"""
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_ft.py"),
        "--csv", request.csv_path,
        "--model-id", request.model_id,
        "--outdir", request.output_dir,
        "--max-len", str(request.max_len),
        "--epochs", str(request.epochs),
        "--batch-size", str(request.batch_size),
        "--grad-accum", str(request.grad_accum),
        "--lr", str(request.learning_rate),
        "--lora-r", str(request.lora_r),
        "--lora-alpha", str(request.lora_alpha),
        "--lora-dropout", str(request.lora_dropout),
        "--val-size", str(request.val_split),
    ]
    
    if not request.include_mlp:
        cmd.append("--no-mlp")
    if request.merge_full:
        cmd.append("--merge-full")
    
    active_tasks[task_id] = {"status": "running", "progress": 0, "message": "Starting training..."}
    task_logs[task_id] = []
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            if line:
                task_logs[task_id].append(line.strip())
                # Parse training progress (you'd need to add progress logging to train_ft.py)
                if "Training" in line or "Step" in line:
                    active_tasks[task_id]["message"] = line.strip()[:100]
        
        process.wait()
        
        if process.returncode == 0:
            active_tasks[task_id] = {"status": "completed", "progress": 100, "message": "Training complete"}
        else:
            active_tasks[task_id] = {"status": "failed", "progress": 0, "message": "Training failed"}
    
    except Exception as e:
        active_tasks[task_id] = {"status": "failed", "progress": 0, "message": str(e)}
        task_logs[task_id].append(f"ERROR: {e}")

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="LLM Training Studio", version="1.0.0")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Serve main UI"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>LLM Training Studio</h1><p>Create static/index.html</p>")

@app.get("/api/stats")
async def get_stats():
    """Get project statistics"""
    return get_project_stats()

@app.get("/api/gpu")
async def get_gpu():
    """Get GPU stats"""
    return get_gpu_stats()

@app.get("/api/system")
async def get_system():
    """Get system status (GPU, CUDA, Tesseract)"""
    return get_system_status()

@app.get("/api/files")
async def list_files(path: str = None):
    """List files in a directory"""
    if not path or path.strip() == "":
        path = str(DOCS_DIR)
    return {"files": scan_folder(path), "path": path}

@app.post("/api/dataset/build")
async def build_dataset(request: DatasetBuildRequest, background_tasks: BackgroundTasks):
    """Start dataset building"""
    task_id = f"dataset_{int(time.time())}"
    background_tasks.add_task(run_dataset_builder, task_id, request)
    return {"task_id": task_id, "message": "Dataset building started"}

@app.post("/api/validate")
async def validate_dataset(request: ValidationRequest, background_tasks: BackgroundTasks):
    """Start validation"""
    task_id = f"validate_{int(time.time())}"
    background_tasks.add_task(run_validation, task_id, request)
    return {"task_id": task_id, "message": "Validation started"}

@app.post("/api/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start training"""
    task_id = f"train_{int(time.time())}"
    background_tasks.add_task(run_training, task_id, request)
    return {"task_id": task_id, "message": "Training started"}

@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    status = active_tasks.get(task_id, {"status": "not_found", "progress": 0, "message": "Task not found"})
    logs = task_logs.get(task_id, [])[-50:]  # Last 50 lines
    return {"status": status, "logs": logs}

@app.websocket("/ws/task/{task_id}")
async def task_websocket(websocket: WebSocket, task_id: str):
    """WebSocket for real-time task updates"""
    await websocket.accept()
    
    try:
        while True:
            status = active_tasks.get(task_id, {"status": "not_found"})
            gpu = get_gpu_stats()
            logs = task_logs.get(task_id, [])[-10:]  # Last 10 lines
            
            await websocket.send_json({
                "task_id": task_id,
                "status": status,
                "gpu": gpu,
                "logs": logs,
                "timestamp": datetime.now().isoformat()
            })
            
            if status.get("status") in ["completed", "failed"]:
                break
            
            await asyncio.sleep(2)
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for task {task_id}")

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 LLM Training Studio")
    print("=" * 60)
    print(f"📁 Project: {BASE_DIR}")
    print(f"📂 Docs:    {DOCS_DIR}")
    print(f"💾 Output:  {OUTPUT_DIR}")
    print(f"🖥️  GPU:     {get_gpu_stats().get('name', 'No GPU')}")
    print("=" * 60)
    print("\n🌐 Starting server at http://localhost:7860")
    print("   Press Ctrl+C to stop\n")
    
    # Create directories
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info"
    )

# Changelog

All notable changes to NamelyCorp LLM Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-02-04

### 🎉 Initial Release

First stable release of NamelyCorp LLM Studio—a local-first LoRA fine-tuning toolkit with web UI.

### Added

#### Core Functionality
- **Environment Setup** (`setup_llm.bat`)
  - One-shot Python 3.11 virtual environment creation
  - CUDA 12.1 PyTorch installation
  - Core ML dependencies (transformers, peft, accelerate)
  - Document processing libraries (pymupdf, python-docx, pandas, openpyxl)
  - OCR support (pytesseract, pdf2image)

#### Dataset Pipeline
- **Dataset Builder** (`make_dataset_from_docs.py`)
  - Multi-format document support: PDF, DOCX, TXT, CSV, XLSX
  - Intelligent document chunking with configurable size
  - Optional OCR for scanned PDFs using Tesseract
  - Tokenization-aware chunking
  - Q&A pair generation with configurable patterns
  - CSV output with `question` and `answer` columns

- **Dataset Validation** (`validate_qa.py`)
  - Header validation (required columns check)
  - Duplicate detection and removal
  - Token count validation against model limits
  - Low-quality entry detection (length thresholds)
  - Markdown report generation with statistics
  - Optional cleaned CSV output

#### Training System
- **LoRA Fine-Tuning** (`train_ft.py`)
  - Low-Rank Adaptation (LoRA) implementation
  - Masked loss on assistant tokens only (proper chat template handling)
  - Configurable LoRA parameters (rank, alpha, dropout)
  - Optional MLP layer targeting for content learning
  - Training/validation split with perplexity metrics
  - Automatic precision selection (bf16 for Ampere+, fp16 fallback)
  - Gradient checkpointing for memory efficiency
  - Resume from checkpoint support
  - LoRA adapter export
  - Optional full model merging for standalone deployment

- **Inference Testing** (`test_inference.py`)
  - Smoke-test inference on trained models
  - Quick quality verification
  - Support for both adapter and merged model testing

#### Studio Web UI
- **FastAPI Backend** (`studio/app.py`)
  - RESTful API for all pipeline operations
  - Real-time GPU monitoring (NVIDIA NVML)
  - System status checks (CUDA, Tesseract, Python environment)
  - Background task management with progress tracking
  - WebSocket support for streaming logs
  - Project statistics dashboard

- **Setup Scripts**
  - `studio/setup_studio.bat` - UI dependencies installation
  - `studio/launch_studio.bat` - One-click server launcher with dependency checks

- **Frontend UI** (`studio/static/`)
  - **Dashboard Tab**
    - Project statistics (documents, datasets, trained models)
    - Quick action buttons
    - Real-time system status (GPU, CUDA, Tesseract, Python)
    - GPU VRAM monitoring with usage percentages
  
  - **Dataset Builder Tab**
    - Document directory scanning
    - Configuration UI for all build parameters
    - OCR language selection
    - Max files limit
    - Pattern mixing options
    - Real-time build progress
  
  - **Validation Tab**
    - Input CSV selection
    - Model selection for tokenization
    - Max tokens configuration
    - Cleaned CSV output option
    - Validation report display
  
  - **Training Tab**
    - Training preset selection (Quick Test, Balanced, High Quality)
    - Comprehensive parameter configuration
      - Epochs, batch size, gradient accumulation
      - Learning rate, max sequence length
      - LoRA rank, alpha, dropout
      - MLP layer inclusion toggle
      - Validation split ratio
    - Model merge option
    - Real-time training progress
  
  - **Export & Test Tab**
    - GGUF conversion UI (optional)
    - Quantization type selection
    - Model testing interface

- **UI Features**
  - Dark mode toggle
  - Responsive design (desktop and mobile)
  - Real-time GPU metrics in header
  - WebSocket-based live log streaming
  - Progress indicators for long-running tasks

#### Documentation
- Comprehensive README with:
  - Feature overview and screenshots
  - Detailed prerequisites
  - Step-by-step quick start guide
  - Complete UI tab reference
  - Command-line usage documentation
  - Troubleshooting section
  - LoRA vs. RAG explanation
  - LAN access guidance
  - Security best practices

- Security Policy (SECURITY.md):
  - Threat model definition
  - Vulnerability reporting process
  - LAN exposure guidance
  - Safe-use recommendations
  - In-scope and out-of-scope issues

- MIT License (LICENSE)
  - Open-source under MIT
  - Copyright 2026 NamelyCorp

#### Repository Infrastructure
- `.gitignore` - Comprehensive ignore rules:
  - Python artifacts (`__pycache__`, `.pyc`)
  - Virtual environments (`.venv`, `venv`)
  - Training outputs (`ft_out/`, `outputs/`, `checkpoints/`)
  - Generated data (`data_qa*.csv`, `qa_report*.md`)
  - Hugging Face caches
  - Environment files (`.env`)
  - OS-specific files

- `requirements.txt` - Full dependency list
  - Core ML: torch, transformers, peft, accelerate
  - Document processing: pymupdf, python-docx, pandas
  - OCR: pytesseract, pdf2image
  - Web UI: fastapi, uvicorn
  - GPU monitoring: py3nvml

- Example files:
  - `data_qa (Example).xlsx` - Sample QA report format

### Features

#### Local-First Design
- No telemetry or data collection
- All processing happens on user's machine
- Documents never leave local storage
- Complete offline operation (except model downloads)

#### LAN Access (Intentional)
- Server can be accessed from local network devices
- Useful for monitoring training from mobile/other computers
- Explicit security guidance in documentation
- Default `localhost`-only binding with opt-in LAN access

#### Flexible Deployment
- Works with local or Hugging Face models
- CPU-only operation supported (slow but functional)
- NVIDIA GPU acceleration recommended
- CUDA 12.1 optimization

#### Training Efficiency
- LoRA reduces trainable parameters to ~0.1-1% of full model
- Gradient accumulation for large effective batch sizes
- Mixed precision training (bf16/fp16)
- Gradient checkpointing for reduced memory usage
- Validation split with perplexity tracking

#### Document Processing
- Multi-format support with format-specific parsers
- Intelligent text extraction
- OCR fallback for scanned PDFs
- Configurable chunking strategies
- Token-aware splitting

### Known Limitations

- Windows-first design (Linux/macOS may require modifications)
- Requires Python 3.11 specifically
- CUDA 12.1 PyTorch (may need adjustment for different CUDA versions)
- No built-in authentication in web UI
- No multi-GPU training support in v1.0.0
- No distributed training
- Dataset builder uses simple Q&A pattern generation (may need manual curation)
- OCR requires separate Tesseract installation
- GGUF conversion not yet integrated in UI (command-line only)
- No real-time training metrics visualization beyond perplexity

### Technical Details

- **Python Version:** 3.11
- **PyTorch:** CUDA 12.1 build
- **Base Model Support:** Llama 3.2 (tested), compatible with other Llama-family models
- **Server:** FastAPI + uvicorn
- **Frontend:** Vanilla JavaScript (no framework dependencies)
- **Default Port:** 7860
- **Minimum RAM:** 8GB (16GB+ recommended)
- **Minimum VRAM:** 4GB (8GB+ recommended for 3B models)

### Dependencies

See `requirements.txt` for complete list. Key dependencies:
- torch >= 2.0
- transformers >= 4.35
- peft >= 0.6
- fastapi >= 0.104
- uvicorn >= 0.24

### Installation

```bash
git clone https://github.com/NamelyCorp/NamelyCorp-LLM-Studio.git
cd NamelyCorp-LLM-Studio
setup_llm.bat
cd studio
setup_studio.bat
launch_studio.bat
```

Access at http://localhost:7860

---

## Future Roadmap (Not Yet Implemented)

Items we're considering for future releases:

- Multi-GPU training support
- Distributed training across machines
- Built-in authentication system
- More sophisticated dataset generation (GPT-4/Claude integration)
- Real-time training metrics dashboard
- Model comparison tools
- Checkpoint management UI
- Experiment tracking integration (W&B, MLflow)
- Linux/macOS native support
- Docker/container deployment option
- GGUF conversion in UI
- Ollama integration for testing
- RAG pipeline integration
- Dataset augmentation tools

---

## [Unreleased]

No unreleased changes yet.

---

**Note:** Version numbers follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for added functionality in a backwards compatible manner  
- **PATCH** version for backwards compatible bug fixes

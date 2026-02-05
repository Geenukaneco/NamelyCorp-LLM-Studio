# NamelyCorp LLM Studio v1.0.0 🎉

**Local-first LoRA fine-tuning Studio with web UI**

We're excited to announce the first stable release of NamelyCorp LLM Studio—a complete end-to-end system for fine-tuning language models on your own documents, running entirely on your local machine.

---

## ✨ What's New

### Complete Fine-Tuning Pipeline

NamelyCorp LLM Studio provides a complete workflow from documents to deployable models:

- 📄 **Dataset Builder** — Generate Q&A training pairs from PDF, DOCX, TXT, CSV, XLSX with optional OCR
- ✅ **Validation** — Clean and validate datasets with comprehensive quality checks
- ⚡ **LoRA Training** — Efficient fine-tuning with masked loss and configurable parameters
- 📦 **Export** — Save adapters, merge models, optional GGUF conversion
- 🧪 **Testing** — Smoke-test inference before deployment

### Intuitive Web UI

![Dashboard](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/blob/main/assets/dashboard.png?raw=true)

- Real-time GPU monitoring with VRAM usage
- Step-by-step guided workflow
- Training presets for different use cases
- Live progress tracking for long-running operations
- Mobile-friendly responsive design

### Local-First & Privacy-Focused

- **No telemetry** - Your data never leaves your machine
- **Offline operation** - Works without internet (except model downloads)
- **Full control** - You own your data, models, and infrastructure

### Production-Ready Features

- Masked loss on assistant tokens (proper chat template handling)
- Configurable LoRA parameters (rank, alpha, dropout, MLP targeting)
- Training/validation split with perplexity tracking
- Mixed precision (bf16/fp16) and gradient checkpointing
- Resume from checkpoint support
- Optional full model merging for standalone deployment

---

## 🚀 Quick Start

### Prerequisites

- Windows 11 (or Windows 10)
- Python 3.11
- NVIDIA GPU with CUDA 12.1 drivers (recommended but optional)
- 16GB+ RAM, 50GB+ disk space

### Installation

```bash
# Clone the repository
git clone https://github.com/NamelyCorp/NamelyCorp-LLM-Studio.git
cd NamelyCorp-LLM-Studio

# Set up Python environment (installs CUDA PyTorch + dependencies)
setup_llm.bat

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Set up and launch Studio UI
cd studio
setup_studio.bat
launch_studio.bat
```

Access the Studio at **http://localhost:7860**

**Full documentation:** [README.md](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/blob/main/README.md)

---

## 📋 What's Included

### Core Scripts
- `setup_llm.bat` - Environment setup with CUDA PyTorch
- `make_dataset_from_docs.py` - Dataset generation from documents
- `validate_qa.py` - Dataset validation and cleaning
- `train_ft.py` - LoRA fine-tuning script
- `test_inference.py` - Model inference testing

### Studio Web UI
- `studio/` - FastAPI backend + modern web interface
- Real-time GPU monitoring and system status
- Guided workflow through all pipeline stages
- Background task management with progress tracking

### Documentation
- Comprehensive README with step-by-step guides
- Security policy with LAN access guidance
- Contribution guidelines
- Example files and troubleshooting

---

## 🔐 Security Note

**LAN Access is an Intentional Feature:** The Studio can be accessed from other devices on your local network for monitoring and collaboration. By default, it runs on `localhost` only. See [SECURITY.md](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/blob/main/SECURITY.md) for safe LAN usage guidelines.

**Do NOT expose to the public internet** without proper security controls (authentication, TLS, firewall rules, network isolation).

---

## ❌ What's NOT Included

This repository **intentionally does not include:**

- Base model weights (you must download separately, e.g., Llama 3.2)
- Training data (you provide your own documents)
- Pre-trained adapters or fine-tuned models

**Why?** Base models are typically 5-10GB+ with their own licenses. Training data is domain-specific and private. This keeps the repository lightweight and license-compliant.

---

## 🎯 Use Cases

Perfect for:

- **Enterprise teams** training models on internal documentation
- **Developers** fine-tuning on domain-specific content
- **Researchers** conducting reproducible fine-tuning experiments
- **ML engineers** who need full control over the training pipeline

---

## 📊 System Requirements

### Minimum (CPU-only, slow)
- Windows 10/11
- Python 3.11
- 8GB RAM
- 30GB disk space

### Recommended (GPU-accelerated)
- Windows 11
- Python 3.11
- NVIDIA RTX 3060 or higher (8GB+ VRAM)
- 16GB+ RAM
- 50GB+ disk space
- CUDA 12.1 drivers

### Optional
- Tesseract OCR (for scanned PDF processing)

---

## 🐛 Known Limitations

- **Windows-first design** - Linux/macOS may require modifications
- **No built-in authentication** - Add your own if exposing beyond localhost
- **Single-GPU only** - No multi-GPU or distributed training in v1.0.0
- **Python 3.11 required** - Other versions not tested
- **Dataset generation uses simple patterns** - May need manual curation for best results
- **GGUF conversion** - Currently command-line only (not in UI)

See [CHANGELOG.md](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/blob/main/CHANGELOG.md) for complete details.

---

## 🔧 Troubleshooting

### CUDA not detected
```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of memory (OOM)
- Reduce batch size to 1
- Increase gradient accumulation to 32+
- Reduce max sequence length to 512

### Tesseract not found
- Download: https://github.com/UB-Mannheim/tesseract/wiki
- Add to PATH: `C:\Program Files\Tesseract-OCR`

**Full troubleshooting guide:** [README.md#troubleshooting](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/blob/main/README.md#troubleshooting)

---

## 📚 Documentation

- **[README.md](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/blob/main/README.md)** - Complete user guide
- **[SECURITY.md](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/blob/main/SECURITY.md)** - Security policy and best practices
- **[CONTRIBUTING.md](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/blob/main/CONTRIBUTING.md)** - Contribution guidelines
- **[CHANGELOG.md](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/blob/main/CHANGELOG.md)** - Detailed changelog

---

## 🤝 Contributing

Contributions are welcome! Areas we'd especially appreciate help:

- Multi-GPU training support
- Linux/macOS compatibility
- Additional document formats (EPUB, Markdown)
- Automated testing
- UI/UX improvements

See [CONTRIBUTING.md](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/blob/main/CONTRIBUTING.md) for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/blob/main/LICENSE) for details.

**Important:** Base models you use (e.g., Llama 3.2) are subject to their own licenses. You are responsible for compliance.

---

## 🙏 Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/docs/transformers/) - Hugging Face library
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - Optical character recognition

---

## 💬 Community

- **Report bugs:** [GitHub Issues](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/issues)
- **Feature requests:** [GitHub Issues](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/issues)
- **Discussions:** [GitHub Discussions](https://github.com/NamelyCorp/NamelyCorp-LLM-Studio/discussions)

---

## ⚠️ Responsible Use

- Only train on documents you have the right to use
- Do not include personal, confidential, or regulated data without proper controls
- Respect base model licenses
- Review exports before sharing—models can encode training data patterns
- Do not upload trained artifacts publicly unless your dataset is safe to disclose

---

**Thank you for using NamelyCorp LLM Studio!**

We're excited to see what you build with it. If you find this project useful, please ⭐ star the repository and share it with others.

---

<div align="center">
  <strong>NamelyCorp LLM Studio v1.0.0</strong>
  <br/>
  <em>© 2026 NamelyCorp. All rights reserved.</em>
</div>

# Contributing to NamelyCorp LLM Studio

Thank you for your interest in contributing to NamelyCorp LLM Studio! This document provides guidelines and information for contributors.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

---

## Code of Conduct

This project follows a simple code of conduct:

- **Be respectful** - Treat all contributors and users with respect
- **Be constructive** - Provide helpful feedback and suggestions
- **Be collaborative** - Work together to improve the project
- **Be inclusive** - Welcome contributors of all backgrounds and skill levels

Harassment, discrimination, or disrespectful behavior will not be tolerated.

---

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please create an issue with:

- **Clear title** - Descriptive summary of the problem
- **Steps to reproduce** - How to trigger the bug
- **Expected behavior** - What should happen
- **Actual behavior** - What actually happens
- **Environment** - OS, Python version, GPU/driver details
- **Logs/screenshots** - Any relevant error messages (with secrets removed)

**Security vulnerabilities** should be reported privately—see [SECURITY.md](SECURITY.md).

### Suggesting Features

We welcome feature suggestions! Please create an issue with:

- **Use case** - Why is this feature needed?
- **Proposed solution** - How would it work?
- **Alternatives considered** - Other approaches you've thought about
- **Impact** - Who would benefit from this feature?

### Improving Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples or tutorials
- Improve installation instructions
- Translate documentation (if you want to start a translation)
- Add screenshots or diagrams

### Contributing Code

We welcome code contributions! Areas where we'd especially appreciate help:

- **Document format support** - Add support for new file types
- **Training features** - Multi-GPU, distributed training, new optimizers
- **UI improvements** - Better visualization, mobile support, accessibility
- **Testing** - Add unit tests, integration tests, or end-to-end tests
- **Performance** - Optimize slow operations
- **Bug fixes** - Fix reported issues

---

## Development Setup

### Prerequisites

- Windows 11 (or Windows 10)
- Python 3.11
- Git
- NVIDIA GPU with CUDA drivers (optional but recommended)

### Setup Steps

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/NamelyCorp-LLM-Studio.git
   cd NamelyCorp-LLM-Studio
   ```

3. **Set up environment:**
   ```bash
   setup_llm.bat
   .\.venv\Scripts\Activate.ps1
   ```

4. **Set up Studio (if working on UI):**
   ```bash
   cd studio
   setup_studio.bat
   ```

5. **Create a branch for your changes:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Running the Studio

```bash
cd studio
launch_studio.bat
```

Access at http://localhost:7860

---

## Pull Request Process

### Before Submitting

1. **Test your changes** - Make sure everything works
2. **Update documentation** - Update README, docstrings, comments as needed
3. **Check code style** - Follow our coding standards (see below)
4. **Add changelog entry** - Add a note to CHANGELOG.md under "Unreleased"
5. **Commit with clear messages** - See commit message guidelines below

### Submitting Your PR

1. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create pull request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to related issues (e.g., "Fixes #123")
   - Screenshots (if UI changes)
   - Testing performed

3. **Address review feedback** - Respond to reviewer comments and make requested changes

### Commit Message Guidelines

Good commit messages help us understand changes:

- **Format:** `type: brief description`
- **Types:**
  - `feat:` - New feature
  - `fix:` - Bug fix
  - `docs:` - Documentation changes
  - `style:` - Code style/formatting (no functional changes)
  - `refactor:` - Code refactoring
  - `perf:` - Performance improvement
  - `test:` - Adding or updating tests
  - `chore:` - Maintenance tasks

**Examples:**
```
feat: add support for EPUB file format
fix: correct token counting in validation
docs: update troubleshooting section with OOM solutions
refactor: extract dataset builder to separate module
```

---

## Coding Standards

### Python Code

- **Style:** Follow [PEP 8](https://pep8.org/)
- **Line length:** 120 characters max (flexible for readability)
- **Imports:** Group standard library, third-party, local imports
- **Type hints:** Use type hints for function signatures when practical
- **Docstrings:** Use for all public functions, classes, and modules

**Example:**
```python
def build_messages(question: str, answer: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Build chat messages for Llama chat template.
    
    Args:
        question: User question text
        answer: Assistant answer text (optional)
    
    Returns:
        List of message dictionaries with role and content
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    # ... implementation
    return msgs
```

### JavaScript Code

- **Style:** Modern ES6+ syntax
- **Formatting:** 2-space indentation
- **Naming:** camelCase for variables/functions, PascalCase for classes
- **Comments:** Explain complex logic

### Batch Scripts

- **Comments:** Use REM for documentation
- **Error handling:** Check %errorlevel% after critical operations
- **User feedback:** Echo clear messages about what's happening

---

## Testing Guidelines

### Manual Testing

Before submitting, test:

1. **Fresh install** - Test on a clean environment if possible
2. **Core workflows** - Dataset build → validation → training → inference
3. **Error handling** - Try invalid inputs, missing files, etc.
4. **UI functionality** - Test all tabs and buttons
5. **Different configurations** - Various model sizes, parameter combinations

### Automated Testing

We currently don't have extensive automated tests, but contributions are welcome:

- **Unit tests** - Test individual functions
- **Integration tests** - Test component interactions
- **End-to-end tests** - Test complete workflows

Place tests in a `tests/` directory and use `pytest`.

---

## Documentation

### Code Documentation

- **Docstrings** - All public functions need docstrings
- **Inline comments** - Explain non-obvious logic
- **Type hints** - Help users understand function signatures

### User Documentation

- **README.md** - Primary user-facing documentation
- **Inline help** - UI tooltips, help text where appropriate
- **Examples** - Working code examples are very helpful
- **Troubleshooting** - Add solutions to common problems

### Keeping Docs Updated

When making changes:

- Update README if behavior changes
- Update docstrings if function signatures change
- Add troubleshooting entries for new error conditions
- Update screenshots if UI changes

---

## Community

### Getting Help

- **GitHub Issues** - Ask questions, report bugs
- **GitHub Discussions** - General discussion, show your work, ask for advice
- **Pull Request Comments** - Discuss specific code changes

### Staying Updated

- **Watch the repository** - Get notified of new issues and PRs
- **Star the repository** - Show your support
- **Check CHANGELOG** - See what's new in each release

### Recognition

Contributors will be:

- Listed in release notes
- Credited in CHANGELOG.md
- Acknowledged in README (for significant contributions)

---

## Development Roadmap

Current priorities for contributions:

**High Priority:**
- Multi-GPU training support
- Linux/macOS compatibility
- More document formats (EPUB, Markdown)
- Automated testing framework

**Medium Priority:**
- Authentication system for web UI
- Experiment tracking integration
- GGUF conversion in UI
- More training presets

**Low Priority (Nice to Have):**
- Docker/container deployment
- Additional UI themes
- Internationalization (i18n)
- Mobile app

---

## License

By contributing to NamelyCorp LLM Studio, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the `question` label
- Start a discussion on GitHub Discussions
- Reach out to maintainers

Thank you for contributing to NamelyCorp LLM Studio! 🎉

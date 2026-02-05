# GitHub Release Checklist for v1.0.0

Quick reference for creating the GitHub release.

---

## Pre-Release Checklist

- [ ] All documentation files reviewed and approved
- [ ] Screenshots in `assets/` directory committed to repository
- [ ] All code tested and working
- [ ] Version numbers updated in relevant files (if any)
- [ ] No sensitive data in repository

---

## Git Commands

```bash
# Ensure you're on the main branch
git checkout main

# Stage all the new/updated files
git add .

# Commit with clear message
git commit -m "Release v1.0.0 - Complete documentation and release package"

# Push to GitHub
git push origin main

# Create and push the version tag
git tag -a v1.0.0 -m "NamelyCorp LLM Studio v1.0.0 - Initial stable release"
git push origin v1.0.0
```

---

## Creating the GitHub Release

### Step 1: Navigate to Releases
1. Go to your GitHub repository
2. Click "Releases" (right sidebar)
3. Click "Draft a new release"

### Step 2: Choose Tag
- **Tag version:** `v1.0.0`
- **Target:** `main` branch

### Step 3: Release Title
```
NamelyCorp LLM Studio v1.0.0
```

### Step 4: Release Description
Copy the entire contents of `RELEASE_NOTES_v1.0.0.md` into the description field.

### Step 5: Attach Files (Optional)
You can attach:
- ZIP of source code (GitHub does this automatically)
- Individual screenshot files from `assets/` if you want them separately downloadable

### Step 6: Release Options
- [ ] Check "Set as the latest release"
- [ ] Leave "Set as a pre-release" **unchecked** (this is stable)
- [ ] Leave "Create a discussion for this release" checked (optional, but recommended)

### Step 7: Publish
Click **"Publish release"**

---

## Post-Release Actions

### Update Repository Settings
1. Go to Settings → General → Features
2. Ensure "Releases" is enabled
3. Update repository description to match README tagline:
   ```
   Local-first LoRA fine-tuning Studio with web UI - Windows-first, no Docker required
   ```

### Add Topics/Tags
Go to repository main page → "About" section → Add topics:
- `machine-learning`
- `fine-tuning`
- `lora`
- `llm`
- `pytorch`
- `transformers`
- `huggingface`
- `local-first`
- `fastapi`
- `windows`

### Create Announcement (Optional)
Consider announcing on:
- GitHub Discussions (create a "Release Announcement" post)
- Twitter/X
- Reddit (r/MachineLearning, r/LocalLLaMA)
- Hugging Face Discord
- LinkedIn

---

## Release Notes Template for Social Media

**Short Version (Twitter/X):**
```
🎉 Introducing NamelyCorp LLM Studio v1.0.0!

Local-first LoRA fine-tuning with web UI
✅ Build datasets from your docs
✅ Clean & validate data
✅ Train with LoRA
✅ Export & test models

Windows-first, no Docker, runs entirely on your machine.

🔗 [link to GitHub]

#MachineLearning #LLM #LoRA #LocalFirst
```

**Medium Version (LinkedIn/Reddit):**
```
We're excited to release NamelyCorp LLM Studio v1.0.0 - a complete local-first LoRA fine-tuning toolkit with web UI!

**What it does:**
- Generate training datasets from PDF, DOCX, TXT, CSV, XLSX files
- Validate and clean your data with comprehensive quality checks
- Fine-tune language models using efficient LoRA training
- Export adapters and merged models
- Test inference before deployment

**Key features:**
- Complete web UI with real-time GPU monitoring
- No telemetry - your data never leaves your machine
- Runs entirely offline (except model downloads)
- Windows-first design with straightforward setup
- Optional LAN access for monitoring and collaboration

Perfect for teams training models on internal documentation, developers working with domain-specific content, and ML engineers who want full control over the fine-tuning pipeline.

Built with PyTorch, Transformers, PEFT, and FastAPI.

MIT Licensed. No model weights included - you supply your own base models.

Check it out: [link to GitHub]
```

---

## Common Questions to Prepare For

**Q: Why Windows-first?**
A: Optimized for the most common ML workstation setup. Linux/macOS support is on the roadmap - contributions welcome!

**Q: Why not include base models?**
A: Models are 5-10GB+ with their own licenses. Keeping the repo lightweight and letting users choose their models ensures license compliance.

**Q: Is this production-ready?**
A: Yes! v1.0.0 is a stable release suitable for production use. We're following semantic versioning.

**Q: What about multi-GPU training?**
A: Not in v1.0.0, but it's on the roadmap. Single-GPU training is well-optimized and sufficient for most use cases.

**Q: Is this better than RAG?**
A: Different use cases. LoRA embeds knowledge in weights (good for stable, authoritative content). RAG retrieves at runtime (good for large, changing knowledge bases). You can use both together!

---

## Monitoring After Release

Keep an eye on:
- **GitHub Issues** - Respond to bug reports and questions
- **GitHub Discussions** - Engage with the community
- **Stars/Forks** - Track adoption
- **Pull Requests** - Review and merge contributions

---

## Version Numbering for Future Releases

Following [Semantic Versioning](https://semver.org/):

- **MAJOR** (x.0.0) - Breaking changes (e.g., 2.0.0)
- **MINOR** (1.x.0) - New features, backwards compatible (e.g., 1.1.0)
- **PATCH** (1.0.x) - Bug fixes, backwards compatible (e.g., 1.0.1)

Examples:
- Add multi-GPU training → `1.1.0` (new feature)
- Fix validation bug → `1.0.1` (bug fix)
- Change API structure → `2.0.0` (breaking change)

---

**Ready to release! 🚀**

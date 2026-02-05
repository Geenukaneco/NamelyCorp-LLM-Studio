# Security Policy — NamelyCorp LLM Studio

NamelyCorp LLM Studio is a **local-first** fine-tuning and model export toolkit. It is designed to run on your machine and process your documents. Security is primarily about preventing accidental exposure of sensitive data, safe defaults for file handling and execution, and clear boundaries around what this project does (and does not) secure for you.

---

## Supported Versions

Security fixes are applied to the most recent release and the current `main` branch.

If you are using an older release, upgrade to the latest version before reporting an issue.

---

## Reporting a Vulnerability

If you believe you've found a security issue, please **do not open a public GitHub issue**.

Instead, report it privately:

- Use GitHub private vulnerability reporting (Security tab → "Report a vulnerability"), **or**
- Email the maintainer (use the contact address listed in the repository / GitHub profile)

Please include:

- A clear description of the issue and impact
- Steps to reproduce (proof-of-concept if possible)
- Affected OS (Windows/Linux), Python version, GPU/driver details if relevant
- Logs or screenshots **with secrets removed**

We aim to acknowledge reports quickly and provide a fix or mitigation as soon as practical.

---

## LAN Exposure (Intentional Feature)

**Important:** The Studio UI is designed to be accessible on your local area network (LAN). This is an **intentional feature** for development and monitoring workflows, such as:

- Monitoring training progress from another computer on your network
- Mobile device access to check GPU status and system health
- Team development environments where multiple users need access
- Running the server on a dedicated machine while accessing from workstations

**By default**, the Studio launches on `localhost:7860`, which is **only** accessible from the machine running it.

**To enable LAN access**, you would need to explicitly bind the server to `0.0.0.0` in `studio/app.py` (at the bottom of the file in the `uvicorn.run()` call). This makes the server accessible to other devices on your local network.

### Safe LAN Use Recommendations

If you enable LAN access:

1. **Use a firewall allowlist:** Only permit connections from known/trusted IP addresses
2. **Do NOT port-forward** the Studio to the public internet
3. **Do NOT expose** to public networks (coffee shops, hotels, etc.)
4. **Use a VPN** if accessing remotely over the internet
5. **Consider adding authentication:** The Studio does not include built-in authentication by default
6. **Use TLS/HTTPS:** If exposing beyond `localhost`, configure HTTPS to encrypt traffic
7. **Review logs:** Monitor access logs for unexpected connections
8. **Isolate the network:** Run on a dedicated development network, not your primary home/office network with IoT devices

**Never expose the Studio directly to the public internet without proper security controls** (authentication, TLS, network isolation, and ideally a reverse proxy with rate limiting).

---

## Threat Model (What We Consider In-Scope)

### In-scope

These are security issues we actively address:

- **Path traversal / arbitrary file read/write** issues in the UI or backend
- **Command injection** or unsafe process execution (e.g., conversion scripts, training commands)
- **Remote code execution** vulnerabilities
- **Insecure defaults** that expose local services unintentionally
- **Sensitive data leakage** through logs, debug output, or generated artifacts
- **Dependency vulnerabilities** introduced by pinned/unpinned packages (when actionable)

### Partially in-scope (best-effort)

We'll address these when feasible, but they're not guaranteed to be fixed:

- Denial-of-service via extremely large files or malicious document structures
- Resource exhaustion attacks (VRAM/RAM/disk fill) triggered by inputs
- Prompt-injection behaviors in dataset generation (mitigations welcomed)

---

## Out of Scope (What This Project Does Not Claim to Secure)

The following are **not** security issues in this project:

- The security of your operating system, drivers, or GPU stack
- The security of any base model you download (and its license compliance)
- Protection against all forms of "model misbehavior" (hallucination, toxicity, bias)
- Risks inherent to training on proprietary or sensitive documents
- Third-party tools you install (Tesseract, llama.cpp, Ollama, etc.)
- Man-in-the-middle attacks on unencrypted local network traffic (use TLS/VPN if concerned)
- Physical access to the machine running the Studio
- Social engineering attacks

---

## Security Best Practices (Strongly Recommended)

### Keep it local by default

- Run the UI on `localhost:7860` for single-machine use
- Only enable LAN access (`0.0.0.0` binding) when explicitly needed
- Do **not** expose the service to the public internet without:
  - Strong authentication/authorization
  - Network restrictions (firewall allowlist)
  - TLS/HTTPS encryption
  - Rate limiting and monitoring
  - Careful security review

### Treat training data as sensitive

- Only train on content you have the right to use
- Avoid personal, confidential, or regulated data unless you have explicit permission and compliance controls
- Remember: **trained models can memorize and reproduce training data**

### Avoid committing secrets

- Do not store tokens/keys in repo files
- Use environment variables or a local `.env` file
- Ensure `.env` is in `.gitignore` (it is by default)
- Rotate tokens/API keys periodically

### Review exports before sharing

- Merged models and adapters can encode sensitive patterns from your training data
- Do not upload trained artifacts publicly unless you are sure your dataset is safe to disclose
- Test inference on sensitive topics to ensure no data leakage

### Keep dependencies updated

- Periodically run `pip list --outdated` to check for updates
- Review security advisories for `transformers`, `torch`, `fastapi`, and other dependencies
- Update dependencies when critical security fixes are released

### Use virtual environments

- Always use `.venv` as created by `setup_llm.bat`
- Do not install packages globally
- Isolate this project from other Python projects

### Secure your base models

- Download models from trusted sources (Hugging Face, official repos)
- Verify checksums/signatures when available
- Be aware that base models may have their own security considerations (prompt injection, jailbreaks, etc.)

### Monitor system resources

- Set up alerts for unusual CPU/GPU/memory usage
- Watch disk space—training outputs can grow large
- Monitor network traffic if LAN access is enabled

---

## Coordinated Disclosure

We support coordinated disclosure and will credit reporters if they want attribution.

Please allow a reasonable time window for fixes before publishing details publicly. We aim for:

- **Critical issues:** Fix within 7 days
- **High severity:** Fix within 30 days
- **Medium/Low:** Fix in next release cycle

---

## Security Checklist for Deployment

Before deploying or sharing access to your Studio:

- [ ] Running on `localhost` only (unless LAN access explicitly needed)?
- [ ] Firewall rules configured to block public internet access?
- [ ] No sensitive data in training documents?
- [ ] `.env` file in `.gitignore`?
- [ ] No API keys or tokens committed to repo?
- [ ] Dependencies up-to-date?
- [ ] TLS/HTTPS configured if exposing beyond `localhost`?
- [ ] Authentication/authorization configured if multi-user?
- [ ] Logs reviewed for unusual access patterns?
- [ ] Trained models reviewed for data leakage before sharing?

---

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CWE - Common Weakness Enumeration](https://cwe.mitre.org/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)

---

Thank you for helping keep NamelyCorp LLM Studio safe and secure. Your responsible disclosure helps protect all users of this project.

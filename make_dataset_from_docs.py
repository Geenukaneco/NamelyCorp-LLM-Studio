# make_dataset_from_docs.py
# Build a Q&A dataset from mixed docs (PDF/DOCX/TXT/CSV/XLSX) with high-quality OCR fallback.
# Outputs data_qa.csv with columns: question, answer

from __future__ import annotations
import os, re, csv, sys, argparse
from pathlib import Path
from typing import List, Tuple, Optional

# ---------- Third-party ----------
# pip install pymupdf pypdf (optional), python-docx pandas openpyxl tabulate pdf2image pytesseract pillow transformers
import fitz  # PyMuPDF
import pandas as pd
from docx import Document as DocxDocument

from PIL import Image, ImageOps, ImageFilter
import pytesseract
from pdf2image import convert_from_path

from transformers import AutoTokenizer

# ---------- Defaults (Windows-friendly) ----------
DEFAULT_DOCS_DIR = Path(r"C:\dev\llm\docs")
DEFAULT_OUT_CSV  = Path(r"C:\dev\llm\data_qa.csv")
DEFAULT_MODEL_ID = r"C:\models\Llama-3.2-3B-Instruct"  # or "meta-llama/Llama-3.2-3B-Instruct"

CHARS_PER_CHUNK  = 1400    # target text chunk size pre-templating
MAX_TOKENS       = 1024    # train-time max seq len
MIN_ANSWER_CHARS = 40      # drop tiny/noise answers

# OCR tuning
OCR_DPI          = 300     # 300–400 works well
OCR_LANG         = "eng"   # add more: "eng+deu" etc.
OCR_CONF_MIN     = 70      # ignore words below this confidence (0–100)
OCR_PSM          = 6       # Assume a block of text
# If Tesseract is not on PATH, uncomment and set:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- Helpers ----------
def clean_text(t: str) -> str:
    t = t.replace("\x00", " ").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t

def chunk_text(t: str, chars: int) -> List[str]:
    t = clean_text(t)
    if not t:
        return []
    if len(t) <= chars:
        return [t]
    out, start, L = [], 0, len(t)
    while start < L:
        end = min(L, start + chars)
        cut = max(t.rfind(". ", start, end), t.rfind("\n", start, end))
        if cut == -1 or cut < start + int(chars * 0.6):
            cut = end
        seg = t[start:cut].strip()
        if seg:
            out.append(seg)
        start = cut
    return out

def df_to_markdown(df: pd.DataFrame, max_rows=12, max_cols=10) -> str:
    df = df.copy()
    if df.shape[0] > max_rows: df = df.head(max_rows)
    if df.shape[1] > max_cols: df = df.iloc[:, :max_cols]
    return df.to_markdown(index=False)

# ---------- Tokenization ----------
def ensure_tokenizer(model_id: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok

def count_tokens_chat(tok: AutoTokenizer, q: str, a: str) -> int:
    msgs = [
        {"role":"system","content":"You are a careful assistant. Answer strictly from documented facts."},
        {"role":"user","content":q},
        {"role":"assistant","content":a},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return len(tok(prompt, truncation=False, add_special_tokens=False)["input_ids"])

def adapt_chunk_for_tokens(tok: AutoTokenizer, fname: str, raw_chunk: str,
                           max_tokens: int, base_chars: int) -> List[str]:
    """If a single (Q,A) would exceed max_tokens, adaptively re-split smaller."""
    chunks = chunk_text(raw_chunk, base_chars)
    out: List[str] = []
    for ch in chunks:
        q = f"Summarize the following section from {fname} and include the source note:\n\n(source: {fname})\n{ch}"
        a = f"(source: {fname})\n{ch}"
        if count_tokens_chat(tok, q, a) <= max_tokens:
            out.append(ch)
            continue
        # Split more aggressively
        small_chars = max(300, int(len(ch) * 0.6))
        subs = chunk_text(ch, small_chars)
        for sc in subs:
            q2 = f"Summarize the following section from {fname} and include the source note:\n\n(source: {fname})\n{sc}"
            a2 = f"(source: {fname})\n{sc}"
            if count_tokens_chat(tok, q2, a2) <= max_tokens:
                out.append(sc)
            else:
                for s2 in chunk_text(sc, 300):
                    out.append(s2)
    return out

# ---------- OCR (quality-first) ----------
def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Gentle denoise + grayscale + adaptive contrast + slight sharpening.
    Avoids over-aggressive thresholding that can erase thin fonts.
    """
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    # mild de-noise
    g = g.filter(ImageFilter.MedianFilter(size=3))
    # slight sharpen
    g = g.filter(ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=3))
    return g

def ocr_page_image(img: Image.Image, lang=OCR_LANG, psm=OCR_PSM, conf_min=OCR_CONF_MIN) -> str:
    config = f"--psm {psm}"
    data = pytesseract.image_to_data(_preprocess_for_ocr(img), lang=lang, config=config, output_type=pytesseract.Output.DICT)
    words, confs, line_nums, block_nums, par_nums, page_nums = (
        data["text"], data["conf"], data["line_num"], data["block_num"], data["par_num"], data["page_num"]
    )
    # Rebuild text by lines with confidence filtering
    lines = {}
    for w, c, ln, bn, pn in zip(words, confs, line_nums, block_nums, par_nums):
        try:
            c = float(c)
        except:  # some Tesseract builds return -1 or ''
            c = -1
        if not w or c < conf_min:
            continue
        key = (bn, pn, ln)
        lines.setdefault(key, []).append(w)
    ordered = []
    for key in sorted(lines.keys()):
        ordered.append(" ".join(lines[key]))
    return clean_text("\n".join(ordered))

def ocr_pdf_page(path: Path, page_index: int, dpi=OCR_DPI) -> str:
    """Render a single PDF page to image and OCR."""
    images = convert_from_path(str(path), dpi=dpi, first_page=page_index+1, last_page=page_index+1)
    texts = []
    for img in images:
        texts.append(ocr_page_image(img))
    return clean_text("\n".join(texts))

# ---------- Readers ----------
def read_pdf(path: Path, chars_per_chunk: int, use_ocr: bool) -> Tuple[str, List[str]]:
    """
    Prefer PyMuPDF text; if empty/suspicious and OCR enabled, OCR that page.
    """
    chunks: List[str] = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc):
            try:
                # "text" extractor gives best fidelity for most docs
                txt = page.get_text("text") or ""
            except Exception:
                txt = ""
            txt = clean_text(txt)

            # Heuristic: treat as "no text" if length very small OR whitespace ratio high
            def poor(txt_: str) -> bool:
                if len(txt_) < 40: return True
                letters = sum(ch.isalnum() for ch in txt_)
                return letters / max(1, len(txt_)) < 0.15

            if (not txt or poor(txt)) and use_ocr:
                try:
                    txt = ocr_pdf_page(path, i) or txt
                except Exception as e:
                    print(f"[WARN] OCR failed on {path.name} p.{i+1}: {e}")

            if not txt:
                continue

            for j, c in enumerate(chunk_text(txt, chars_per_chunk)):
                chunks.append(f"(source: {path.name} p.{i+1} c.{j+1})\n{c}")

    return path.name, chunks

def read_docx(path: Path, chars_per_chunk: int) -> Tuple[str, List[str]]:
    doc = DocxDocument(str(path))
    parts = [clean_text(p.text) for p in doc.paragraphs if clean_text(p.text)]
    full = "\n".join(parts)
    return path.name, [f"(source: {path.name})\n{c}" for c in chunk_text(full, chars_per_chunk)]

def read_txt(path: Path, chars_per_chunk: int) -> Tuple[str, List[str]]:
    full = path.read_text(encoding="utf-8", errors="ignore")
    return path.name, [f"(source: {path.name})\n{c}" for c in chunk_text(full, chars_per_chunk)]

def read_csv_file(path: Path, chars_per_chunk: int, nrows: int) -> Tuple[str, List[str]]:
    try:
        df = pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(path, nrows=nrows, sep=None, engine="python", low_memory=False, encoding="utf-8")
        except Exception:
            df = pd.read_csv(path, nrows=nrows, sep=None, engine="python", low_memory=False, encoding="latin-1")
    md = df_to_markdown(df)
    return path.name, [f"(source: {path.name})\n{c}" for c in chunk_text(md, chars_per_chunk)]

def read_xlsx(path: Path, chars_per_chunk: int) -> Tuple[str, List[str]]:
    xls = pd.ExcelFile(path)
    chunks = []
    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet)
            md = f"# Sheet: {sheet}\n" + df_to_markdown(df)
            for c in chunk_text(md, chars_per_chunk):
                chunks.append(f"(source: {path.name} | sheet: {sheet})\n{c}")
        except Exception as e:
            print(f"[WARN] Skipping sheet '{sheet}' in {path.name}: {e}")
    return path.name, chunks

# ---------- Extraction pipeline ----------
def extract_all(
    docs_dir: Path,
    chars_per_chunk: int,
    csv_nrows: int,
    max_files: Optional[int],
    tokenizer: Optional[AutoTokenizer],
    max_tokens: int,
    use_ocr: bool,
    ocr_lang: str,
    mix_summary_pattern: bool = True
) -> List[Tuple[str, str]]:
    # allow dynamic OCR locale
    if ocr_lang:
        try:
            pytesseract.pytesseract.tesseract_cmd  # probe
        except Exception:
            pass  # fine
    qa_pairs: List[Tuple[str, str]] = []

    files = [p for p in docs_dir.rglob("*") if p.is_file()]
    if max_files: files = files[:max_files]
    print(f"[INFO] Scanning {len(files)} files in {docs_dir} ...")

    for idx, p in enumerate(files, start=1):
        ext = p.suffix.lower()
        try:
            if ext == ".pdf":
                fname, chunks = read_pdf(p, chars_per_chunk, use_ocr=use_ocr)
            elif ext == ".docx":
                fname, chunks = read_docx(p, chars_per_chunk)
            elif ext in (".txt", ".md", ".rtf"):
                fname, chunks = read_txt(p, chars_per_chunk)
            elif ext == ".csv":
                fname, chunks = read_csv_file(p, chars_per_chunk, csv_nrows)
            elif ext in (".xlsx", ".xlsm", ".xls"):
                fname, chunks = read_xlsx(p, chars_per_chunk)
            else:
                continue
        except Exception as e:
            print(f"[WARN] Skipping {p.name}: {e}")
            continue

        if not chunks:
            continue

        # Token-aware adaptation: guarantee each (Q,A) fits max_tokens
        if tokenizer is not None:
            adapted: List[str] = []
            for c in chunks:
                marker, body = c.split("\n", 1) if "\n" in c else ("", c)
                for sc in adapt_chunk_for_tokens(tokenizer, fname, body, max_tokens, chars_per_chunk):
                    adapted.append(f"{marker}\n{sc}" if marker else sc)
            chunks = adapted

        for k, c in enumerate(chunks, start=1):
            if mix_summary_pattern and (k % 2 == 0):
                q = f"Summarize the key points from {fname} (include the source note) in concise bullet points:\n\n{c}"
            else:
                q = f"Summarize the following section from {fname} and include the source note:\n\n{c}"
            a = c
            qa_pairs.append((q, a))

        if idx % 5 == 0 or idx == len(files):
            print(f"[INFO] Processed {idx}/{len(files)} ...")
    return qa_pairs

def postprocess_and_write(qa: List[Tuple[str, str]], out_csv: Path) -> None:
    # de-dup & drop tiny answers
    seen, out = set(), []
    for q, a in qa:
        if len(a.strip()) < MIN_ANSWER_CHARS: continue
        key = (q.strip(), a.strip())
        if key in seen: continue
        seen.add(key)
        out.append((q, a))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["question","answer"]); w.writerows(out)
    print(f"[DONE] Wrote {len(out)} Q&A rows to {out_csv}")

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Build Q&A dataset from mixed docs with high-quality OCR fallback.")
    ap.add_argument("--docs-dir", type=str, default=str(DEFAULT_DOCS_DIR))
    ap.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--model-id", type=str, default=str(DEFAULT_MODEL_ID),
                    help="Local model folder or HF repo id for tokenizer (token-aware).")
    ap.add_argument("--chars-per-chunk", type=int, default=CHARS_PER_CHUNK)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--csv-nrows", type=int, default=5000)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--no-token-check", action="store_true", help="Disable token-aware splitting.")
    ap.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback for PDFs.")
    ap.add_argument("--ocr-lang", type=str, default=OCR_LANG, help="Tesseract language, e.g., 'eng' or 'eng+spa'.")
    ap.add_argument("--no-bullets", action="store_true", help="Disable alternating bullet-summary prompts.")
    return ap.parse_args()

def main():
    args = parse_args()
    docs_dir, out_csv = Path(args.docs_dir), Path(args.out_csv)

    if not docs_dir.exists():
        print(f"[INFO] Creating docs folder: {docs_dir}")
        docs_dir.mkdir(parents=True, exist_ok=True)
        print("[INFO] Put files there and re-run.")
        return

    # Try to load tokenizer for token-aware checks
    tok = None
    if not args.no_token_check:
        try:
            mid = args.model_id
            if mid and "\\" in mid and Path(mid).exists():
                tok = ensure_tokenizer(mid)
            else:
                tok = ensure_tokenizer(mid or "meta-llama/Llama-3.2-3B-Instruct")
            print(f"[INFO] Tokenizer loaded for token checks: {args.model_id}")
        except Exception as e:
            print(f"[WARN] Tokenizer not available ({e}). Proceeding without token checks.")

    qa = extract_all(
        docs_dir=docs_dir,
        chars_per_chunk=args.chars_per_chunk,
        csv_nrows=args.csv_nrows,
        max_files=args.max_files,
        tokenizer=tok,
        max_tokens=args.max_tokens,
        use_ocr=(not args.no_ocr),
        ocr_lang=args.ocr_lang,
        mix_summary_pattern=(not args.no_bullets)
    )

    if not qa:
        print(f"[INFO] No content found in {docs_dir}.")
        return

    postprocess_and_write(qa, out_csv)

if __name__ == "__main__":
    main()

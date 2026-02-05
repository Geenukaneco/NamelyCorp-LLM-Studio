# validate_qa.py
# Sanity-check a Q&A CSV for fine-tuning chat LLMs (Llama 3.2-3B-Instruct by default).
# - Ensures headers, non-empty fields, no weird whitespace
# - Tokenizes with the model's chat template to verify each (Q,A) fits max tokens
# - Flags duplicates and very short/very long entries
# - Can write a CLEANED CSV and a Markdown report with actionable stats

from __future__ import annotations
import os, sys, csv, argparse, math, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
from transformers import AutoTokenizer

# ------------ Defaults (Windows-friendly) ------------
DEFAULT_CSV       = Path(r"C:\dev\llm\data_qa.csv")
DEFAULT_MODEL_ID  = r"C:\models\Llama-3.2-3B-Instruct"  # or "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_MAXTOK    = 1024
MIN_ANSWER_CHARS  = 40           # drop/flag trivial answers
MIN_QUESTION_CHARS= 6            # flag suspiciously short questions
SAMPLE_PRINT      = 3            # examples per section in report

SYSTEM_PROMPT = "You are a careful assistant. Answer strictly from documented facts."

# ------------ Data structures ------------
@dataclass
class RowCheck:
    idx: int
    question: str
    answer: str
    q_chars: int
    a_chars: int
    tokens: int
    issues: List[str]  # list of issue tags

@dataclass
class Summary:
    total: int
    ok: int
    empty_fields: int
    short_q: int
    short_a: int
    overlong: int
    dup_questions: int
    dup_pairs: int

# ------------ Helpers ------------
def load_tokenizer(model_id: str) -> AutoTokenizer:
    """
    Loads tokenizer from local folder if it exists; otherwise from HF hub.
    Sets pad_token to eos_token if missing (avoids warnings).
    """
    local = Path(model_id)
    mid = model_id if (local.exists() and local.is_dir()) else (model_id or "meta-llama/Llama-3.2-3B-Instruct")
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok

def chat_token_len(tok: AutoTokenizer, question: str, answer: str) -> int:
    msgs = [
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role":"user", "content": question},
        {"role":"assistant","content": answer},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ids = tok(prompt, truncation=False, add_special_tokens=False)["input_ids"]
    return len(ids)

def normalize(s: str) -> str:
    return (s or "").replace("\u0000", " ").strip()

def read_csv_strict(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)  # treat empty cells as ""
    # enforce exact headers
    expected = ["question","answer"]
    if list(df.columns) != expected:
        raise ValueError(f"CSV headers must be exactly {expected}, got {list(df.columns)}")
    rows = df.to_dict(orient="records")
    # normalize whitespace
    for r in rows:
        r["question"] = normalize(r["question"])
        r["answer"]   = normalize(r["answer"])
    return rows

def pct(x: int, n: int) -> str:
    return f"{(100*x/n):.1f}%" if n else "0.0%"

def head_examples(items: List[RowCheck], tag: str, k: int) -> List[RowCheck]:
    return [rc for rc in items if tag in rc.issues][:k]

def dedupe(rows: List[Dict[str,str]]) -> Tuple[List[int], List[int]]:
    """
    Returns indices (1-based) for duplicate questions and duplicate (q,a) pairs.
    """
    q_seen, qa_seen = {}, {}
    dup_q_idx, dup_qa_idx = [], []

    for i, r in enumerate(rows, start=1):
        q = r["question"].strip().lower()
        qa = (r["question"].strip().lower(), r["answer"].strip().lower())
        if q in q_seen:
            dup_q_idx.append(i)
        else:
            q_seen[q] = i
        if qa in qa_seen:
            dup_qa_idx.append(i)
        else:
            qa_seen[qa] = i
    return dup_q_idx, dup_qa_idx

# ------------ Validation core ------------
def validate(
    csv_path: Path,
    model_id: str,
    max_tokens: int,
    allow_overlong: bool = False,
    sample_limit: Optional[int] = None
) -> Tuple[List[RowCheck], Summary]:
    tok = load_tokenizer(model_id)
    rows = read_csv_strict(csv_path)
    if sample_limit:
        rows = rows[:sample_limit]

    dup_q_idx, dup_qa_idx = dedupe(rows)

    checks: List[RowCheck] = []
    start = time.time()

    for i, r in enumerate(rows, start=1):
        q = r["question"]; a = r["answer"]
        issues: List[str] = []

        if not q or not a:
            issues.append("empty_fields")
        if q and len(q) < MIN_QUESTION_CHARS:
            issues.append("short_q")
        if a and len(a) < MIN_ANSWER_CHARS:
            issues.append("short_a")

        # tokenize with chat template
        toks = 0
        try:
            toks = chat_token_len(tok, q, a) if q and a else 0
            if toks > max_tokens:
                issues.append("overlong")
        except Exception as e:
            issues.append(f"tokenize_error:{str(e)[:80]}")

        if i in dup_q_idx:
            issues.append("dup_question")
        if i in dup_qa_idx:
            issues.append("dup_pair")

        checks.append(RowCheck(
            idx=i, question=q, answer=a,
            q_chars=len(q), a_chars=len(a),
            tokens=toks, issues=issues
        ))

        if i % 200 == 0:
            elapsed = time.time() - start
            print(f"[INFO] Checked {i}/{len(rows)} rows ({i/elapsed:.1f} r/s) ...")

    # summarize
    total = len(checks)
    ok = sum(1 for c in checks if not c.issues or (allow_overlong and c.issues == ["overlong"]))
    summary = Summary(
        total=total,
        ok=ok,
        empty_fields=sum(1 for c in checks if "empty_fields" in c.issues),
        short_q=sum(1 for c in checks if "short_q" in c.issues),
        short_a=sum(1 for c in checks if "short_a" in c.issues),
        overlong=sum(1 for c in checks if "overlong" in c.issues),
        dup_questions=sum(1 for c in checks if "dup_question" in c.issues),
        dup_pairs=sum(1 for c in checks if "dup_pair" in c.issues),
    )
    return checks, summary

# ------------ Reporting / outputs ------------
def write_clean_csv(checks: List[RowCheck], out_path: Path, keep_overlong: bool = False) -> int:
    """
    Writes a cleaned CSV removing rows with empty fields, dup pairs, and trivial answers.
    If keep_overlong=False (default), drops rows exceeding max_tokens.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question","answer"])
        for c in checks:
            # drop duplicates (pair), empty fields, too-short answers
            if "empty_fields" in c.issues:            continue
            if "dup_pair" in c.issues:                continue
            if "short_a" in c.issues:                 continue
            if not keep_overlong and "overlong" in c.issues:  continue
            w.writerow([c.question, c.answer])
            kept += 1
    return kept

def write_report_md(
    checks: List[RowCheck], summ: Summary, out_md: Path, max_tokens: int, sample_n: int = SAMPLE_PRINT
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    longest = sorted(checks, key=lambda c: c.tokens, reverse=True)[:sample_n]
    overlong_samples = head_examples(checks, "overlong", sample_n)
    short_a_samples  = head_examples(checks, "short_a",  sample_n)
    dup_q_samples    = head_examples(checks, "dup_question", sample_n)
    dup_pair_samples = head_examples(checks, "dup_pair", sample_n)

    with out_md.open("w", encoding="utf-8") as f:
        f.write(f"# Q&A Validation Report\n\n")
        f.write(f"- **Rows:** {summ.total}\n")
        f.write(f"- **OK:** {summ.ok} ({pct(summ.ok, summ.total)})\n")
        f.write(f"- **Empty fields:** {summ.empty_fields}\n")
        f.write(f"- **Short questions (<{MIN_QUESTION_CHARS} chars):** {summ.short_q}\n")
        f.write(f"- **Short answers (<{MIN_ANSWER_CHARS} chars):** {summ.short_a}\n")
        f.write(f"- **Overlong examples (>{max_tokens} tokens w/ chat template):** {summ.overlong}\n")
        f.write(f"- **Duplicate questions:** {summ.dup_questions}\n")
        f.write(f"- **Duplicate pairs:** {summ.dup_pairs}\n\n")

        def block(title: str, items: List[RowCheck]):
            if not items: return
            f.write(f"## {title}\n\n")
            for c in items:
                preview_q = (c.question[:220] + "…") if len(c.question) > 220 else c.question
                preview_a = (c.answer[:220] + "…") if len(c.answer) > 220 else c.answer
                f.write(f"- Row **{c.idx}** • **{c.tokens}** tokens • issues: `{', '.join(c.issues)}`\n\n")
                f.write(f"  - Q: {preview_q}\n\n")
                f.write(f"  - A: {preview_a}\n\n")
            f.write("\n")

        block("Top longest (by tokens)", longest)
        block(f"Overlong examples (> {max_tokens} tokens)", overlong_samples)
        block(f"Short answers (< {MIN_ANSWER_CHARS} chars)", short_a_samples)
        block("Duplicate questions", dup_q_samples)
        block("Duplicate Q&A pairs", dup_pair_samples)

# ------------ CLI ------------
def parse_args():
    ap = argparse.ArgumentParser(description="Validate a Q&A CSV for Llama-style chat fine-tuning.")
    ap.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Path to data_qa.csv")
    ap.add_argument("--model-id", type=str, default=str(DEFAULT_MODEL_ID),
                    help="Local model folder or HF repo id for tokenizer.")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAXTOK,
                    help="Max tokens per (Q,A) including chat template.")
    ap.add_argument("--sample-limit", type=int, default=None,
                    help="Validate only first N rows (faster dev cycles).")
    ap.add_argument("--allow-overlong", action="store_true",
                    help="Treat 'overlong' as OK (report only).")
    ap.add_argument("--write-clean", type=str, default="",
                    help="If set, write a cleaned CSV to this path.")
    ap.add_argument("--report", type=str, default="",
                    help="If set, write a Markdown report to this path.")
    return ap.parse_args()

def main():
    args = parse_args()
    csv_path = Path(args.csv)

    print(f"[INFO] CSV:       {csv_path}")
    print(f"[INFO] Model:     {args.model_id}")
    print(f"[INFO] MaxTokens: {args.max_tokens}")
    if args.sample_limit:
        print(f"[INFO] Sample:    first {args.sample_limit} rows")

    checks, summ = validate(
        csv_path=csv_path,
        model_id=args.model_id,
        max_tokens=args.max_tokens,
        allow_overlong=args.allow_overlong,
        sample_limit=args.sample_limit
    )

    print("\n=== SUMMARY ===")
    print(f"Rows: {summ.total}")
    print(f"OK: {summ.ok} ({pct(summ.ok, summ.total)})")
    print(f"Empty fields: {summ.empty_fields}")
    print(f"Short questions (<{MIN_QUESTION_CHARS} chars): {summ.short_q}")
    print(f"Short answers  (<{MIN_ANSWER_CHARS} chars): {summ.short_a}")
    print(f"Overlong (>{args.max_tokens} tokens): {summ.overlong}")
    print(f"Duplicate questions: {summ.dup_questions}")
    print(f"Duplicate pairs:     {summ.dup_pairs}")

    if args.write_clean:
        out_csv = Path(args.write_clean)
        kept = write_clean_csv(checks, out_csv, keep_overlong=args.allow_overlong)
        print(f"[CLEAN] Wrote {kept} rows -> {out_csv}")

    if args.report:
        out_md = Path(args.report)
        write_report_md(checks, summ, out_md, max_tokens=args.max_tokens)
        print(f"[REPORT] {out_md}")

    # exit code for CI
    if summ.empty_fields or summ.dup_pairs:
        # Hard failures you probably want to fix before training
        sys.exit(2)
    elif summ.overlong and not args.allow_overlong:
        # Training will truncate these unless you shorten/split
        sys.exit(3)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()

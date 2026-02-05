# train_ft.py
# Supervised fine-tuning (LoRA) for Llama-3.2-3B-Instruct on a Q&A CSV.
# - Uses chat template correctly
# - Masks labels so loss is only on assistant spans
# - LoRA on attention + MLP (safe defaults)
# - Train/val split, perplexity metric, bf16/fp16 auto, resume, gradient checkpointing
# - Saves both: LoRA adapters and a merged full model

from __future__ import annotations
import os, sys, math, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType

# ---------------- Defaults (Windows-friendly) ----------------
DEFAULT_CSV       = r"C:\dev\llm\data_qa.csv"
DEFAULT_MODEL_ID  = r"C:\models\Llama-3.2-3B-Instruct"  # or "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_OUTDIR    = r"C:\dev\llm\ft_out"
DEFAULT_MAX_LEN   = 1024
SYSTEM_PROMPT     = "You are a careful assistant. Answer strictly from the organization’s documented facts."

# ---------------- Utilities ----------------
def bf16_supported() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+
    except Exception:
        return False

def load_tokenizer(model_id: str) -> AutoTokenizer:
    local = os.path.isdir(model_id)
    mid = model_id if local else (model_id or "meta-llama/Llama-3.2-3B-Instruct")
    tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok

def build_messages(question: str, answer: Optional[str] = None) -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs

def tokenize_with_label_mask(
    tok: AutoTokenizer, question: str, answer: str, max_len: int
) -> Dict[str, List[int]]:
    """
    Create input_ids/labels such that loss is applied only on the assistant tokens.
    Steps:
      1) 'prompt' ids: system+user with add_generation_prompt=True (includes assistant header)
      2) 'full'   ids: system+user+assistant (complete turn)
      3) labels = [-100]*len(prompt) + full[len(prompt):]
    """
    # prompt-only (no assistant content)
    prompt = tok.apply_chat_template(
        build_messages(question), tokenize=True,
        add_generation_prompt=True, return_tensors=None
    )
    prompt_ids: List[int] = prompt["input_ids"] if isinstance(prompt, dict) else prompt

    # full turn (with assistant content)
    full = tok.apply_chat_template(
        build_messages(question, answer), tokenize=True,
        add_generation_prompt=False, return_tensors=None
    )
    full_ids: List[int] = full["input_ids"] if isinstance(full, dict) else full

    # Truncate from the left if too long (keeps end-of-sequence)
    if len(full_ids) > max_len:
        # try to keep the tail (assistant text tends to be near the end)
        full_ids = full_ids[-max_len:]

    # If prompt is longer than full (after truncation), clip prompt
    if len(prompt_ids) > len(full_ids):
        prompt_ids = prompt_ids[: len(full_ids)]

    # Pad up to max_len (to keep fixed shape for Trainer)
    attention_mask = [1] * len(full_ids)
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]

    # final pad
    if len(full_ids) < max_len:
        pad_len = max_len - len(full_ids)
        full_ids += [tok.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

    assert len(full_ids) == len(labels) == len(attention_mask) == max_len
    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}

# ---------------- Data Collator ----------------
@dataclass
class SimpleCollator:
    tokenizer: PreTrainedTokenizerBase
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {k: [f[k] for f in features] for k in features[0].keys()}
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

# ---------------- Metrics ----------------
def compute_metrics(eval_pred):
    # Trainer returns (loss only), but we’ll compute perplexity from it at the end.
    # Returning empty dict keeps logs clean; we print ppl after eval loop.
    return {}

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="LoRA SFT for Llama-3.2-3B-Instruct on a Q&A CSV.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to data_qa.csv")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                        help="Local folder or HF repo id")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch")
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.05, help="Fraction for validation split")
    parser.add_argument("--save-steps", type=int, default=0, help="0=no checkpoints (save last only)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if present")
    parser.add_argument("--merge-full", action="store_true", help="Also save merged full model at end")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-mlp", action="store_true", help="LoRA only on q/k/v/o (skip MLP)")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load tokenizer & model
    tok = load_tokenizer(args.model_id)

    # Prefer bf16 on >=Ampere, otherwise fp16
    dtype_flag = {}
    if bf16_supported():
        dtype_flag = {"torch_dtype": torch.bfloat16}
        print("[INFO] Using bf16")
    else:
        dtype_flag = {"torch_dtype": torch.float16}
        print("[INFO] Using fp16")

    print("[INFO] Loading base model…")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        attn_implementation="sdpa",   # faster/leaner attention in recent PyTorch
        **dtype_flag
    )

    # CRITICAL: Enable gradient checkpointing and input gradients for LoRA
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA config
    target_modules = ["q_proj","k_proj","v_proj","o_proj"]
    if not args.no_mlp:
        target_modules += ["gate_proj","up_proj","down_proj"]  # MLP LoRA improves style/content
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Dataset
    print("[INFO] Loading dataset…")
    raw = load_dataset("csv", data_files=args.csv)["train"].train_test_split(test_size=args.val_size, seed=args.seed)
    train_ds, val_ds = raw["train"], raw["test"]

    # Map → tokenized with label masking
    def mapper(example):
        q = (example["question"] or "").strip()
        a = (example["answer"] or "").strip()
        return tokenize_with_label_mask(tok, q, a, max_len=args.max_len)

    train_tok = train_ds.map(mapper, remove_columns=train_ds.column_names, num_proc=None)
    val_tok   = val_ds.map(mapper,   remove_columns=val_ds.column_names,   num_proc=None)

    collator = SimpleCollator(tok)

    # Training args
    os.makedirs(args.outdir, exist_ok=True)
    fp16_flag = (not bf16_supported())
    training_args = TrainingArguments(
        output_dir=args.outdir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=args.save_steps if args.save_steps > 0 else 10_000_000,  # effectively "no checkpoints"
        save_total_limit=1,
        bf16=bf16_supported(),
        fp16=fp16_flag,
        # gradient_checkpointing=True,  # Enabled manually above
        report_to="none",
        dataloader_num_workers=2,
        optim="adamw_torch",
        run_name="llama32_3b_lora",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    # Resume?
    resume_ckpt = None
    if args.resume:
        # look for the last checkpoint under outdir
        checkpoints = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_ckpt = sorted(checkpoints, key=lambda p: int(p.rsplit("-",1)[-1]))[-1]
            print(f"[INFO] Resuming from {resume_ckpt}")

    print("[INFO] Starting training…")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    print("[INFO] Evaluating…")
    eval_out = trainer.evaluate()
    eval_loss = eval_out.get("eval_loss", None)
    if eval_loss is not None:
        try:
            ppl = math.exp(eval_loss)
            print(f"[EVAL] loss={eval_loss:.4f}  ppl={ppl:.2f}")
        except OverflowError:
            print(f"[EVAL] loss={eval_loss:.4f}  ppl=inf")

    # Save LoRA adapters
    adapters_dir = os.path.join(args.outdir, "lora_adapters")
    os.makedirs(adapters_dir, exist_ok=True)
    trainer.model.save_pretrained(adapters_dir)
    tok.save_pretrained(args.outdir)
    print(f"[SAVE] LoRA adapters → {adapters_dir}")

    # Optionally merge adapters into full model and save
    if args.merge_full:
        print("[INFO] Merging LoRA into base weights (this can take a minute)…")
        merged = trainer.model.merge_and_unload()
        merged_dir = os.path.join(args.outdir, "merged_model")
        os.makedirs(merged_dir, exist_ok=True)
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        print(f"[SAVE] Merged full model → {merged_dir}")

    print("[DONE] Training complete.")

if __name__ == "__main__":
    main()

# test_inference.py
"""
Test your fine-tuned model after training completes
Run: python test_inference.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

print("=" * 80)
print("🧪 Testing Fine-Tuned Llama 3.2 Model")
print("=" * 80)
print()

# Model path
MODEL_PATH = r"C:\dev\llm\ft_out\merged_model"

print(f"[1/3] Loading model from: {MODEL_PATH}")
print("      This may take 2-3 minutes...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✅ Model loaded successfully")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM allocated: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
    else:
        print("⚠️  Running on CPU (will be slow)")
        
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    print()
    print("Make sure training completed successfully and the model exists at:")
    print(f"  {MODEL_PATH}")
    exit(1)

print()
print("[2/3] Creating inference pipeline...")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

print("✅ Pipeline ready")
print()
print("[3/3] Testing with sample questions...")
print("=" * 80)

# Test questions from your training data
test_questions = [
    "What is LoRA fine-tuning and why is it used?",
    "How do I validate my Q&A dataset?",
    "What GPU is recommended for training Llama models?",
    "What is the LLM Training Studio?",
    "How does masked loss training work?",
]

for i, question in enumerate(test_questions, 1):
    print()
    print(f"Question {i}/{len(test_questions)}:")
    print(f"Q: {question}")
    print()
    
    # Format as chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based on your training data."},
        {"role": "user", "content": question}
    ]
    
    # Generate
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    try:
        result = pipe(
            prompt,
            max_new_tokens=200,
            do_sample=False,  # Deterministic
            temperature=1.0,
            top_p=1.0,
            return_full_text=False
        )
        
        answer = result[0]["generated_text"]
        print(f"A: {answer}")
        
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
    
    print("-" * 80)

print()
print("=" * 80)
print("✅ Testing Complete!")
print("=" * 80)
print()
print("If answers look good and cite sources, your fine-tuning was successful! 🎉")
print()
print("Next steps:")
print("  1. Try your own questions")
print("  2. Deploy to production")
print("  3. Export to GGUF for llama.cpp")
print("  4. Add more training data for even better results")
print()

# Optional: Interactive mode
print("Would you like to ask custom questions? (y/n): ", end="")
try:
    response = input().strip().lower()
    
    if response == 'y' or response == 'yes':
        print()
        print("=" * 80)
        print("🎯 Interactive Q&A Mode")
        print("=" * 80)
        print("Type 'quit' or 'exit' to stop")
        print()
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not question:
                continue
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            result = pipe(
                prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_full_text=False
            )
            
            print(f"\nAssistant: {result[0]['generated_text']}\n")
            
except KeyboardInterrupt:
    print("\n\nInterrupted. Exiting...")
except:
    pass

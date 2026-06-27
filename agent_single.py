import os
import json
import re
import argparse
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="AI Agent for Vietnamese Student HackAIthon 2026")
    parser.add_argument("--input_path", type=str, default="public-test_1780368312.json", 
                        help="Path to input json or csv file")
    parser.add_argument("--output_path", type=str, default="pred.csv", 
                        help="Path to save prediction csv")
    parser.add_argument("--model_id", type=str, default="/models/Qwen3.5-7B-Instruct", 
                        help="HuggingFace model ID or local weights path")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Target running device: cuda, cpu, auto")
    parser.add_argument("--load_in_4bit", action="store_true", 
                        help="Load the LLM in 4-bit quantized mode to save VRAM and speed up")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit questions to process for debugging")
    parser.add_argument("--mock", action="store_true", 
                        help="Run in mock mode without loading AI models or PyTorch")
    return parser.parse_args()

def load_dataset(input_path):
    print(f"[*] Loading dataset from: {input_path}")
    if not os.path.exists(input_path):
        # Check current folder or fallback if directory doesn't exist
        dir_name = os.path.dirname(input_path) or "."
        print(f"[!] Warning: {input_path} not found. Checking {dir_name}:")
        if os.path.exists(dir_name):
            print(os.listdir(dir_name))
        for file in os.listdir(dir_name):
            if file.endswith(".json") or file.endswith(".csv"):
                input_path = os.path.join(dir_name, file)
                print(f"[*] Using fallback file: {input_path}")
                break
                
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
        data = []
        for _, row in df.iterrows():
            choices_raw = row.get("choices", "[]")
            if isinstance(choices_raw, str):
                try:
                    choices = json.loads(choices_raw)
                except:
                    import ast
                    try:
                        choices = ast.literal_eval(choices_raw)
                    except:
                        choices = [c.strip() for c in choices_raw.split(",") if c.strip()]
            else:
                choices = list(choices_raw)
            
            data.append({
                "qid": row.get("qid"),
                "question": row.get("question"),
                "choices": choices
            })
    else:
        raise ValueError(f"Unsupported file format for {input_path}! Must be .json or .csv")
    
    print(f"[+] Loaded {len(data)} questions.")
    return data

def build_prompt(question_text, choices):
    # Attempt to load from prompts folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    system_path = os.path.join(base_dir, "prompts", "system_instructions.md")
    rag_path = os.path.join(base_dir, "prompts", "rag_instructions.md")
    
    system_instructions = ""
    rag_instructions = ""
    
    if os.path.exists(system_path):
        with open(system_path, "r", encoding="utf-8") as f:
            system_instructions = f.read()
    else:
        system_instructions = (
            "Bạn là một siêu trợ lý AI chuyên nghiệp giải quyết các câu hỏi trắc nghiệm.\n"
            "Hãy trả lời câu hỏi trắc nghiệm sau bằng cách chọn đáp án đúng nhất từ danh sách các lựa chọn."
        )
        
    if os.path.exists(rag_path):
        with open(rag_path, "r", encoding="utf-8") as f:
            rag_instructions = f.read()
    else:
        rag_instructions = (
            "Yêu cầu:\n"
            "1. Hãy phân tích ngắn gọn và suy luận từng bước (Chain-of-Thought).\n"
            "2. Kết thúc câu trả lời BẮT BUỘC bằng định dạng sau: 'Đáp án cuối cùng: X' (trong đó X là chữ cái in hoa tương ứng với đáp án đúng, ví dụ: Đáp án cuối cùng: A)."
        )

    choices_str = ""
    for idx, choice in enumerate(choices):
        letter = chr(ord('A') + idx)
        choices_str += f"{letter}. {choice}\n"
        
    prompt = (
        f"{system_instructions}\n\n"
        f"--- ĐỀ BÀI ---\nCâu hỏi:\n{question_text}\n\n"
        f"Các lựa chọn:\n{choices_str}\n"
        f"--- HƯỚNG DẪN TRẢ LỜI ---\n{rag_instructions}"
    )
    return prompt

def extract_answer(model_output, num_choices):
    match = re.search(r"Đáp án cuối cùng:\s*([A-K])", model_output, re.IGNORECASE)
    if match:
        ans = match.group(1).upper()
        ans_idx = ord(ans) - ord('A')
        if 0 <= ans_idx < num_choices:
            return ans
            
    matches = re.findall(r"\b([A-K])\b", model_output)
    if matches:
        for ans in reversed(matches):
            ans = ans.upper()
            ans_idx = ord(ans) - ord('A')
            if 0 <= ans_idx < num_choices:
                return ans
                
    return "A"

def main():
    args = parse_args()
    
    # 1. Load dataset
    dataset = load_dataset(args.input_path)
    if args.limit:
        dataset = dataset[:args.limit]
        
    if args.mock:
        print("[*] Running in MOCK mode. Heavy libraries will not be imported.")
        results = []
        for item in dataset:
            qid = item["qid"]
            choices = item["choices"]
            idx = sum(ord(c) for c in str(qid)) % len(choices)
            ans = chr(ord('A') + idx)
            results.append({
                "qid": qid,
                "answer": ans
            })
            
            if len(results) <= 2:
                print(f"\n--- Preview QID: {qid} ---")
                print(f"Answer: {ans}")
                print("-" * 30)
                
        out_dir = os.path.dirname(args.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_out = pd.DataFrame(results)
        df_out.to_csv(args.output_path, index=False)
        print(f"[+] Saved predictions to: {args.output_path}")
        return
        
    # 2. Determine device
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[*] Running on device: {device}")
    
    # 3. Load model and tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    print(f"[*] Loading model from: {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    model_kwargs = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "cpu"
        
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if device == "cuda" else None,
        device=None if device == "cuda" else -1
    )
    
    print("[+] Model loaded successfully!")
    
    # 4. Processing loop
    results = []
    for item in tqdm(dataset, desc="Answering"):
        qid = item["qid"]
        question = item["question"]
        choices = item["choices"]
        num_choices = len(choices)
        
        prompt = build_prompt(question, choices)
        messages = [{"role": "user", "content": prompt}]
        templated_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = gen_pipeline(
            templated_prompt,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response_text = outputs[0]["generated_text"]
        assistant_reply = response_text[len(templated_prompt):].strip()
        answer = extract_answer(assistant_reply, num_choices)
        
        results.append({
            "qid": qid,
            "answer": answer
        })
        
        if len(results) <= 2:
            print(f"\n--- Preview QID: {qid} ---")
            print(f"Prompt: {prompt[:120]}...")
            print(f"Reply: {assistant_reply}")
            print(f"Answer: {answer}")
            print("-" * 30)
            
    # 5. Save results
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out = pd.DataFrame(results)
    df_out.to_csv(args.output_path, index=False)
    print(f"[+] Saved predictions to: {args.output_path}")

if __name__ == "__main__":
    main()

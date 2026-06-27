import os
import json
import re
import argparse
import random
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Official Predict Script for HackAIthon 2026")
    parser.add_argument("--input_path", type=str, default="/code/private_test.json", 
                        help="Path to input json or csv file")
    parser.add_argument("--output_path", type=str, default="submission.csv", 
                        help="Path to save final prediction csv")
    parser.add_argument("--time_output_path", type=str, default="submission_time.csv", 
                        help="Path to save execution time logs")
    parser.add_argument("--log_path", type=str, default="double_predictions.csv", 
                        help="Path to save double run comparison log")
    parser.add_argument("--model_id", type=str, default="/models/Qwen3.5-4B", 
                        help="HuggingFace model ID or local weights path")
    parser.add_argument("--embed_id", type=str, 
                        default="/app/models/bge-m3" if os.path.exists("/app/models/bge-m3") else "BAAI/bge-m3", 
                        help="HuggingFace model ID or local path for Embedding model")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Target running device: cuda, cpu, auto")
    parser.add_argument("--load_in_4bit", action="store_true", 
                        help="Load the LLM in 4-bit quantized mode to save VRAM and speed up")
    parser.add_argument("--top_k", type=int, default=1, 
                        help="Number of chunks to retrieve per question")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Number of questions to group in a single prompt")
    parser.add_argument("--inference_batch_size", type=int, default=4, 
                        help="Number of prompts to batch together for GPU inference")
    parser.add_argument("--limit", type=int, default=None, 
                        help="Limit questions to process for debugging")
    parser.add_argument("--mock", action="store_true", 
                        help="Run in mock mode without loading AI models or PyTorch")
    return parser.parse_args()

def load_dataset(input_path):
    print(f"[*] Loading dataset from: {input_path}")
    if not os.path.exists(input_path):
        dir_name = os.path.dirname(input_path) or "."
        print(f"[!] Warning: {input_path} not found. Checking directory {dir_name}:")
        if os.path.exists(dir_name):
            try:
                print(os.listdir(dir_name))
            except:
                pass
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

def parse_question_chunks(question_text):
    parts = re.split(r"(Câu hỏi:|Question:)", question_text, flags=re.IGNORECASE)
    if len(parts) >= 3:
        context_part = parts[0]
        question_part = parts[1] + parts[2]
    else:
        context_part = question_text
        question_part = question_text
        
    chunk_matches = re.split(r"\[\d+\]", context_part)
    chunks = []
    for c in chunk_matches:
        c_clean = c.strip()
        if len(c_clean) > 25 and not c_clean.startswith("Đoạn thông tin"):
            chunks.append(c_clean)
            
    if len(chunks) <= 1:
        paragraphs = [p.strip() for p in context_part.split("\n\n") if p.strip()]
        chunks = [p for p in paragraphs if len(p) > 25 and not p.startswith("Đoạn thông tin")]
        
    if not chunks:
        chunks = [context_part.strip()]
        
    return chunks, question_part.strip()

def retrieve_top_chunks(embed_model, chunks, query, top_k=1):
    if len(chunks) <= top_k:
        return chunks
    if embed_model is None:
        return chunks[:top_k]
    q_vec = embed_model.encode(query, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True)
    
    scores = []
    for c_vec in chunk_vecs:
        dot_product = np.dot(q_vec, c_vec)
        norm_q = np.linalg.norm(q_vec)
        norm_c = np.linalg.norm(c_vec)
        sim = dot_product / (norm_q * norm_c + 1e-8)
        scores.append(sim)
        
    top_indices = np.argsort(scores)[::-1][:top_k]
    selected_chunks = [chunks[i] for i in sorted(top_indices)]
    return selected_chunks

def load_prompt_file(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "prompts", filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def shuffle_choices(choices):
    indexed_choices = list(enumerate(choices))
    random.seed(len("".join(choices)))
    shuffled_indexed = random.sample(indexed_choices, len(indexed_choices))
    random.seed()
    
    shuffled_texts = [item[1] for item in shuffled_indexed]
    
    letter_mapping = {}
    for shuffled_idx, (original_idx, _) in enumerate(shuffled_indexed):
        shuffled_letter = chr(ord('A') + shuffled_idx)
        original_letter = chr(ord('A') + original_idx)
        letter_mapping[shuffled_letter] = original_letter
        
    return shuffled_texts, letter_mapping

def build_batch_prompt(batch_items, use_shuffled_choices=False):
    prompt = "Hãy trả lời các câu hỏi trắc nghiệm dưới đây. Đọc kỹ phần Ngữ cảnh đi kèm của từng câu hỏi.\n\n"
    
    prompt += (
        "--- VÍ DỤ MINH HỌA ---\n"
        "Đầu vào:\n"
        "=== CÂU HỎI 1 (Mã số: test_9991) ===\n"
        "Ngữ cảnh:\nHà Nội là thủ đô của Việt Nam.\n"
        "Câu hỏi: Thủ đô của Việt Nam là gì?\n"
        "Các lựa chọn:\n"
        "  A. Hà Nội\n"
        "  B. TP. Hồ Chí Minh\n\n"
        "=== CÂU HỎI 2 (Mã số: test_9992) ===\n"
        "Ngữ cảnh:\nTrái Đất quay quanh mặt trời.\n"
        "Câu hỏi: Trái Đất quay quanh cái gì?\n"
        "Các lựa chọn:\n"
        "  A. Mặt Trăng\n"
        "  B. Mặt Trời\n\n"
        "Kết quả đầu ra BẮT BUỘC chỉ được là:\n"
        "test_9991: A\n"
        "test_9992: B\n\n"
        "-----------------------\n\n"
        "Bây giờ hãy trả lời các câu hỏi thực tế sau:\n\n"
    )
    
    for idx, item in enumerate(batch_items):
        qid = item["qid"]
        context_chunks = item["selected_chunks"]
        query = item["query"]
        
        if use_shuffled_choices:
            choices = item["shuffled_choices"]
        else:
            choices = item["choices"]
        
        context_text = "\n".join(f"- {chunk}" for chunk in context_chunks)
        choices_str = ""
        for c_idx, choice in enumerate(choices):
            letter = chr(ord('A') + c_idx)
            choices_str += f"  {letter}. {choice}\n"
            
        prompt += f"=== CÂU HỎI {idx + 1} (Mã số: {qid}) ===\n"
        if context_text.strip():
            prompt += f"Ngữ cảnh:\n{context_text}\n"
        prompt += f"Câu hỏi: {query}\n"
        prompt += f"Các lựa chọn:\n{choices_str}\n\n"
        
    prompt += (
        "--- HƯỚNG DẪN TRẢ LỜI ---\n"
        "Hãy trả lời tất cả các câu hỏi trên.\n"
        "BẮT BUỘC chỉ xuất ra kết quả đáp án cuối cùng theo định dạng dòng chính xác sau (KHÔNG giải thích thêm, KHÔNG phân tích, KHÔNG thêm bất kỳ văn bản nào khác):\n"
    )
    for item in batch_items:
        prompt += f"{item['qid']}: <chữ cái đáp án chọn>\n"
        
    return prompt

def parse_batch_response(response_text, batch_items):
    parsed_answers = {}
    for item in batch_items:
        qid = item["qid"]
        num_choices = len(item["choices"])
        
        pattern = re.escape(qid) + r"\s*[:\-=\s]\s*([A-K])\b"
        match = re.search(pattern, response_text, re.IGNORECASE)
        
        if match:
            ans = match.group(1).upper()
            ans_idx = ord(ans) - ord('A')
            if 0 <= ans_idx < num_choices:
                parsed_answers[qid] = ans
                
    return parsed_answers

def mock_choose_answer(qid, choices):
    idx = sum(ord(c) for c in str(qid)) % len(choices)
    return chr(ord('A') + idx)

def main():
    args = parse_args()
    
    # Clean output files
    for p in [args.output_path, args.time_output_path, args.log_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass
                
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    # Dict to log total inference time per sample (qid -> seconds)
    inference_times = {}
    
    # 1. Load data
    dataset = load_dataset(args.input_path)
    if args.limit:
        dataset = dataset[:args.limit]
        
    inference_batch_size = args.inference_batch_size
        
    system_instructions = load_prompt_file("system_instructions.md")
    if not system_instructions:
        system_instructions = "Bạn là một siêu trợ lý AI chuyên nghiệp giải quyết các câu hỏi trắc nghiệm."
    system_instructions += "\nBẮT BUỘC CHỈ ĐƯỢC XUẤT RA ĐÁP ÁN THEO ĐÚNG ĐỊNH DẠNG YÊU CẦU. KHÔNG GIẢI THÍCH, KHÔNG PHÂN TÍCH, KHÔNG CHÀO HỎI, KHÔNG ĐƯỢC THÊM BẤT KỲ TỪ NGỮ NÀO KHÁC."
        
    if args.mock:
        print("[*] Running in MOCK mode. Heavy libraries will not be imported.")
        device = "cpu"
        embed_model = None
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Running on device: {device}")
        
        # 2. Load Embedding Model
        print(f"[*] Loading Embedding Model: {args.embed_id} ...")
        embed_model = SentenceTransformer(args.embed_id, device=device)
    
    # Pre-process RAG chunks and shuffle mappings
    print("[*] Performing pre-retrieval context pruning and option shuffling mappings...")
    processed_items = []
    for item in tqdm(dataset, desc="Pruning contexts"):
        qid = item["qid"]
        raw_question = item["question"]
        choices = item["choices"]
        
        # Measure Embedding (Retrieval) time for this sample
        start_embed = time.time()
        chunks, query = parse_question_chunks(raw_question)
        selected_chunks = retrieve_top_chunks(embed_model, chunks, query, top_k=args.top_k)
        end_embed = time.time()
        
        shuffled_choices, letter_mapping = shuffle_choices(choices)
        
        processed_items.append({
            "qid": qid,
            "query": query,
            "choices": choices,
            "shuffled_choices": shuffled_choices,
            "letter_mapping": letter_mapping,
            "selected_chunks": selected_chunks
        })
        
        inference_times[qid] = end_embed - start_embed
        
    # 3. Load LLM Model
    if args.mock:
        tokenizer = None
        gen_pipeline = None
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        # Resolve the model path dynamically
        model_id = args.model_id
        if not os.path.exists(model_id) and not model_id.startswith("Qwen/"):
            parent_dir = os.path.dirname(model_id) or "/models"
            if os.path.exists(parent_dir):
                subdirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
                if subdirs:
                    qwen_dirs = [d for d in subdirs if "qwen" in d.lower()]
                    model_id = qwen_dirs[0] if qwen_dirs else subdirs[0]
                    print(f"[!] Path {args.model_id} not found. Dynamic fallback to: {model_id}")
            
        print(f"[*] Loading LLM Model: {model_id} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
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
            
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        
        # Dynamic VRAM-based batch size tuning
        if device == "cuda":
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                vram_gb = total_memory / (1024 ** 3)
                if args.inference_batch_size == 8:
                    if vram_gb < 15:
                        inference_batch_size = 4
                    elif vram_gb < 23:
                        inference_batch_size = 8
                    elif vram_gb < 39:
                        inference_batch_size = 16
                    else:
                        inference_batch_size = 32
                    print(f"[*] Auto-adjusted inference_batch_size to {inference_batch_size} based on {vram_gb:.2f} GB VRAM.")
            except Exception as e:
                print(f"[!] GPU VRAM detection failed: {e}. Using default batch size.")
                
        gen_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto" if device == "cuda" else None,
            device=None if device == "cuda" else -1,
            batch_size=inference_batch_size
        )
    
    print("[+] Models loaded successfully! Starting PASS 1 double-run validation...")
    
    resolved_answers = {}
    detailed_logs = []
    conflicted_items = []
    
    # 4. PASS 1: Processing in Batches of 4
    batch_size = args.batch_size
    num_batches = (len(processed_items) + batch_size - 1) // batch_size
    
    all_batches = []
    for i in range(num_batches):
        all_batches.append(processed_items[i * batch_size : (i + 1) * batch_size])
        
    num_inference_steps = (len(all_batches) + inference_batch_size - 1) // inference_batch_size
    
    first_batch_reply_1 = ""
    first_batch_reply_2 = ""
    
    for step in tqdm(range(num_inference_steps), desc="Processing PASS 1"):
        step_batches = all_batches[step * inference_batch_size : (step + 1) * inference_batch_size]
        
        # Flattened question list for this batch step to distribute generation time
        qids_in_step = []
        for batch_items in step_batches:
            for item in batch_items:
                qids_in_step.append(item["qid"])
        
        if not args.mock:
            # Measure generation time for this step
            start_gen_step = time.time()
            
            # --- RUN 1 (Original Choices) ---
            templated_prompts_1 = []
            first_qids = []
            
            for batch_items in step_batches:
                prompt_1 = build_batch_prompt(batch_items, use_shuffled_choices=False)
                messages_1 = [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": prompt_1}
                ]
                templated_1 = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
                first_qid = batch_items[0]["qid"]
                templated_1 += f"{first_qid}:"
                
                templated_prompts_1.append(templated_1)
                first_qids.append(first_qid)
                
            outputs_1 = gen_pipeline(
                templated_prompts_1,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                batch_size=len(templated_prompts_1)
            )
            
            # --- RUN 2 (Shuffled Choices) ---
            templated_prompts_2 = []
            for batch_items in step_batches:
                prompt_2 = build_batch_prompt(batch_items, use_shuffled_choices=True)
                messages_2 = [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": prompt_2}
                ]
                templated_2 = tokenizer.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)
                first_qid = batch_items[0]["qid"]
                templated_2 += f"{first_qid}:"
                
                templated_prompts_2.append(templated_2)
                
            outputs_2 = gen_pipeline(
                templated_prompts_2,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                batch_size=len(templated_prompts_2)
            )
            
            end_gen_step = time.time()
            generation_time_step = end_gen_step - start_gen_step
            time_per_question = generation_time_step / len(qids_in_step)
            for qid in qids_in_step:
                inference_times[qid] += time_per_question
                
            # --- PARSE AND EVALUATE EACH BATCH ---
            for idx, batch_items in enumerate(step_batches):
                first_qid = first_qids[idx]
                templated_1 = templated_prompts_1[idx]
                templated_2 = templated_prompts_2[idx]
                
                reply_1 = f"{first_qid}:" + outputs_1[idx][0]["generated_text"][len(templated_1):].strip()
                answers_run_1 = parse_batch_response(reply_1, batch_items)
                
                reply_2 = f"{first_qid}:" + outputs_2[idx][0]["generated_text"][len(templated_2):].strip()
                shuffled_answers_run_2 = parse_batch_response(reply_2, batch_items)
                
                if step == 0 and idx == 0:
                    first_batch_reply_1 = reply_1
                    first_batch_reply_2 = reply_2
                    
                # Map Run 2 answers back to original choices
                answers_run_2 = {}
                for qid, shuffled_ans in shuffled_answers_run_2.items():
                    item = next(it for it in batch_items if it["qid"] == qid)
                    mapping = item["letter_mapping"]
                    if shuffled_ans in mapping:
                        answers_run_2[qid] = mapping[shuffled_ans]
                        
                for item in batch_items:
                    qid = item["qid"]
                    ans1 = answers_run_1.get(qid, None)
                    ans2 = answers_run_2.get(qid, None)
                    
                    if ans1 and ans2 and ans1 == ans2:
                        resolved_answers[qid] = ans1
                        detailed_logs.append({
                            "qid": qid,
                            "ans1_p1": ans1,
                            "ans2_p1": ans2,
                            "ans1_p2": "N/A",
                            "ans2_p2": "N/A",
                            "final_ans": ans1,
                            "status": "MATCH"
                        })
                    else:
                        item["ans1_p1"] = ans1
                        item["ans2_p1"] = ans2
                        conflicted_items.append(item)
        else:
            # --- MOCK MODE ---
            # Simulate a realistic GPU generation time per step (approx 0.8s to 1.5s per question)
            simulated_gen_time = sum(random.uniform(0.6, 1.1) for _ in qids_in_step)
            for qid in qids_in_step:
                inference_times[qid] += (simulated_gen_time / len(qids_in_step))
                
            for idx, batch_items in enumerate(step_batches):
                reply_1 = ""
                for item in batch_items:
                    ans1_letter = mock_choose_answer(item["qid"], item["choices"])
                    reply_1 += f"{item['qid']}: {ans1_letter}\n"
                answers_run_1 = parse_batch_response(reply_1, batch_items)
                
                reply_2 = ""
                for item in batch_items:
                    ans1_letter = mock_choose_answer(item["qid"], item["choices"])
                    qid_sum = sum(ord(c) for c in str(item["qid"]))
                    if qid_sum % 2 == 0:
                        ans2_letter = ans1_letter
                    else:
                        ans2_letter = chr(ord('A') + (ord(ans1_letter) - ord('A') + 1) % len(item["choices"]))
                    
                    shuffled_letter = next(k for k, v in item["letter_mapping"].items() if v == ans2_letter)
                    reply_2 += f"{item['qid']}: {shuffled_letter}\n"
                shuffled_answers_run_2 = parse_batch_response(reply_2, batch_items)
                
                if step == 0 and idx == 0:
                    first_batch_reply_1 = reply_1
                    first_batch_reply_2 = reply_2
                    
                # Map Run 2 answers back to original choices
                answers_run_2 = {}
                for qid, shuffled_ans in shuffled_answers_run_2.items():
                    item = next(it for it in batch_items if it["qid"] == qid)
                    mapping = item["letter_mapping"]
                    if shuffled_ans in mapping:
                        answers_run_2[qid] = mapping[shuffled_ans]
                        
                for item in batch_items:
                    qid = item["qid"]
                    ans1 = answers_run_1.get(qid, None)
                    ans2 = answers_run_2.get(qid, None)
                    
                    if ans1 and ans2 and ans1 == ans2:
                        resolved_answers[qid] = ans1
                        detailed_logs.append({
                            "qid": qid,
                            "ans1_p1": ans1,
                            "ans2_p1": ans2,
                            "ans1_p2": "N/A",
                            "ans2_p2": "N/A",
                            "final_ans": ans1,
                            "status": "MATCH"
                        })
                    else:
                        item["ans1_p1"] = ans1
                        item["ans2_p1"] = ans2
                        conflicted_items.append(item)
                        
    # Preview first batch results
    print("\n--- PASS 1 Preview Batch 1 Run 1 Reply ---")
    print(first_batch_reply_1)
    print("--- PASS 1 Preview Batch 1 Run 2 (Shuffled) Reply ---")
    print(first_batch_reply_2)
    print("-" * 40)
            
    # 5. PASS 2: Retry conflicting items in batches of 4
    if conflicted_items:
        print(f"\n[!] PASS 1 finished. Found {len(conflicted_items)} conflicted questions. Starting PASS 2 retries...")
        
        p2_batches = []
        num_batches_p2 = (len(conflicted_items) + batch_size - 1) // batch_size
        for i in range(num_batches_p2):
            p2_batches.append(conflicted_items[i * batch_size : (i + 1) * batch_size])
            
        num_inference_steps_p2 = (len(p2_batches) + inference_batch_size - 1) // inference_batch_size
        
        for step in tqdm(range(num_inference_steps_p2), desc="Processing PASS 2"):
            step_batches = p2_batches[step * inference_batch_size : (step + 1) * inference_batch_size]
            
            qids_in_step_p2 = []
            for batch_items in step_batches:
                for item in batch_items:
                    qids_in_step_p2.append(item["qid"])
                    
            if not args.mock:
                start_gen_step_p2 = time.time()
                
                # --- RUN 1 (Original Choices) ---
                templated_prompts_1 = []
                first_qids = []
                for batch_items in step_batches:
                    prompt_1 = build_batch_prompt(batch_items, use_shuffled_choices=False)
                    messages_1 = [
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": prompt_1}
                    ]
                    templated_1 = tokenizer.apply_chat_template(messages_1, tokenize=False, add_generation_prompt=True)
                    first_qid = batch_items[0]["qid"]
                    templated_1 += f"{first_qid}:"
                    
                    templated_prompts_1.append(templated_1)
                    first_qids.append(first_qid)
                    
                outputs_1 = gen_pipeline(
                    templated_prompts_1,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    batch_size=len(templated_prompts_1)
                )
                
                # --- RUN 2 (Shuffled Choices) ---
                templated_prompts_2 = []
                for batch_items in step_batches:
                    prompt_2 = build_batch_prompt(batch_items, use_shuffled_choices=True)
                    messages_2 = [
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": prompt_2}
                    ]
                    templated_2 = tokenizer.apply_chat_template(messages_2, tokenize=False, add_generation_prompt=True)
                    first_qid = batch_items[0]["qid"]
                    templated_2 += f"{first_qid}:"
                    
                    templated_prompts_2.append(templated_2)
                    
                outputs_2 = gen_pipeline(
                    templated_prompts_2,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    batch_size=len(templated_prompts_2)
                )
                
                end_gen_step_p2 = time.time()
                generation_time_step_p2 = end_gen_step_p2 - start_gen_step_p2
                time_per_question_p2 = generation_time_step_p2 / len(qids_in_step_p2)
                for qid in qids_in_step_p2:
                    inference_times[qid] += time_per_question_p2
                
                # --- PARSE AND EVALUATE EACH BATCH ---
                for idx, batch_items in enumerate(step_batches):
                    first_qid = first_qids[idx]
                    templated_1 = templated_prompts_1[idx]
                    templated_2 = templated_prompts_2[idx]
                    
                    reply_1 = f"{first_qid}:" + outputs_1[idx][0]["generated_text"][len(templated_1):].strip()
                    answers_run_1 = parse_batch_response(reply_1, batch_items)
                    
                    reply_2 = f"{first_qid}:" + outputs_2[idx][0]["generated_text"][len(templated_2):].strip()
                    shuffled_answers_run_2 = parse_batch_response(reply_2, batch_items)
                    
                    # Map Run 2 answers back to original choices
                    answers_run_2 = {}
                    for qid, shuffled_ans in shuffled_answers_run_2.items():
                        item = next(it for it in batch_items if it["qid"] == qid)
                        mapping = item["letter_mapping"]
                        if shuffled_ans in mapping:
                            answers_run_2[qid] = mapping[shuffled_ans]
                            
                    # Pass 2 Evaluation
                    for item in batch_items:
                        qid = item["qid"]
                        ans1_p1 = item["ans1_p1"]
                        ans2_p1 = item["ans2_p1"]
                        ans1_p2 = answers_run_1.get(qid, None)
                        ans2_p2 = answers_run_2.get(qid, None)
                        
                        # Gather all valid votes
                        votes = [v for v in [ans1_p1, ans2_p1, ans1_p2, ans2_p2] if v is not None]
                        if votes:
                            from collections import Counter
                            counts = Counter(votes)
                            most_common = counts.most_common()
                            max_votes = most_common[0][1]
                            candidates = [cand for cand, cnt in most_common if cnt == max_votes]
                            
                            # Resolve with tie-breaking
                            if len(candidates) == 1:
                                final_ans = candidates[0]
                                status = "RESOLVED_P2"
                            else:
                                if ans1_p1 in candidates:
                                    final_ans = ans1_p1
                                elif ans1_p2 in candidates:
                                    final_ans = ans1_p2
                                else:
                                    final_ans = candidates[0]
                                status = "RESOLVED_TIE"
                        else:
                            final_ans = "A"
                            status = "NO_ANS"
                            
                        resolved_answers[qid] = final_ans
                        detailed_logs.append({
                            "qid": qid,
                            "ans1_p1": ans1_p1 or "None",
                            "ans2_p1": ans2_p1 or "None",
                            "ans1_p2": ans1_p2 or "None",
                            "ans2_p2": ans2_p2 or "None",
                            "final_ans": final_ans,
                            "status": status
                        })
            else:
                # --- MOCK MODE ---
                simulated_gen_time_p2 = sum(random.uniform(0.6, 1.1) for _ in qids_in_step_p2)
                for qid in qids_in_step_p2:
                    inference_times[qid] += (simulated_gen_time_p2 / len(qids_in_step_p2))
                    
                for idx, batch_items in enumerate(step_batches):
                    # We simulate Pass 2 Run 1 & Run 2
                    reply_1 = ""
                    for item in batch_items:
                        ans1_letter = mock_choose_answer(item["qid"], item["choices"])
                        reply_1 += f"{item['qid']}: {ans1_letter}\n"
                    answers_run_1 = parse_batch_response(reply_1, batch_items)
                    
                    reply_2 = ""
                    for item in batch_items:
                        ans2_letter = mock_choose_answer(item["qid"], item["choices"])
                        shuffled_letter = next(k for k, v in item["letter_mapping"].items() if v == ans2_letter)
                        reply_2 += f"{item['qid']}: {shuffled_letter}\n"
                    shuffled_answers_run_2 = parse_batch_response(reply_2, batch_items)
                    
                    # Map Run 2 answers back to original choices
                    answers_run_2 = {}
                    for qid, shuffled_ans in shuffled_answers_run_2.items():
                        item = next(it for it in batch_items if it["qid"] == qid)
                        mapping = item["letter_mapping"]
                        if shuffled_ans in mapping:
                            answers_run_2[qid] = mapping[shuffled_ans]
                            
                    for item in batch_items:
                        qid = item["qid"]
                        ans1_p1 = item["ans1_p1"]
                        ans2_p1 = item["ans2_p1"]
                        ans1_p2 = answers_run_1.get(qid, None)
                        ans2_p2 = answers_run_2.get(qid, None)
                        
                        votes = [v for v in [ans1_p1, ans2_p1, ans1_p2, ans2_p2] if v is not None]
                        if votes:
                            from collections import Counter
                            counts = Counter(votes)
                            most_common = counts.most_common()
                            max_votes = most_common[0][1]
                            candidates = [cand for cand, cnt in most_common if cnt == max_votes]
                            
                            if len(candidates) == 1:
                                final_ans = candidates[0]
                                status = "RESOLVED_P2"
                            else:
                                if ans1_p1 in candidates:
                                    final_ans = ans1_p1
                                elif ans1_p2 in candidates:
                                    final_ans = ans1_p2
                                else:
                                    final_ans = candidates[0]
                                status = "RESOLVED_TIE"
                        else:
                            final_ans = "A"
                            status = "NO_ANS"
                            
                        resolved_answers[qid] = final_ans
                        detailed_logs.append({
                            "qid": qid,
                            "ans1_p1": ans1_p1 or "None",
                            "ans2_p1": ans2_p1 or "None",
                            "ans1_p2": ans1_p2 or "None",
                            "ans2_p2": ans2_p2 or "None",
                            "final_ans": final_ans,
                            "status": status
                        })
                
    # 6. WRITE FINAL LOGS AND PREDICTIONS
    print(f"\n[*] Writing final comparison logs to: {args.log_path}")
    df_logs = pd.DataFrame(detailed_logs)
    cols_order = ["qid", "ans1_p1", "ans2_p1", "ans1_p2", "ans2_p2", "final_ans", "status"]
    df_logs = df_logs.reindex(columns=cols_order)
    df_logs.to_csv(args.log_path, index=False)
    
    print(f"[*] Writing final predictions to: {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write("qid,answer\n")
        for item in dataset:
            qid = item["qid"]
            ans = resolved_answers.get(qid, "A")
            f.write(f"{qid},{ans}\n")
            
    print(f"[*] Writing final execution times to: {args.time_output_path}")
    with open(args.time_output_path, "w", encoding="utf-8") as f:
        f.write("qid,answer,time\n")
        for item in dataset:
            qid = item["qid"]
            ans = resolved_answers.get(qid, "A")
            t = inference_times.get(qid, 1.0)
            f.write(f"{qid},{ans},{t:.4f}\n")
            
    print(f"[+] Process complete. Saved {len(dataset)} predictions.")

if __name__ == "__main__":
    main()

import os
import json
import pandas as pd

def load_dataset(input_path):
    print(f"[*] Loading dataset from: {input_path}")
    if not os.path.exists(input_path):
        # Fallback to current folder list
        dir_name = os.path.dirname(input_path) or "."
        print(f"[!] Warning: {input_path} not found. Searching directory: {dir_name}")
        if os.path.exists(dir_name):
            print(os.listdir(dir_name))
        for file in os.listdir(dir_name):
            if file.endswith(".json") or file.endswith(".csv"):
                input_path = os.path.join(dir_name, file)
                print(f"[*] Found fallback input file: {input_path}")
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

def save_predictions(results, output_path):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"[+] Saved {len(results)} predictions to: {output_path}")

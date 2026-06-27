import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_llm_pipeline(model_id, device="auto", load_in_4bit=False):
    print(f"[*] Loading tokenizer for: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if device == "auto":
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_type = device
        
    print(f"[*] Target device: {device_type} (Quantization 4-bit: {load_in_4bit})")
    
    model_kwargs = {
        "trust_remote_code": True
    }
    
    if device_type == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
        if load_in_4bit:
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
        
    print(f"[*] Loading model weights from: {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if device_type == "cuda" else None,
        device=None if device_type == "cuda" else -1
    )
    
    print("[+] LLM loaded successfully!")
    return gen_pipeline, tokenizer

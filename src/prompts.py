import os
import re

def load_prompt_file(filename):
    """
    Utility to load instruction markdown files dynamically from the prompts folder.
    """
    # Find prompts directory relative to the current file (src/prompts.py -> root/prompts)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "prompts", filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def build_prompt(context_chunks, question_query, choices):
    # Load markdown system and reasoning instructions
    system_instructions = load_prompt_file("system_instructions.md")
    rag_instructions = load_prompt_file("rag_instructions.md")
    
    # Fallback default instructions if file load fails
    if not system_instructions:
        system_instructions = "Bạn là một trợ lý AI thông minh giải quyết các câu hỏi trắc nghiệm."
    if not rag_instructions:
        rag_instructions = "Hãy lập luận từng bước (Chain-of-Thought) và kết thúc bằng 'Đáp án cuối cùng: X'."

    # Format selected context chunks
    context_text = "\n\n".join(f"[Thông tin] {chunk}" for chunk in context_chunks)
    
    # Format options
    choices_str = ""
    for idx, choice in enumerate(choices):
        letter = chr(ord('A') + idx)
        choices_str += f"{letter}. {choice}\n"
        
    prompt = (
        f"{system_instructions}\n\n"
        f"--- NGỮ CẢNH ---\n{context_text}\n\n"
        f"--- ĐỀ BÀI ---\nCâu hỏi: {question_query}\n\n"
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

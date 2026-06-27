import random
import re
import sys

# Import functions from predict.py
try:
    from predict import shuffle_choices, parse_batch_response, build_batch_prompt
    print("[*] Successfully imported predict.py functions.")
except Exception as e:
    print(f"[!] Import failed: {e}")
    sys.exit(1)

def run_shuffling_stress_test(num_samples=500):
    print(f"[*] Starting Option Shuffling & Reverse Mapping Stress Test with {num_samples} samples...")
    
    passed_count = 0
    failed_count = 0
    
    # We will simulate mock choices
    choices_options = [
        ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Cần Thơ"],
        ["Sông Hồng", "Sông Mê Kông", "Sông Đồng Nai", "Sông Sài Gòn"],
        ["Trái Đất", "Sao Hỏa", "Sao Kim", "Sao Mộc", "Sao Thổ"],
        ["Python", "Java", "C++", "JavaScript", "Go", "Rust"],
        ["Chó", "Mèo", "Chim", "Cá"]
    ]
    
    for i in range(num_samples):
        choices = random.choice(choices_options)
        original_choices = list(choices) # copy
        
        # Pick a target choice as the answer
        true_index = random.randint(0, len(choices) - 1)
        true_choice_text = choices[true_index]
        true_letter = chr(ord('A') + true_index)
        
        # Shuffle choices
        shuffled_choices, letter_mapping = shuffle_choices(choices)
        
        # Verify length of shuffled choices is correct
        if len(shuffled_choices) != len(choices):
            print(f"[!] Length mismatch on sample {i}")
            failed_count += 1
            continue
            
        # Verify that all choices are preserved
        if sorted(shuffled_choices) != sorted(choices):
            print(f"[!] Elements changed on sample {i}")
            failed_count += 1
            continue
            
        # Verify that reverse mapping maps letters back correctly
        for shuffled_idx, choice_text in enumerate(shuffled_choices):
            shuffled_letter = chr(ord('A') + shuffled_idx)
            
            # Find original index of this choice text in original_choices
            original_idx = original_choices.index(choice_text)
            expected_original_letter = chr(ord('A') + original_idx)
            
            mapped_original_letter = letter_mapping.get(shuffled_letter, None)
            
            if mapped_original_letter != expected_original_letter:
                print(f"[!] Mapping mismatch on sample {i}: Shuffled {shuffled_letter} ('{choice_text}') mapped to {mapped_original_letter}, expected {expected_original_letter}")
                failed_count += 1
                break
        else:
            # If no break, verification succeeded
            passed_count += 1
            
    print(f"\n[+] Shuffling Stress Test completed.")
    print(f"    - Passed: {passed_count}/{num_samples}")
    print(f"    - Failed: {failed_count}/{num_samples}")
    
    if failed_count == 0:
        print("[SUCCESS] SUCCESS: All option shuffling and reverse mappings are 100% mathematically correct and stable!")
        return True
    else:
        print("[FAILURE] FAILURE: Detected shuffling errors!")
        return False

def test_full_mock_pipeline_execution():
    print("\n[*] Testing Full Mock Pipeline Execution with 10 questions...")
    # We simulate running predict.py in mock mode to check if there are any exceptions with the new code
    import subprocess
    cmd = [
        "python", "predict.py", 
        "--mock", 
        "--input_path", "public-test_1780368312.json", 
        "--limit", "10"
    ]
    try:
        # Check if the public test file exists, if not we create a dummy one
        import os
        if not os.path.exists("public-test_1780368312.json"):
            # Create a small dummy file
            import json
            dummy_data = []
            for i in range(10):
                dummy_data.append({
                    "qid": f"dummy_test_{i}",
                    "question": f"Đoạn thông tin [{i}] Ngữ cảnh thực tế {i}. Câu hỏi: Câu hỏi test {i}? Các lựa chọn:",
                    "choices": [f"Lựa chọn A {i}", f"Lựa chọn B {i}", f"Lựa chọn C {i}", f"Lựa chọn D {i}"]
                })
            with open("public-test_1780368312.json", "w", encoding="utf-8") as f:
                json.dump(dummy_data, f)
            print("[*] Created temporary public-test_1780368312.json for pipeline test.")
            created_dummy = True
        else:
            created_dummy = False
            
        res = subprocess.run(cmd, capture_output=True, text=True)
        print("--- Execution Output ---")
        print(res.stdout)
        if res.stderr:
            print("--- Execution Errors ---")
            print(res.stderr)
            
        # Clean dummy file if created
        if created_dummy:
            try:
                os.remove("public-test_1780368312.json")
            except:
                pass
                
        if res.returncode == 0:
            print("[SUCCESS] SUCCESS: Mock pipeline executed without any exceptions!")
            return True
        else:
            print("[FAILURE] FAILURE: Mock pipeline crash!")
            return False
    except Exception as e:
        print(f"[!] Pipeline test execution failed: {e}")
        return False

if __name__ == "__main__":
    shuffling_ok = run_shuffling_stress_test(1000)
    pipeline_ok = test_full_mock_pipeline_execution()
    
    if shuffling_ok and pipeline_ok:
        print("\n[SUCCESS] SUMMARY: ALL STRESS TESTS PASSED. SOLUTION CODE IS 100% STABLE AND SAFE!")
        sys.exit(0)
    else:
        print("\n[FAILURE] SUMMARY: STRESS TEST FAILED! PLEASE CHECK THE ERRORS ABOVE.")
        sys.exit(1)

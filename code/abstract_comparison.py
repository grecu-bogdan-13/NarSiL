import json
import torch
import os
import re
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

class AbstractComparator:
    def __init__(self, model_path, hf_token=None):
        self.model_path = model_path
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        self.system_prompt = """You are an expert literary critic.
You are given the abstract theme of an anchor story, and the abstract theme of other two stories, Story A and Story B.

Task: Identify which of the two candidates (A or B) shares a closer Abstract Theme with the Anchor.
Constraint: Ignore similarity in setting (e.g., space, medieval) or specific plot events. Focus ONLY on the central moral, philosophical question, or character arc type.

Reasoning: First, compare Anchor vs A. Then, compare Anchor vs B. Finally, output the winner.
After determining the winner, classify your confidence in this decision as:
- High: The thematic overlap is obvious and distinct.
- Medium: The winner is closer, but the distinction is subtle.
- Low: Both are equally similar or dissimilar; the choice is a guess.

On the last line format your final answer exactly like this: {final_answer: , confidence: }"""

    def _load_model(self):
        if self.model is None:
            print(f"[Comparison] Loading model: {self.model_path}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, token=self.hf_token)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    token=self.hf_token,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            except Exception as e:
                print(f"Error loading model: {e}")
                sys.exit(1)

    def _extract_json_result(self, text):
        """
        Parses the last line for {final_answer: ..., confidence: ...}
        """
        # print(text)
        if not text:
            return None, None

        pattern = r"\{.*?final_answer\s*:\s*[\"']?(Story [AB])[\"']?.*?,.*?confidence\s*:\s*[\"']?(High|Medium|Low)[\"']?.*?\}"
        
        matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
        
        if matches:
            last_match = matches[-1]
            return last_match.group(1).title(), last_match.group(2).title() 
        
        last_line = text.strip().split('\n')[-1]
        
        pred = None
        conf = "Low" 
        
        if "Text A" in last_line: pred = "Text A"
        elif "Text B" in last_line: pred = "Text B"
        
        if "High" in last_line: conf = "High"
        elif "Medium" in last_line: conf = "Medium"
        
        return pred, conf

    def _compare_abstracts(self, anchor_abs, a_abs, b_abs):
        user_msg = (
            f"Anchor Abstract: {anchor_abs}\n\n"
            f"Story A Abstract: {a_abs}\n\n"
            f"Story B Abstract: {b_abs}\n\n"
            "Comparision:"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg}
        ]
        
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([prompt_str], return_tensors="pt", padding=True, return_attention_mask=True).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512, 
                do_sample=False, 
                temperature=None,
                top_p=None
            )

        gen_ids_trimmed = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        decoded_output = self.tokenizer.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]
        return decoded_output

    def process_files(self, file_paths):
        self._load_model()

        for filepath in file_paths:
            if not os.path.exists(filepath):
                print(f"Skipping {filepath}, not found.")
                continue

            directory, filename = os.path.split(filepath)
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_output{ext}"
            output_path = os.path.join(directory, new_filename)

            print(f"Comparing abstracts in {filename} -> {new_filename}...")

            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))

            output_data = []
            count_processed = 0

            for entry in tqdm(data, desc="Comparing"):
                if 'final_answer' not in entry and 'anchor_text_abs' in entry:
                    
                    raw_response = self._compare_abstracts(
                        entry.get('anchor_text_abs', ''),
                        entry.get('text_a_abs', ''),
                        entry.get('text_b_abs', '')
                    )
                    
                    decision, confidence = self._extract_json_result(raw_response)
                    
                    if decision:
                        entry['closer_abstract'] = decision
                        entry['confidence_abstract'] = confidence
                    else:
                        entry['closer_abstract'] = "Error"
                        entry['confidence_abstract'] = "None"
                        print(f"\nWarning: Could not parse response for entry.")

                    
                    count_processed += 1
                
                output_data.append(entry)

            with open(output_path, 'w', encoding='utf-8') as f:
                for item in output_data:
                    f.write(json.dumps(item) + "\n")

            print(f"Saved {len(output_data)} entries. (Abstract Comparisons performed: {count_processed})")
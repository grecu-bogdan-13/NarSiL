import json
import torch
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

class NarrativeAbstractor:
    def __init__(self, model_path, hf_token=None):
        self.model_path = model_path
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        self.system_prompt = """Analyze the story provided.

1. First, list the Protagonist's Goal and the Primary Obstacle.
2. Second, summarize the Resolution.
3. Finally, write an Abstract Theme Statement (1 sentence) based on the above. 

Constraints for the Theme Statement:
- Describe the central human problem and the moral realization.
- Strictly avoid proper names, specific locations, or unique terminology (e.g., 'John' -> 'a man').

Format your response exactly as follows:
Analysis: [Your analysis here]
Theme: [Your single sentence abstract here]"""

    def _load_model(self):
        """Loads the model only when processing starts to save resources."""
        if self.model is None:
            print(f"[Abstraction] Loading model: {self.model_path}...")
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

    def _get_abstract(self, text):
        if not text:
            return ""

        story_text = text[:2000]
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"STORY: {story_text}\n\nABSTRACT:"}
        ]
        
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([prompt_str], return_tensors="pt", padding=True, return_attention_mask=True).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=256,
                do_sample=False, 
            )

        gen_ids_trimmed = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        decoded_output = self.tokenizer.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]
        if "Theme:" in decoded_output:
            return decoded_output.split("Theme:")[-1].strip()
        else:
            return decoded_output.strip()
        return decoded_output.strip()

    def process_files(self, file_paths):
        self._load_model()
        
        for filepath in file_paths:
            if not os.path.exists(filepath):
                print(f"Skipping {filepath}, not found.")
                continue
                
            directory, filename = os.path.split(filepath)
            name, ext = os.path.splitext(filename)
            clean_name = name.replace('_output', '')
            new_filename = f"{clean_name}_abstract{ext}" 
            output_path = os.path.join(directory, new_filename)
            
            print(f"Abstracting {filename} -> {new_filename}...")
            
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            
            output_data = []
            count_abstracted = 0
            
            for entry in tqdm(data, desc=f"Processing entries"):
                if 'final_answer' in entry:
                    output_data.append(entry)
                else:
                    try:
                        abs_anchor = self._get_abstract(entry.get('anchor_text', ''))
                        abs_a = self._get_abstract(entry.get('text_a', ''))
                        abs_b = self._get_abstract(entry.get('text_b', ''))
                        
                        entry['anchor_text_abs'] = abs_anchor
                        entry['text_a_abs'] = abs_a
                        entry['text_b_abs'] = abs_b
                        
                        output_data.append(entry)
                        count_abstracted += 1
                        
                    except Exception as e:
                        print(f"Error processing entry: {e}")
                        output_data.append(entry)

            with open(output_path, 'w', encoding='utf-8') as f:
                for item in output_data:
                    f.write(json.dumps(item) + "\n")
            
            print(f"Saved {len(output_data)} entries. (Abstracted: {count_abstracted})")
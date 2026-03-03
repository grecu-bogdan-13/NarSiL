import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from tqdm import tqdm
import re

class NarrativeSimilarityClassifier:
    def __init__(self, model_id, access_token=None):
        """
        Initializes Gemma 3 for structural comparison of event sequences.
        """
        print(f"[Classification] Loading model: {model_id}...")
        
        self.device = (
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=access_token,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        )
        self.model.eval()

    def _get_decision(self, anchor, story_a, story_b):
        """
        Directly prompts the model to compare narrative structures based on AMR/SemLink events.
        """
        system_prompt = (
            "You are an expert in computational narratology. "
            "Your task is to identify which of two candidate stories (Story A or Story B) "
            "is structurally closer to an Anchor Story based on the 'Course of Action'.\n\n"
            "CRITERIA:\n"
            "1. Focus on the sequence of predicates (e.g., 'run-01') and frame tags.\n"
            "2. Focus on polarity (e.g., NOT_ vs positive) and role alignment (ARG0, ARG1).\n"
            "3. Analyze the narrative arc (e.g., MOTION -> POSSESSION -> CHANGE).\n\n"
            "OUTPUT FORMAT:\n"
            "You must return ONLY a JSON object with keys: 'analysis', 'closer_story' (A or B), "
            "and 'confidence' (High/Medium/Low)."
        )

        user_input = f"""
        ANCHOR STORY EVENTS:
        {json.dumps(anchor)}

        STORY A EVENTS:
        {json.dumps(story_a)}

        STORY B EVENTS:
        {json.dumps(story_b)}
        """

        chat = [{"role": "user", "content": system_prompt + "\n\n" + user_input}]
        
        prompt_tokens = self.tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                prompt_tokens,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][prompt_tokens.shape[1]:], skip_special_tokens=True)
        
        try:
            clean_res = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
            return json.loads(clean_res)
        except Exception as e:
            return {
                "closer_story": "A", 
                "confidence": "Low", 
                "analysis": f"Failed to parse LLM response: {str(e)}. Raw: {response[:100]}"
            }

    def process_files(self, file_list):
        """
        Iterates through the provided sequence files and saves to _output.jsonl files.
        """
        for input_path in file_list:
            if not os.path.exists(input_path):
                print(f"[Classification] Skipping {input_path} (Not found)")
                continue

            output_path = input_path.replace("_coreferenced_sequence.jsonl", "_output.jsonl")

            print(f"\n[Classification] Processing: {os.path.basename(input_path)}")
            print(f"[Classification] Saving to:   {os.path.basename(output_path)}")

            output_data = []
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in tqdm(lines, desc="Narrative Inference"):
                item = json.loads(line)
                
                if "final_answer" not in item or item["final_answer"] is None:
                    anchor = item.get("anchor_text_events", [])
                    story_a = item.get("text_a_events", [])
                    story_b = item.get("text_b_events", [])
                    
                    decision = self._get_decision(anchor, story_a, story_b)
                    
                    item["sequence_answer"] = decision.get("closer_story")
                    item["confidence"] = decision.get("confidence")
                
                output_data.append(item)

            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in output_data:
                    f.write(json.dumps(entry) + '\n')
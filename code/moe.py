import json
import os
import sys
import torch
import re
import contextlib
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from google import genai
from google.genai import types

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def extract_decision(text):
    """Verifies if the output contains the required selection format."""
    if not text:
        return "Error"
    
    match = re.search(r"Selection:\s*(Text [AB])", text, re.IGNORECASE)
    if match:
        return match.group(1).title()
    
    clean_text = text.strip().split('\n')[-1]
    if "Text A" in clean_text and "Text B" not in clean_text:
        return "Text A"
    if "Text B" in clean_text and "Text A" not in clean_text:
        return "Text B"
    print(text)
    return "Error"


class NarrativeMoEPipeline:
    def __init__(self, gemma_path, hf_token, openai_key, google_key):
        """
        Initialize all models once when the class is instantiated.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_prompt = self._get_default_prompt()
        
        print(f"Initializing models on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(gemma_path, token=hf_token)
        self.gemma_model = AutoModelForCausalLM.from_pretrained(
            gemma_path, 
            token=hf_token,
            device_map="auto"
        )

        self.openai_client = OpenAI(api_key=openai_key)

        self.client = genai.Client(api_key=google_key)
        self.model_id = "gemini-2.5-flash"
        print("All models initialized successfully.")

    def _get_default_prompt(self):
        return """
You are an expert in narrative analysis. You are asked to identify narratively similar stories. 
We define Narrative similarity by three core similarity components: the abstract theme, the course of action, and the outcomes of a story.

We define these three aspects as follows:
- Abstract Theme: Describes the defining constellation of problems, central ideas, and core motifs of a story.
- Course of Action: Describes sequences of events, actions, conflicts, and turning points.
- Outcomes: Describe the results of the plot at the end of the text.

Your Task:
You will be provided with an "Anchor Story" and two candidate stories, "Text A" and "Text B".
Determine which candidate (Text A or Text B) is more narratively similar to the Anchor Story.

Output Format:
1. Provide a brief analysis (2-3 sentences).
2. End your response with a single line containing exactly: "Selection: Text A" or "Selection: Text B".
"""

    def query_gemma(self, user_message, n_times=5):
        results = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt", padding=True, return_attention_mask=True).to(self.device)

        for _ in range(n_times):
            with torch.no_grad(), suppress_stdout_stderr():
                generated_ids = self.gemma_model.generate(
                    model_inputs.input_ids,
                    attention_mask=model_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7 
                )
            
            gen_ids_trimmed = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            decoded = self.tokenizer.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]
            results.append(decoded.strip())
            
        return results

    def query_gpt(self, user_message, model_name="gpt-3.5-turbo-instruct", n_times=5):
        results = []
        for _ in range(n_times):
            try:
                with suppress_stdout_stderr():
                    resp = self.openai_client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        max_tokens=512,
                        temperature=0.7,
                    )
                results.append(resp.choices[0].message.content.strip())
                if resp.choices[0].finish_reason == 'content_filter':
                    print("CONTENT FILTER TRIGGERED")
                elif resp.choices[0].finish_reason == 'length':
                    print("MAX TOKENS REACHED")
                else:
                    pass
            except Exception as e:
                print(f"GPT Error: {e}")
                results.append("")
        return results

    def query_gemini(self, user_message, n_times=5):
        results = []
        
        config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=512,
            system_instruction=self.system_prompt,
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ]
        )

        for i in range(n_times):
            try:
                with suppress_stdout_stderr():
                    resp = self.client.models.generate_content(
                        model="gemini-2.5-flash", 
                        contents=user_message,
                        config=config
                    )

                if resp and resp.text:
                    results.append(resp.text.strip())
                else:
                    print(f"Iter {i}: No text returned. Finish reason: {resp.candidates[0].finish_reason if resp.candidates else 'No Candidate'}")
                    results.append("")

            except Exception as e:
                print(f"Gemini Error in Iter {i}: {e}")
                results.append("")
                
        return results

    def process_files(self, file_paths, output_suffix="_moe"):
        """
        Main processing loop.
        """
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Skipping {file_path}, not found.")
                continue

            print(f"Processing {file_path}...")
            
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))

            output_data = []

            for entry in tqdm(data, desc=f"Inference on {os.path.basename(file_path)}"):
                user_message = (
                    f"### Anchor Story\n{entry.get('anchor_text', '')}\n\n"
                    f"### Text A\n{entry.get('text_a', '')}\n\n"
                    f"### Text B\n{entry.get('text_b', '')}\n"
                )

                gemma_raw = self.query_gemma(user_message, n_times=5)
                for i, raw in enumerate(gemma_raw):
                    entry[f'gemma_{i+1}'] = extract_decision(raw)

                gpt_raw = self.query_gpt(user_message, model_name="gpt-4", n_times=5)
                for i, raw in enumerate(gpt_raw):
                    entry[f'gpt_{i+1}'] = extract_decision(raw)

                gemini_raw = self.query_gemini(user_message, n_times=5)
                for i, raw in enumerate(gemini_raw):
                    entry[f'gemini_{i+1}'] = extract_decision(raw)

                output_data.append(entry)

            base, ext = os.path.splitext(file_path)
            output_path = f"{base}{output_suffix}{ext}"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in output_data:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"Saved results to {output_path}")
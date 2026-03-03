import json
import os
import torch
import re
import logging
import transformers
from collections import Counter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from fastcoref import LingMessCoref

try:
    from transformers.models.longformer.modeling_longformer import LongformerModel
    def _force_eager_attention(self, config, attn_implementation=None, **kwargs):
        return "eager"
    LongformerModel._check_and_adjust_attn_implementation = _force_eager_attention
    print("[Coreference] Successfully patched LongformerModel compatibility.")
except ImportError:
    print("[Coreference] Warning: Could not patch LongformerModel. Transformers might be too old or broken.")
except Exception as e:
    print(f"[Coreference] Warning: Patch failed with error: {e}")

transformers.logging.set_verbosity_error()
logging.getLogger('fastcoref').setLevel(logging.ERROR)

class NarrativeCoreferenceSolver:
    def __init__(self, model_path, hf_token=None):
        self.model_path = model_path
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.llm_model = None
        self.llm_tokenizer = None
        self.coref_model = None

    def _load_resources(self):
        """Loads both FastCoref (Directly) and Gemma."""
        
        if self.coref_model is None:
            print(f"[Coref] Loading LingMessCoref on {self.device}...")
            self.coref_model = LingMessCoref(
                model_name_or_path='biu-nlp/lingmess-coref',
                device=self.device, 
                enable_progress_bar=False
            )

        if self.llm_model is None:
            print(f"[Coref] Loading Gemma ({self.model_path})...")
            try:
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.model_path, token=self.hf_token)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    device_map="auto", 
                    torch_dtype=torch.bfloat16,
                    token=self.hf_token
                )
            except Exception as e:
                print(f"Error loading Gemma: {e}")
                raise e

    def _ask_gemma_for_name(self, cluster_strings):
        """
        Uses Gemma to pick the most representative name from a list of mentions.

        """
        ignore_list = {'he', 'she', 'it', 'him', 'her', 'his', 'they', 'them', 'their', 'who', 'which', 'that', 'i', 'me', 'my', 'myself', 'himself', 'herself'}
        non_pronouns = [s for s in cluster_strings if s.lower() not in ignore_list]    

        if not non_pronouns:
            return cluster_strings[0]

        candidates_str = ", ".join([f'"{s}"' for s in set(non_pronouns)])

        system_prompt = "You are a text processing assistant."
        user_prompt = (
            f"Identify the best full proper name for a character referred to by these terms: [{candidates_str}].\n"
            f"Reply with ONLY the name inside double quotes (e.g., \"John Smith\")."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        text = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        with torch.no_grad():
            outputs = self.llm_model.generate(**inputs, max_new_tokens=20, do_sample=False)
       
        response = self.llm_tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

        match = re.search(r'["\'](.*?)["\']', response)
        if match:
            return match.group(1)

        return response.strip().strip('.')

    def _resolve_text(self, text):
        """
        Runs the coref model on a single text string and resolves entities.
        Handles nested spans by prioritizing the longest/outermost replacement.
        """
        if not text:
            return "", {}

        preds = self.coref_model.predict(texts=[text])
        
        clusters = preds[0].get_clusters(as_strings=False)
        
        if not clusters:
            return text, {}

        replacements = []
        entity_map = {}
        
        for cluster in clusters:
            cluster_strings = [text[start:end] for start, end in cluster]
            
            best_name = self._ask_gemma_for_name(cluster_strings)
            
            if best_name not in entity_map:
                entity_map[best_name] = set()
            entity_map[best_name].update(cluster_strings)

            for start, end in cluster:
                original_span = text[start:end]
                
                if original_span != best_name:
                    replacements.append({
                        "start": start,
                        "end": end,
                        "text": best_name,
                        "length": end - start
                    })

        final_replacements = self._filter_overlaps(replacements)

        final_replacements.sort(key=lambda x: x["start"], reverse=True)
        
        text_list = list(text)
        
        for r in final_replacements:
            start, end, name = r["start"], r["end"], r["text"]
            text_list[start:end] = list(name)
        
        resolved_text = "".join(text_list)
        
        final_map = {k: list(v) for k, v in entity_map.items()}

        return resolved_text, final_map

    def _filter_overlaps(self, replacements):
        """
        Removes nested or overlapping replacements. 
        Prioritizes the longest span (outermost).
        """
        if not replacements:
            return []

        replacements.sort(key=lambda x: (x["length"], -x["start"]), reverse=True)
        
        kept_indices = []
        
        for i, curr in enumerate(replacements):
            is_overlapping = False
            curr_start, curr_end = curr["start"], curr["end"]
            
            for kept in kept_indices:
                k_start = kept["start"]
                k_end = kept["end"]
                
                if (curr_start < k_end and curr_end > k_start):
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                kept_indices.append(curr)
                
        return kept_indices

    def process_files(self, file_paths):
        self._load_resources()

        for filepath in file_paths:
            if not os.path.exists(filepath):
                print(f"Skipping {filepath}, not found.")
                continue

            directory, filename = os.path.split(filepath)
            name, ext = os.path.splitext(filename)
            clean_name = name.replace('_output', '') 
            new_filename = f"{clean_name}_events_coreferenced{ext}"
            output_path = os.path.join(directory, new_filename)

            print(f"Coreferencing {filename} -> {new_filename}...")

            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))

            output_data = []
            skipped_count = 0
            
            for entry in tqdm(data, desc="Resolving Entities"):
                if 'final_answer' in entry:
                    output_data.append(entry)
                    skipped_count += 1
                    continue
                
                try:
                    if entry.get('anchor_text'):
                        res_text, res_map = self._resolve_text(entry['anchor_text'])
                        entry['anchor_text_coreferenced'] = res_text
                        entry['anchor_text_entity_map'] = res_map
                    
                    if entry.get('text_a'):
                        res_text, res_map = self._resolve_text(entry['text_a'])
                        entry['text_a_coreferenced'] = res_text
                        entry['text_a_entity_map'] = res_map
                        
                    if entry.get('text_b'):
                        res_text, res_map = self._resolve_text(entry['text_b'])
                        entry['text_b_coreferenced'] = res_text
                        entry['text_b_entity_map'] = res_map
                        
                except Exception as e:
                    print(f"Error resolving entry: {e}")

                output_data.append(entry)

            with open(output_path, 'w', encoding='utf-8') as f:
                for item in output_data:
                    f.write(json.dumps(item) + "\n")

            print(f"Saved {len(output_data)} entries. (Skipped {skipped_count} already solved)")
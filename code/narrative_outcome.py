import json
import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

class NarrativeOutcomeAnalyzer:
    def __init__(self, model_path, hf_token=None):
        self.model_path = model_path
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.tokenizer = None
        
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.embedding_model = None
        
        self.sentiment_model_name = "siebert/sentiment-roberta-large-english"
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        
        self.extraction_system_prompt = (
            "You are an expert narrative analyst. "
            "Your task is to read the story provided and extract the narrative outcome. "
            "Focus strictly on the final state of the protagonist and the resolution of the central conflict. "
            "Output a concise summary of 2-3 sentences."
        )
        
        self.classification_system_prompt = (
            "You are a literary classifier. "
            "Classify the provided outcome summary into exactly one of the following categories:\n"
            "- Total Victory\n"
            "- Compromised Success\n"
            "- Pyrrhic Victory\n"
            "- Noble Failure\n"
            "- Tragic Failure\n"
            "- No change\n"
            "- Ambiguous/Open\n\n"
            "Return ONLY the category name. Do not explain."
        )

    def _load_resources(self):
        """Loads all models explicitly onto the defined device."""
        if self.model is None:
            print(f"[Outcome] Loading LLM: {self.model_path}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, token=self.hf_token)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    token=self.hf_token,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
            except Exception as e:
                print(f"Error loading LLM: {e}")
                sys.exit(1)

        if self.embedding_model is None:
            print(f"[Outcome] Loading Embedding Model ({self.embedding_model_name})...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)

        if self.sentiment_model is None:
            print(f"[Outcome] Loading Sentiment Model ({self.sentiment_model_name})...")
            try:
                self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.sentiment_model_name)
                self.sentiment_model.to(self.device)
                self.sentiment_model.eval() 
            except Exception as e:
                print(f"Error loading Sentiment Model: {e}")
                sys.exit(1)

    def _generate_llm(self, system_prompt, user_text, max_tokens=128):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]
        prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([prompt_str], return_tensors="pt", padding=True, return_attention_mask=True).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_tokens,
                do_sample=False, 
            )
        gen_ids_trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0].strip()

    def _get_sentiment(self, text):
        if not text: return 0.0
        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            neg_prob = probs[0][0].item()
            pos_prob = probs[0][1].item()
            return pos_prob - neg_prob

    def _get_embedding(self, text):
        return self.embedding_model.encode(text, convert_to_tensor=False)

    def _calculate_cosine_distance(self, vec1, vec2):
        """Returns 1 - CosineSimilarity. Range [0, 1] (0 is identical)."""
        if vec1 is None or vec2 is None: return 1.0
        v1, v2 = np.array(vec1), np.array(vec2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 1.0
        
        cosine_sim = np.dot(v1, v2) / (norm1 * norm2)
        return float(1.0 - cosine_sim)

    def analyze_story_text(self, text):
        """Generates full analysis INCLUDING embedding."""
        if not text: return None
        summary = self._generate_llm(self.extraction_system_prompt, f"STORY: {text}\n\nOUTCOME SUMMARY:")
        classification = self._generate_llm(self.classification_system_prompt, f"SUMMARY: {summary}\n\nCLASSIFICATION:")
        sentiment_score = self._get_sentiment(summary)
        embedding = self._get_embedding(summary) 

        return {
            "summary": summary,
            "classification": classification,
            "sentiment": sentiment_score,
            "embedding": embedding
        }

    def process_files(self, file_paths):
        self._load_resources()

        for filepath in file_paths:
            if not os.path.exists(filepath):
                print(f"Skipping {filepath}, not found.")
                continue

            directory, filename = os.path.split(filepath)
            name, ext = os.path.splitext(filename)
            clean_name = name.replace('_output', '')
            new_filename = f"{clean_name}_outcome{ext}"
            output_path = os.path.join(directory, new_filename)

            print(f"Generating outcome metrics for {filename} -> {new_filename}...")

            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))

            output_data = []
            
            for entry in tqdm(data, desc="Analyzing Outcomes"):
                if 'final_answer' in entry:
                    output_data.append(entry)
                    continue

                try:
                    if 'anchor_outcome' not in entry:
                        entry['anchor_outcome'] = self.analyze_story_text(entry.get('anchor_text', ''))
                    if 'text_a_outcome' not in entry:
                        entry['text_a_outcome'] = self.analyze_story_text(entry.get('text_a', ''))
                    if 'text_b_outcome' not in entry:
                        entry['text_b_outcome'] = self.analyze_story_text(entry.get('text_b', ''))

                    vec_anchor = entry['anchor_outcome'].get('embedding')
                    vec_a = entry['text_a_outcome'].get('embedding')
                    vec_b = entry['text_b_outcome'].get('embedding')

                    dist_a = self._calculate_cosine_distance(vec_anchor, vec_a)
                    dist_b = self._calculate_cosine_distance(vec_anchor, vec_b)

                    entry['semantic_dist_a'] = dist_a
                    entry['semantic_dist_b'] = dist_b

                    if 'embedding' in entry['anchor_outcome']: del entry['anchor_outcome']['embedding']
                    if 'embedding' in entry['text_a_outcome']: del entry['text_a_outcome']['embedding']
                    if 'embedding' in entry['text_b_outcome']: del entry['text_b_outcome']['embedding']
                    
                except Exception as e:
                    print(f"Error processing entry: {e}")

                output_data.append(entry)

            with open(output_path, 'w', encoding='utf-8') as f:
                for item in output_data:
                    f.write(json.dumps(item) + "\n")

            print(f"Saved {len(output_data)} entries.")
import json
import os
import math
from tqdm import tqdm

class NarrativeOutcomeScorer:
    def __init__(self):
        self.W_CAT = 0.6
        self.W_SENT = 0.25
        self.W_SEM = 0.15

        self.CATEGORY_PENALTY_MAP = {
            ("Total Victory", "Total Victory"): 0.0,
            ("Total Victory", "Compromised Success"): 0.2,
            ("Total Victory", "No change"): 0.35,
            ("Total Victory", "Ambiguous/Open"): 0.45,
            ("Total Victory", "Pyrrhic Victory"): 0.6,
            ("Total Victory", "Noble Failure"): 0.8,
            ("Total Victory", "Tragic Failure"): 1.0,

            ("Compromised Success", "Compromised Success"): 0.0,
            ("Compromised Success", "No change"): 0.15,  
            ("Compromised Success", "Ambiguous/Open"): 0.25,
            ("Compromised Success", "Pyrrhic Victory"): 0.4,
            ("Compromised Success", "Noble Failure"): 0.6,
            ("Compromised Success", "Tragic Failure"): 0.8,

            ("No change", "No change"): 0.0,
            ("No change", "Ambiguous/Open"): 0.1,       
            ("No change", "Pyrrhic Victory"): 0.25,
            ("No change", "Noble Failure"): 0.45,
            ("No change", "Tragic Failure"): 0.65,

            ("Ambiguous/Open", "Ambiguous/Open"): 0.0,
            ("Ambiguous/Open", "Pyrrhic Victory"): 0.15, 
            ("Ambiguous/Open", "Noble Failure"): 0.35,
            ("Ambiguous/Open", "Tragic Failure"): 0.55,

            ("Pyrrhic Victory", "Pyrrhic Victory"): 0.0,
            ("Pyrrhic Victory", "Noble Failure"): 0.2,
            ("Pyrrhic Victory", "Tragic Failure"): 0.4,

            ("Noble Failure", "Noble Failure"): 0.0,
            ("Noble Failure", "Tragic Failure"): 0.2,

            ("Tragic Failure", "Tragic Failure"): 0.0
        }

    def _get_cat_distance(self, tag1, tag2):
        t1, t2 = tag1.strip(), tag2.strip()
        if t1 == t2: return 0.0
        if (t1, t2) in self.CATEGORY_PENALTY_MAP: return self.CATEGORY_PENALTY_MAP[(t1, t2)]
        if (t2, t1) in self.CATEGORY_PENALTY_MAP: return self.CATEGORY_PENALTY_MAP[(t2, t1)]
        return 1.0

    def _get_sent_distance(self, score1, score2):
        return abs(score1 - score2) / 2.0

    def calculate_total_score(self, cat_dist, sent_dist, sem_dist):
        return (self.W_CAT * cat_dist) + (self.W_SENT * sent_dist) + (self.W_SEM * sem_dist)

    def process_files(self, file_paths):
        for filepath in file_paths:
            if not os.path.exists(filepath):
                print(f"Skipping {filepath}, not found.")
                continue
            
            directory, filename = os.path.split(filepath)
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_output{ext}"
            output_path = os.path.join(directory, new_filename)

            print(f"Scoring outcomes in {filename} -> {new_filename}...")

            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))

            output_data = []
            processed_count = 0

            for entry in tqdm(data, desc="Calculating Scores"):
                if 'final_answer' not in entry and 'anchor_outcome' in entry:
                    
                    cat_a = self._get_cat_distance(
                        entry['anchor_outcome'].get('classification', ''),
                        entry['text_a_outcome'].get('classification', '')
                    )
                    cat_b = self._get_cat_distance(
                        entry['anchor_outcome'].get('classification', ''),
                        entry['text_b_outcome'].get('classification', '')
                    )

                    sent_a = self._get_sent_distance(
                        entry['anchor_outcome'].get('sentiment', 0.0),
                        entry['text_a_outcome'].get('sentiment', 0.0)
                    )
                    sent_b = self._get_sent_distance(
                        entry['anchor_outcome'].get('sentiment', 0.0),
                        entry['text_b_outcome'].get('sentiment', 0.0)
                    )

                    sem_a = entry.get('semantic_dist_a', 1.0)
                    sem_b = entry.get('semantic_dist_b', 1.0)

                    total_a = self.calculate_total_score(cat_a, sent_a, sem_a)
                    total_b = self.calculate_total_score(cat_b, sent_b, sem_b)

                    entry['outcome_metrics'] = {
                        'dist_cat_a': cat_a,
                        'dist_sent_a': sent_a,
                        'dist_sem_a': sem_a,
                        'total_score_a': total_a,
                        
                        'dist_cat_b': cat_b,
                        'dist_sent_b': sent_b,
                        'dist_sem_b': sem_b,
                        'total_score_b': total_b
                    }

                    if total_a < total_b:
                        entry['outcome_closer'] = "Text A"
                    elif total_b < total_a:
                        entry['outcome_closer'] = "Text B"
                    else:
                        entry['outcome_closer'] = "Tie"

                    processed_count += 1
                
                output_data.append(entry)

            with open(output_path, 'w', encoding='utf-8') as f:
                for item in output_data:
                    f.write(json.dumps(item) + "\n")

            print(f"Saved {len(output_data)} entries. (Scored: {processed_count})")
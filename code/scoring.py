import json
import os
from tqdm import tqdm

class NarrativeAggregator:
    def __init__(self):
        self.W_T = 1.0  # Abstract Theme
        self.W_A = 1.5  # Course of Action
        self.W_O = 1.2  # Outcomes

        # Confidence to Score mapping
        self.conf_map = {"High": 1.0, "Medium": 0.8, "Low": 0.6}

    def compute_aggregate(self, data):
        """
        Calculates the weighted aggregate distance.
        Lower distance wins.
        """
        # 1. Normalize Abstract Theme (T)
        t_winner = data.get('closer_abstract', "Story A")
        t_weight = self.conf_map.get(data.get('confidence_abstract'), 0.5)
        t_a = (1 - t_weight) if t_winner == "Story A" else t_weight
        t_b = (1 - t_weight) if t_winner == "Story B" else t_weight

        # 2. Normalize Course of Action (A)
        # Note: Step 9 outputs "A" or "B" in 'final_answer', 
        # but since we only run this if final_answer is missing, 
        # we check the intermediate 'sequence_answer' if available.
        a_winner_raw = data.get('sequence_answer', "A")
        a_winner = "Story A" if a_winner_raw == "A" else "Story B"
        a_weight = self.conf_map.get(data.get('confidence'), 0.5)
        a_a = (1 - a_weight) if a_winner == "Story A" else a_weight
        a_b = (1 - a_weight) if a_winner == "Story B" else a_weight

        # 3. Use Outcome Scores (O)
        outcomes = data.get('outcome_metrics', {})
        o_a = outcomes.get('total_score_a', 0.5)
        o_b = outcomes.get('total_score_b', 0.5)

        # 4. Calculate Aggregate Weighted Distance
        dist_a = (self.W_T * t_a) + (self.W_A * a_a) + (self.W_O * o_a)
        dist_b = (self.W_T * t_b) + (self.W_A * a_b) + (self.W_O * o_b)

        # 5. Determine Final Winner
        final_answer = "Story A" if dist_a <= dist_b else "Story B"
        
        return {
            "final_answer": final_answer,
            "calculated_metrics": {
                "score_a": round(dist_a, 3),
                "score_b": round(dist_b, 3)
            }
        }

    def process_files(self, file_list):
        for input_path in file_list:
            if not os.path.exists(input_path):
                print(f"[Aggregation] Skipping {input_path} (Not found)")
                continue

            # Naming convention as requested
            output_path = input_path.replace("_moe_abstract_outcome_events_output.jsonl", "_done.jsonl")

            print(f"\n[Aggregation] Final Fusion: {os.path.basename(input_path)}")
            
            output_data = []
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    
                    # Only compute if final_answer does not exist
                    if "final_answer" not in data or data["final_answer"] is None:
                        results = self.compute_aggregate(data)
                        data.update(results)
                    
                    output_data.append(data)

            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                for entry in output_data:
                    f.write(json.dumps(entry) + '\n')
            
            print(f"[Aggregation] Saved to: {output_path}")

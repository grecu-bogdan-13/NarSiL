import json
import os
from collections import Counter

class ConsensusAnalyzer:
    def __init__(self):
        pass

    def _get_majority(self, votes, threshold=4):
        """
        Returns the majority label if it appears at least 'threshold' times.
        Otherwise returns None.
        Votes is a list like ['Text A', 'Text A', 'Text B', ...]
        """
        valid_votes = [v for v in votes if v in ["Text A", "Text B"]]
        
        if not valid_votes:
            return None

        counts = Counter(valid_votes)
        most_common, count = counts.most_common(1)[0]
        
        if count >= threshold:
            return most_common
        return None

    def analyze_file(self, input_path, output_path):
        """
        Reads input_path, applies consensus logic, saves to output_path.
        Returns the count of entries that FAILED to get a final_answer.
        """
        if not os.path.exists(input_path):
            print(f"Skipping {input_path}, file not found.")
            return 0

        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        missing_consensus_count = 0
        processed_data = []

        for entry in data:
            gemma_votes = [entry.get(f'gemma_{i}', '') for i in range(1, 6)]
            gpt_votes = [entry.get(f'gpt_{i}', '') for i in range(1, 6)]
            gemini_votes = [entry.get(f'gemini_{i}', '') for i in range(1, 6)]

            gemma_res = self._get_majority(gemma_votes, threshold=4)
            gpt_res = self._get_majority(gpt_votes, threshold=4)
            gemini_res = self._get_majority(gemini_votes, threshold=4)

            if (gemma_res and gpt_res and gemini_res) and (gemma_res == gpt_res == gemini_res):
                entry['final_answer'] = gemma_res
            else:
                missing_consensus_count += 1
            
            processed_data.append(entry)

        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in processed_data:
                f.write(json.dumps(entry) + '\n')

        return missing_consensus_count

    def process_files(self, file_paths):
        print("Starting Consensus Analysis...")
        
        total_missing = 0
        
        for input_path in file_paths:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_output{ext}"
            
            print(f"Analyzing {os.path.basename(input_path)}...")
            missing_count = self.analyze_file(input_path, output_path)
            
            print(f"  -> Saved to {os.path.basename(output_path)}")
            print(f"  -> Entries missing final_answer: {missing_count}")
            total_missing += missing_count

        print(f"\nAnalysis Complete. Total entries without consensus across all files: {total_missing}")